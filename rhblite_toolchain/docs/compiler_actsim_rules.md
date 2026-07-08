# ACompiler / ACSim / ACTSim Rule Notes

This document records empirically verified constraints found while debugging
the ONNX -> ACompiler -> ACSim -> ACTSim pipeline. The goal is to keep compiler
and simulator rules explicit, with evidence and known workarounds.

## Status Levels

- `verified`: Reproduced with local test models and logs.
- `likely`: Strong evidence, but root cause still needs source-level
  confirmation.
- `hypothesis`: Plausible explanation that still needs a focused test.

## Rule 1: Prefer 1x1 Conv Forms for Patch Embedding in ACTSim

Status: `verified`

Direct patch embedding forms such as:

```text
Conv2d(3, 64, kernel_size=2, stride=2)
Conv2d(3, 64, kernel_size=4, stride=4)
Conv2d(3, 64, kernel_size=16, stride=16)
```

can compile and can pass ACSim, but they do not reliably close in ACTSim.
Failures appear in the CTC Conv / RRAM weight loading or dependent scheduling
path, typically with `Segmentation fault` or `timeout: the monitored command
dumped core`.

Validated failure cases:

- `models/vit_patch_test/failure/direct_p2.py`
- `models/vit_patch_test/failure/direct_p2_c4.py`
- `models/vit_patch_test/failure/direct_p4.py`
- `models/vit_patch_test/failure/direct_p16.py`

Validated ACTSim-friendly alternatives:

- `models/vit_patch_test/exact_p2_packed1x1.py`
- `models/vit_patch_test/exact_p4_packed1x1.py`
- `models/vit_patch_test/approx_k1_s2.py`
- `models/vit_patch_test/approx_k1_s2_c4.py`

Workarounds:

- Exact behavior: perform patchify / space-to-depth outside ACTSim, then use
  `Conv2d(patch_area * input_channels, output_channels, kernel_size=1,
  stride=1)`.
- Approximate behavior: use `Conv2d(input_channels, output_channels,
  kernel_size=1, stride=2)`. This is hardware-friendly but not mathematically
  equivalent to standard patch embedding, because it samples one pixel per
  stride window.

Evidence:

- `output_vit_patch_matrix_success/`
- `output_vit_patch_matrix_fail/`
- `output_vit_patch_small/`
- `run_vit_patch_matrix.sh`
- `run_patch_embed_sims.sh`

## Rule 2: EPU S-ADV Const Parameters Must Use 16-bit Param Width

Status: `verified`

For learned constant elementwise operations such as:

```python
x = x + self.pos_embed
```

ACompiler can lower the operation to an EPU `ALU_S_ADV` path. ACSim requires
the S-ADV parameter to be 16-bit. If the constant is quantized as 8-bit,
ACSim fails with:

```text
EPU ALU_S_ADV compute
param_width = 0
AssertionError: param_width should be 16 bits when alu mode is S-ADV
```

Fix:

Add the relevant constant / tensor names to `tensor_16` in the model file.
For the ViT position embedding probe this fixed ACSim:

```python
tensor_16 = [
    "pos_embed",
    "/Add_output_0",
    "/fc2/MatMul_output_0",
]
```

Validated case:

- `models/vit_patch_test/failure/vit_tiny_approx_ffn_k1s2_pos16.py`

Comparable existing working pattern:

- `models/epu_test/test_2.py`

Important distinction:

- This fixes ACSim for the position embedding const Add.
- It does not by itself guarantee ACTSim closure. ACTSim can still fail later
  due to CTC/package/scheduling constraints.

Evidence:

- `output_vit_tiny_approx_ffn_k1s2_pos16/`
- Existing `epu_test.test_2` uses explicit `tensor_16` entries and succeeds.

## Rule 3: Avoid Non-aligned Small Final Output Channels at the End of a Long CTC Chain

Status: `verified symptom`, `likely root cause`

The following chains close in ACTSim:

```text
3 -> 64 -> 128 -> 64
3 -> 64 -> 128 -> 64 -> 64
3 -> 64 -> 10
3 -> 64 -> 128 -> 64 -> 16
3 -> 64 -> 128 -> 64 -> 32
```

The following chain does not close:

```text
3 -> 64 -> 128 -> 64 -> 10
```

The failure is not caused by `64 -> 10` alone, because the shorter
`3 -> 64 -> 10` case succeeds. The failure is triggered by using a small,
non-16-aligned final output channel count after a longer high-channel CTC
chain.

Observed failing output config:

```text
Output Address: 64 (0x40)
Size: 4096
Layout: NCHW
Dims: [1, 10, 16, 16]
c_align: 1
w_align: 1
ctc_size: 34
```

Validated replacement configs:

```text
Dims: [1, 16, 16, 16]  # ACTSim closes
Dims: [1, 32, 16, 16]  # ACTSim closes
```

Workaround:

- Replace final `Conv2d(64, 10, kernel_size=1)` with `Conv2d(64, 16,
  kernel_size=1)` or `Conv2d(64, 32, kernel_size=1)`.
- Copy the original 10 output-channel weights into the first 10 channels.
- Set the extra output channels to zero or ignore them in host-side
  post-processing.

Validated cases:

- Fails: `models/vit_patch_test/failure/vit_tiny_approx_ffn_k1s2_nores.py`
- Passes: `models/vit_patch_test/k1s2_ffn_head16.py`
- Passes: `models/vit_patch_test/k1s2_ffn_head32.py`
- Passes: `models/vit_patch_test/k1s2_then_head.py`
- Passes: `models/vit_patch_test/k1s2_then_3x1_64.py`

Evidence:

- `output_vit_head_align_check/`
- `output_vit_actsim_min/`

## Known Non-fatal Noise

Status: `verified`

ACSim logs often contain:

```text
YAML:15:1: error: unknown key 'ParamFromReRAM'
```

This appears in many existing successful models and is not the direct cause of
ACTSim failure. It is an ACSim arch YAML parser compatibility issue with
`arch_16.yaml` / `arch_256.yaml`.

Evidence:

- Many models in `output_opcheck_2026-06-16/` show this message while still
  completing ACSim and ACTSim.

## Candidate Rule 4: Mixed CTC + Token/EPU Residual Graphs Need Separate ACTSim Validation

Status: `verified symptom`

Padding the final head from 10 channels to 16 channels fixes the pure CTC
chain case, but it does not by itself make the ViT-style position embedding
probe close in ACTSim.

The following token-only patterns were validated:

```text
Add                                      # closes
FFN                                      # closes
FFN -> residual Add                      # closes
pos Add as standalone graph output       # closes
```

Validated closing cases:

- `models/vit_patch_test/token_residual_add.py`
- `models/vit_patch_test/token_ffn_nopos.py`
- `models/vit_patch_test/token_ffn_residual.py`
- `models/vit_patch_test/token_pos_input_add.py`

The following composed patterns are not currently safe:

```text
learned const pos Add -> FFN -> residual Add
pos input Add -> FFN -> residual Add
Linear -> Add -> Linear -> residual Add
FFN -> residual Add -> pos input Add
```

Observed failures:

- `models/vit_patch_test/failure/token_pos_ffn_residual.py`
  - ACompiler and ACSim pass.
  - ACTSim aborts at `op_idx = 0`:
    `no such op type = EltwiseConstArith`, followed by
    `free(): invalid pointer`.
- `models/vit_patch_test/failure/token_pos_input_ffn_residual.py`
  - ACompiler and ACSim pass.
  - ACTSim aborts at `op_idx = 0`:
    `no such op type = Add`, followed by `free(): invalid pointer`.
- `models/vit_patch_test/failure/token_ffn_mid_pos_input_residual.py`
  - ACompiler does not generate a complete instruction package.
  - Compile log reports `Unsupported node type PiecewiseActFun`.
- `models/vit_patch_test/failure/token_ffn_residual_pos_output.py`
  - ACompiler aborts in `setEltOpIO` with:
    `not supported other bitdepth in eltwise op`.

Practical workaround:

- Treat `pos_embed` Add as a stage boundary.
- Run `pos` as a standalone Add stage or perform it on the host.
- Feed the resulting token tensor into the verified `token_ffn_residual.py`
  stage.

Likely rule:

- Current ACTSim/compiler packaging tolerates Add/EPU as a final output or
  standalone graph.
- Add/EPU feeding later CTC/Linear ops is not reliable and must be validated
  case by case.

## Rule 6: ViT Attention and MLP Need Separate Stage Boundaries

Status: `verified symptom`

The token MLP block can close as an independent ACTSim stage:

```text
Linear -> Hardswish -> Linear -> residual Add
```

Validated case:

- `models/vit_patch_test/token_ffn_residual.py`

The pure attention core can also close as an independent stage when written in
the same style as `net_test.attn_sep`:

```text
Q/K/V projection -> QK MatMul -> SoftMax -> AV MatMul -> Concat
```

Validated case:

- `models/vit_patch_test/token_attn_sep.py`

However, the following standard ViT compositions do not currently close:

```text
attention -> output projection
attention -> residual Add
attention -> residual Add -> MLP -> residual Add
```

Observed failures:

- `models/vit_patch_test/failure/token_attn_sep_proj.py`
  - ACompiler and ACSim pass.
  - ACTSim core dumps at startup.
- `models/vit_patch_test/failure/token_attn_sep_residual.py`
  - ACompiler and ACSim pass.
  - ACTSim core dumps at startup.
- `models/vit_patch_test/failure/token_attn_mlp_residual.py`
  - ACompiler and ACSim pass.
  - ACTSim core dumps at startup.

Practical workaround:

- Treat attention core as one ACTSim stage.
- Treat MLP residual as a separate ACTSim stage.
- Keep attention output projection and residual outside this stage until a
  smaller validated pattern is found or ACTSim/compiler packaging is fixed.

## Rule 7: Cdist / Squared Distance Is Not Currently a Closed ACTSim Pattern

Status: `verified`

The direct squared difference form:

```python
diff = x1 - x2
out = diff * diff
```

exports to:

```text
Sub, Mul
```

Current ACompiler lowering rejects `onnx.Sub`:

```text
loc("/Sub"): error: failed to legalize operation 'onnx.Sub' that was explicitly marked illegal
```

Validated failing case:

- `models/vit_patch_test/failure/token_elem_square_diff.py`

The PyTorch cdist form:

```python
torch.cdist(x1, x2, p=2) ** 2
```

exports to:

```text
Unsqueeze, Unsqueeze, Sub, Pow, ReduceSum, Pow, Pow
```

It fails in ACompiler around `Pow` / layout handling:

```text
There are more than two W or C in XLayoutStr
loc("/Pow"): error: There are more than two W or C in XLayoutStr
```

Validated failing case:

- `models/vit_patch_test/failure/token_cdist_square.py`

The common algebraic rewrite:

```python
x1_sq = (x1 * x1).sum(-1, keepdim=True)
x2_sq = (x2 * x2).sum(-1).unsqueeze(1)
cross = x1 @ x2.transpose(-2, -1)
out = x1_sq + x2_sq - 2 * cross
```

also fails today. With `Sub`, it fails at the illegal `onnx.Sub`. Without
`Sub`, using:

```python
out = x1_sq + x2_sq + (-2.0 * cross)
```

it still fails during ACompiler verification:

```text
loc("/Add"): error: 'achigh.Add' op requires at least one operand same as results
```

Validated failing cases:

- `models/vit_patch_test/failure/token_pairwise_sqdist_manual.py`
- `models/vit_patch_test/failure/token_pairwise_sqdist_no_sub.py`

One partial workaround exists for elementwise squared difference:

```python
diff = x1 + (-1.0 * x2)
out = diff * diff
```

This avoids `onnx.Sub` and reaches ACompiler + ACSim:

```text
Mul, Add, Mul
```

but it still fails ACTSim with a startup core dump, so it is not a closed
deployment pattern.

Validated partial case:

- `models/vit_patch_test/failure/token_elem_square_diff_no_sub.py`

Practical workaround:

- Compute cdist / squared distance outside ACTSim for now.
- If distance-like behavior is needed on chip, reformulate it as validated
  MatMul/attention-style stages and avoid `Sub`, `Pow`, `ReduceSum`, and
  broadcasted Add patterns until they are individually validated.

## Rule 5: Treat Img2Mat Tokenization as an ACTSim Boundary

Status: `verified symptom`

The standard ViT patch embedding tail:

```python
x = x.flatten(2).transpose(1, 2)
```

is not always a no-op for the compiler. Some cases are folded into the output
metadata and can close in ACTSim, while others lower to an explicit `Img2Mat`
DMU op.

Validated failure:

- `models/vit_patch_test/failure/standard_patch_exact_p4_packed1x1_tokens.py`

This model uses external packed patch input followed by:

```text
1x1 Conv -> flatten(2) -> transpose(1, 2)
```

ACompiler and ACSim pass, but ACTSim fails at startup. The generated
`Img2Mat` YAML lacks complete ACTSim scheduling metadata, and ACTSim reports an
invalid-looking `op_idx` before core dump.

Validated related success:

- `models/vit_patch_test/vit_patch_flatten_only.py`

In this case the compiler output contains only the Conv op, so the
flatten/transpose is effectively handled at the graph boundary.

Workaround:

- Keep tokenization at a stage boundary when possible.
- For a single ACTSim graph, prefer NCHW 1x1 Conv pipelines and convert to
  `[B, N, D]` outside ACTSim.

Validated result:

- Modified `models/vit_patch_test/failure/vit_tiny_approx_ffn_k1s2_pos16.py` from
  `Linear(64, 10)` to `Linear(64, 16)`.
- ACompiler succeeds.
- ACSim succeeds and reports `ThreadID 0 completed the simulation.`
- The generated output config is padded:

```text
Dims: [1, 16, 16, 16]
ctc_size: 34
```

- ACTSim still fails at startup with `timeout: the monitored command dumped
  core`.

This means the previous final-head channel rule is necessary for the pure CTC
chain, but not sufficient for a graph that also includes token layout
conversion, `EltwiseConstArith` position embedding, and residual `Add`.

Observed graph:

```text
Conv -> EltwiseConstArith(pos_embed) -> Conv -> Conv -> Add -> Conv(head16)
```

Current suspected cause:

- The generated ACTSim case YAML lacks normal scheduling fields such as
  `op_idx`, `l2_input_buf`, and `cluster_idx`.
- ACTSim crashes before useful op execution logs, so the failure is likely in
  startup/package/scheduling metadata rather than in CTC arithmetic.

Evidence:

- `output_vit_pos16_headpad_check/`

Additional repair attempts:

- `models/vit_patch_test/failure/vit_tiny_approx_ffn_k1s2_pos_nchw_head16.py`
  keeps position embedding in NCHW constant form. ACompiler rejects the const
  Add because the constant is tagged as layout `"4D"` instead of an accepted
  `"NCHW"` const layout.
- `models/vit_patch_test/failure/vit_tiny_approx_ffn_k1s2_pos_input_head16.py`
  makes position embedding a second NCHW input. ACompiler and ACSim pass, but
  ACTSim still crashes at startup with an invalid-looking `op_idx`.
- `models/vit_patch_test/failure/vit_tiny_approx_ffn_k1s2_pos16_nohead.py`
  removes the final head. ACompiler and ACSim pass, but ACTSim still crashes.
  This confirms the current ACTSim blocker is before the final head.
- `models/vit_patch_test/failure/vit_tiny_approx_ffn_k1s2_pos16_nores_nohead.py`
  also removes the final residual Add. ACompiler and ACSim pass, but ACTSim
  still crashes. This narrows the blocker further to the learned
  `pos_embed` const Add / `EltwiseConstArith` path.

Refined ablation result:

- `models/vit_patch_test/token_pos_const_only.py`: pure token
  `EltwiseConstArith(pos_embed)` closes in ACTSim.
- `models/vit_patch_test/vit_patch_pos_only.py`: patch Conv followed by token
  `pos_embed` const Add closes in ACTSim.
- `models/vit_patch_test/vit_patch_flatten_only.py`: patch Conv followed by
  flatten/transpose output closes in ACTSim.
- `models/vit_patch_test/token_ffn_nopos.py`: pure token FFN closes in
  ACTSim.
- `models/vit_patch_test/vit_patch_fc1_only.py`: patch Conv followed by one
  token Linear closes in ACTSim.
- `models/vit_patch_test/failure/vit_patch_ffn_nopos_nores_nohead_i8.py`: patch Conv
  followed by two token Linear layers fails in ACTSim, even without
  `pos_embed`, residual Add, final head, or `tensor_16`.

This narrows the current issue to the single-graph transition:

```text
image CTC Conv -> token Linear FFN with two Linear layers -> NCHW output
```

The failed case has incomplete ACTSim YAML metadata. In particular, generated
per-op YAML files do not consistently receive:

```text
op_idx
l2_input_buf
l2_output_buf
cluster_idx
```

The corresponding profile also contains an extra trailing `DUMMY` item, unlike
the hand-written Conv2d chain that closes in ACTSim.

Current safer workaround:

- Keep patch embedding and FFN/head in the ACTSim graph.
- For a single ACTSim graph, write FFN as NCHW `Conv2d(..., kernel_size=1)`
  instead of flattening to token layout and using `Linear`.
- If token layout is required, split the pipeline at the token boundary:
  stage 1 `patch + pos` can close in ACTSim, and stage 2 pure token FFN can
  close in ACTSim.

## Rule 8: Map Official CSPN Propagation to Fixed Shift-Sum Conv Stages

The official-style CSPN propagation pads/shifts both `guidance` and `depth`,
multiplies on the padded canvas, sums over 8 directions, then crops back to the
valid region:

```python
depth_pad = cat([pad_one(depth, k) for k in range(8)], dim=1)
gate_pad = cat([pad_one(gate8[:, k:k + 1], k) for k in range(8)], dim=1)
neighbor = (depth_pad * gate_pad).sum(dim=1)[:, :, 1:-1, 1:-1]
depth = center * raw_depth + neighbor
```

For the no-normalization propagation step, this is equivalent to multiplying in
the unpadded image and applying a fixed 3x3 Conv2d over the 8 direction
channels:

```python
weighted_depth = depth * gate8
neighbor = fixed_shift_sum_conv(weighted_depth)
depth = raw_depth * center + neighbor
```

The fixed convolution uses official direction order:

```text
left_top, center_top, right_top,
left_center, right_center,
left_bottom, center_bottom, right_bottom
```

with one nonzero weight per direction:

```python
coords = [
    (2, 2), (2, 1), (2, 0),
    (1, 2),         (1, 0),
    (0, 2), (0, 1), (0, 0),
]
```

Validated equivalence:

- `tools/compare_cspn_official_shift_golden.py`
- Result: `max_abs=4.7683716e-07`, `mean_abs=8.1956387e-08`

Validated ACompiler + ACSim stages:

- `models/cspn_test/cspn_official_one_step_shift_conv.py`
  - one official shift-sum propagation step without sparse anchoring
  - ACompiler passes
  - ACSim reports `ThreadID 0 completed the simulation`
- `models/cspn_test/cspn_official_prop1_shift_conv_anchor.py`
  - one official shift-sum propagation step with official anchor mode
  - `depth = inv_mask * propagated + mask * raw_depth`
  - ACompiler passes
  - ACSim reports `ThreadID 0 completed the simulation`
- `models/cspn_test/cspn_official_prop2_shift_conv_no_anchor.py`
  - two unrolled propagation steps without anchoring
  - ACompiler passes
  - ACSim reports `ThreadID 0 completed the simulation`

Known failing combined form:

- `models/cspn_test/cspn_official_prop2_shift_conv_anchor.py`
  - two unrolled propagation steps with per-step official anchoring in one graph
  - ACompiler aborts during codegen:

```text
F... Util.cpp:469] Check failed: shift >= -1
Aborted (core dumped)
```

Current safer deployment rule:

- Keep affinity normalization outside the compiled graph, or train/export the
  head to provide normalized `gate8` and `center` directly.
- Use `cspn_official_prop1_shift_conv_anchor` as the repeatable compiled CSPN
  stage.
- For `prop_time > 1`, loop this one-step stage from host/runtime instead of
  placing all anchored recurrent steps in one ACompiler graph.
- This keeps the official anchor semantics exactly: sparse-mask positions are
  restored to `raw_depth_input`, not to `sparse_depth`.

## Rule 9: CSPN Gate8 Normalization Is Not Yet Fully In-Graph

Official CSPN gate normalization is:

```python
gate_wb = pad_guidance_8(guidance)
abs_weight = gate_wb.abs().sum(dim=1, keepdim=True)
gate_wb = gate_wb / abs_weight
gate_sum = gate_wb.sum(dim=1)[:, :, 1:-1, 1:-1]
center = 1.0 - gate_sum
```

The directional pad/shift part can be expressed as fixed sparse convolutions.
Validated compiler stages:

- `models/cspn_test/cspn_gate_shift_only.py`
  - fixed `Conv2d(8, 8, 3, padding=1)` for directional shift
  - ACompiler passes
  - ACSim reports `ThreadID 0 completed the simulation`
- `models/cspn_test/cspn_gate_sum_only.py`
  - fixed `Conv2d(8, 1, 3, padding=1)` for shifted 8-channel sum
  - ACompiler passes
  - ACSim reports `ThreadID 0 completed the simulation`

Current failing pieces:

- Direct `torch.abs` lowers to ONNX `Abs`, then fails:

```text
loc("/Abs"): error: Op onnx.Abs has no layout attr.
```

- Standalone NCHW/CHW `torch.relu` or `Conv2d + ReLU` used to emulate abs via
  `relu(x) + relu(-x)` fails during layout/type processing:

```text
llvm::cast<mlir::RankedTensorType> ... argument of incompatible type
Aborted (core dumped)
```

- NCHW/CHW EPU `Div` fails even without broadcasting:

```text
F... ACPCUtils.cpp:51] Check failed: inputLayout[3] == 'C'
Input layout should be NDWC or NDCC
Aborted (core dumped)
```

Practical rule:

- Use compiled fixed-conv stages for shift/sum propagation.
- Keep `abs + sum + div` gate normalization on host, or make the network output
  already-normalized `gate8` and `center`.
- Revisit in-graph normalization only if the graph can be rewritten into a
  compiler-supported NDWC/NDCC EPU layout.

## Rule 10: Remaining CSPN Official Ops Not Yet Aligned

Besides gate8 normalization, the official CSPN reference still has several
non-aligned or host-side pieces:

- `GroupNorm` in `ConvGNReLU`
  - current compiler-aligned backbone replaces it with `Conv2d + ReLU`
  - this changes checkpoint/semantic compatibility unless the model is trained
    or calibrated with the aligned block
- Runtime sparse-mask creation
  - official code uses `sparse_depth.sign()` when `sparse_mask is None`
  - current compiled propagation expects `mask` and `inv_mask` as explicit inputs
- Direct official padding path
  - `F.pad + unsqueeze + cat` creates a 5D gate/depth canvas
  - direct compilation is not the chosen path; use fixed sparse conv instead
- In-graph recurrent anchored loop
  - one anchored step passes
  - two anchored steps unrolled in one graph fails codegen with `shift >= -1`
  - loop `cspn_official_step_shift_conv_anchor` from host for `prop_time > 1`
- Optional positive-depth branch
  - `F.softplus(coarse_depth) + min_depth` is not part of the validated aligned
    pipeline; default official-style path keeps `positive_depth=False`
- Dynamic input checks and optional arguments
  - Python `if`, `ValueError`, optional `sparse_mask` behavior are reference-side
    control flow and should be resolved before compiler export

Validated aligned pieces to keep in the compiler path:

- fixed CH4 input backbone producing `raw_depth` and `raw_guidance`
- bilinear upsample at fixed scale in the aligned backbone
- channel concat inside the aligned backbone skip connections
- fixed conv shift/sum representation for CSPN propagation
- one-step official anchor formula with explicit `mask` and `inv_mask` inputs

## Rule 11: Avoid Direct ConvTranspose2d / Deconv

A minimal `nn.ConvTranspose2d` test does not compile directly:

- `models/cspn_test/failed_deconv_compiler/deconv_stride2_min.py`

The ONNX/layout pass reports missing layout attributes around the deconv node:

```text
loc("/deconv/ConvTranspose"): error: Op onnx.ReverseSequence has no layout attr.
loc("/deconv/ConvTranspose"): error: Op onnx.Transpose has no layout attr.
loc("/deconv/ConvTranspose"): error: Op onnx.Pad has no layout attr.
loc("/deconv/ConvTranspose"): error: Op onnx.Concat has no layout attr.
loc("/deconv/ConvTranspose"): error: Op onnx.Split has no layout attr.
...
ArrayRef ... Assertion `Index < Length && "Invalid index!"' failed.
Aborted (core dumped)
```

Practical rule:

- Do not use direct `ConvTranspose2d` in compiler-aligned models.
- Prefer explicit fixed `Upsample` / `Resize` followed by normal `Conv2d`.
- For models already trained with deconv, consider retraining or fine-tuning with
  `Resize + Conv`, because it is not generally weight-equivalent to deconv.
- Exact deconv emulation would require zero insertion, padding/cropping, kernel
  flip, and normal convolution; these intermediate ops are not currently a clean
  compiler path.

## Rule 12: CompletionFormer Backbone Adaptation

Directly compiling the original CompletionFormer-style backbone is not currently
a clean path because it combines several unsupported or risky pieces:

- `ConvTranspose2d` / deconv
- `BatchNorm2d` unless folded into Conv offline
- PVT transformer backbone
- CBAM-style attention inside `BasicBlock`
- optional confidence `Sigmoid` branch
- large 3x3 high-channel decoder convolutions

Validated compiler-aligned experiment:

- `models/completionformer_test/backbone_aligned_rgbd_tiny.py`
  - fixed RGB-D inputs: `rgb [3,32,32]`, `depth [1,32,32]`
  - outputs: `init_depth [1,1,32,32]`, `guide [1,8,32,32]`
  - uses `Upsample + Conv2d` instead of deconv
  - uses Conv/ReLU residual blocks instead of `BasicBlock + CBAM`
  - uses a fixed-shape Conv feature pyramid instead of full PVT
  - ACompiler generates `*.op_insts.bin`
  - ACSim reports `ThreadID 0 completed the simulation`

Known failing full-channel experiment:

- `models/completionformer_test/failed_full_channels/backbone_aligned_rgbd.py`
  - fails during parameter packing because high-channel 3x3 Conv weights exceed
    RM/weight-buffer capacity:

```text
The buffer size is larger than rm size 4096 x 16 x 16
```

Practical rule:

- First validate a reduced-channel compiler-aligned backbone.
- For larger channel counts, split the model by stage or replace high-channel
  3x3 decoder Conv with smaller blocks such as `1x1 -> 3x3 -> 1x1`.
- Treat full PVT attention and confidence gating as separate staged compiler
  problems, not part of the first backbone compile target.

## Rule 13: Strict PVT/JCAT Needs Staged Adaptation

The initial Conv-only `PVTAlignedStub` is not semantically equivalent to real
CompletionFormer PVT/JCAT. A stricter decomposition was tested with PVT-like
micro modules under `models/completionformer_test/`:

- `pvt_patch_embed_nonorm.py`
  - patch Conv + flatten/transpose passes ACompiler + ACSim
- `pvt_patch_embed_ln.py`
  - with `op_version = 18`, PatchEmbed + LayerNorm lowers to `achigh.LayerNorm` and passes ACSim
  - with `op_version = 14`, LayerNorm decomposes and fails around ONNX `Sub` legalization
- `pvt_mlp_gelu.py`
  - Linear + GELU + Linear still fails with `op_version = 18`; GELU lowers via `onnx.Erf` and compiler tuning/codegen stops with `list index out of range`
- `pvt_sra_attention_sr2.py`
  - official fused KV reshape/slice path fails layout rewrite for high-rank transpose
- `pvt_sra_attention_sep_kv_sr2.py`
  - split K/V equivalent generates `op_insts.bin` with `op_version = 18`, but ACSim still fails at runtime reshape/layout
- `pvt_cbam_basicblock.py`
  - CBAM fails due `ReduceMax`/Concat/Sigmoid-style attention lowering problems

Practical rule:

- Do not treat a Conv-only pyramid as a semantic PVT replacement. It is only a
  topology feasibility probe.
- For a ref-compatible path, decompose PVT into stages and validate each stage.
- Use separate K/V projections instead of `kv.reshape(...)[0], kv[1]`; weights
  can be split from the original `kv` Linear.
- Keep PVT LayerNorm, GELU MLP, SRA sr-conv attention, and CBAM as separate
  compiler targets until each reaches ACompiler + ACSim closure.

## Debug Checklist

When adding a new model, validate in this order:

1. ONNX export succeeds.
2. ACompiler generates `*.op_insts.bin`.
3. ACSim reports `ThreadID 0 completed the simulation.`
4. ACTSim reports `ACTSim Ends`.
5. If ACTSim fails, compare:
   - final output channel count and physical alignment,
   - generated per-op YAML `op_idx`,
   - `ctc_size`,
   - RRAM load progress,
   - dependency waits such as `waiting for the resources from pre_op`.
