# SDFormer HW128 Host/RHB Adaptation

This note records the current AgentFlow mapping for
`/root/demo/SDformer-for-Depth-Completion`.

## Current Status

SDFormer is not a direct full-graph RHB offload target. The model mixes
board-friendly Conv blocks with Host-oriented transformer glue:

- Conv stem, pointwise Conv, down/up pre-shuffle Conv, and selected final-head
  split Conv can run on RHB.
- LayerNorm, window reshape, L2 normalization, attention matmul/softmax,
  PixelShuffle/PixelUnshuffle, residual/concat/crop glue, GELU, tensor multiply,
  and strict depthwise group Conv stay on Host.

The upstream repository currently has no checkpoint in the local tree, so this
work validates deployability and board boundaries rather than task accuracy.

The current implementation has two layers of validation:

- board-proven hand probes for representative RHB subgraphs;
- generated compile/CModel matrix coverage for all SDFormer HW128 Conv
  candidate shapes used by the partition runner.
- a retrain-required hardware-aligned model and deployment-plan exporter that
  converts the unsupported window-attention/FFN cores into board-proven
  Conv3x3 mixer contracts.

## Board-Proven RHB Subgraphs

Probe root:

```text
/root/demo/artifacts/output_sdformer_probe_20260717
```

| Submodel | Shape | Board |
| --- | --- | --- |
| `sdformer_test.stem_rgb_relu_128` | `3 -> 18`, `128x128`, Conv3x3+ReLU | `All same: True` |
| `sdformer_test.stem_dep_relu_128` | `1 -> 6`, `128x128`, Conv3x3+ReLU | `All same: True` |
| `sdformer_test.down_conv_l1_128` | `24 -> 12`, `128x128`, Conv3x3 | `All same: True` |
| `sdformer_test.up_conv_l4_16` | `192 -> 384`, `16x16`, Conv3x3 | `All same: True` |
| `sdformer_test.attn_qkv_only_l1_128` | `24 -> 72`, `128x128`, Conv1x1 | `All same: True` |
| `sdformer_test.attn_project_l1_128` | `24 -> 24`, `128x128`, Conv1x1 | `All same: True` |
| `sdformer_test.ffn_project_out_l1_128` | `69 -> 24`, `128x128`, Conv1x1 | `All same: True` |
| `sdformer_test.final_head_chunk16_padded8_128` | final head exact IC chunk | `All same: True` |

## Failed / Host Fallback Boundaries

| Submodel | Result | Decision |
| --- | --- | --- |
| `sdformer_test.attn_qkv_dw_l1_128` | CModel pass, board `All same: False` | split qkv 1x1 to RHB; depthwise on Host |
| `sdformer_test.attn_dw_only_72_128` | CModel pass, board `All same: False` | depthwise group Conv Host fallback |
| `sdformer_test.ffn_project_in_dw_l1_128` | CModel pass, board `All same: False` | split project_in; depthwise on Host |
| `sdformer_test.ffn_dw_only_138_128` | CModel pass, board `All same: False` | depthwise group Conv Host fallback |
| `sdformer_test.ffn_project_in_only_l1_128` | CModel pass, board `All same: False` | output-channel split |
| `sdformer_test.ffn_project_in_padded144_l1_128` | CModel pass, board `All same: False` | padding alone is insufficient |
| `sdformer_test.final_head_relu_128` | compile/CModel generated package cannot parse op insts | use exact IC split |
| `sdformer_test.final_head_relu_padded8_128` | compiler assertion / CModel parse failure | use exact IC split |

## Generated Conv Matrix

AgentFlow now includes a generator for the full SDFormer HW128 Conv candidate
set:

```text
agentflow_rhb/scripts/generate_sdformer_hw128_probe_modules.py
```

It writes deterministic probe modules under:

```text
/root/demo/models/sdformer_test/generated
```

The generated set covers:

- RGB/depth stem Conv+ReLU;
- downsample pre-`PixelUnshuffle` Conv;
- upsample pre-`PixelShuffle` Conv;
- decoder reduce-level Conv1x1;
- attention qkv/project Conv1x1 for `C=24,48,96,192,72`;
- FFN `project_in` exact output-channel split into 72-wide chunks;
- FFN `project_out` Conv1x1;
- final head exact input-channel chunks.

The compile/CModel matrix runner is:

```text
agentflow_rhb/scripts/run_sdformer_hw128_compile_matrix.sh
```

Latest matrix output:

```text
/root/demo/artifacts/output_sdformer_hw128_generated_compile_20260720/compile_cmodel_summary.tsv
```

Summary:

| Metric | Count |
| --- | ---: |
| generated modules | 66 |
| ONNX export pass | 66 |
| compile pass | 66 |
| CModel pass | 66 |
| return-code failures | 0 |

Important boundary: some large `128x128` / 72-channel Conv candidates still
emit allocator/liveness warnings in `compile.log` even with return code 0 and a
passing CModel. Those shapes are accepted for compiler-level coverage, but the
board runtime path should still prefer the board-proven chunk/split contracts
until each large package has an explicit `All same: True` board record.

## Retrain-Required Hardware-Aligned Replacement

The production direction for SDFormer is not the strict upstream attention
graph.  The accepted replacement is:

```text
agentflow_rhb/training/sdformer_aligned_hw/sdformer_aligned_hw.py
```

This model preserves the 128x128 encoder/decoder tensor topology, but replaces
both unsupported SDFormer window-attention cores and original FFN cores with
shape-preserving `Conv3x3 C->C` mixers.  It therefore requires retraining and is
not checkpoint-compatible with the upstream SDFormer weights.

Deployment-plan exporter:

```text
agentflow_rhb/scripts/export_sdformer_aligned_hw_deployment_plan.py
```

Latest verification output:

```text
/root/demo/artifacts/output_sdformer_aligned_hw_deployment_plan_20260720_verify/sdformer_aligned_hw_deployment_plan.json
/root/demo/artifacts/output_sdformer_aligned_hw_deployment_plan_20260720_verify/sdformer_aligned_hw_deployment_plan.md
```

Current plan summary:

| Contract | Count |
| --- | ---: |
| stem Conv then Host ReLU | 2 |
| LayerNorm2d Host glue | 68 |
| single Conv3x3 mixer on RHB | 64 |
| 72-channel Conv3x3 output split | 4 |
| downsample Conv then Host PixelUnshuffle | 3 |
| upsample Conv then Host PixelShuffle | 3 |
| decoder reduce Conv1x1 | 2 |
| final head input-channel split | 1 |

Total planned RHB launches: `91`.

The RHB component evidence is complete for the scheduled mixer contracts:

- `24->24 @128x128`, `48->48 @64x64`, `96->96 @32x32`,
  `192->192 @16x16` Conv3x3 mixers are board `All same: True`;
- `72->72 @128x128` single Conv3x3 is rejected;
- three `72->24 @128x128` output-channel chunks are board `All same: True`.

Latest board smoke rerun:

```text
/root/demo/artifacts/output_sdformer_aligned_hw_deployment_plan_20260720_verify/board_smoke_c24_128.log
/root/demo/artifacts/output_sdformer_aligned_hw_deployment_plan_20260720_verify/board_smoke_c72_oc0.log
```

Both returned `board_status: pass` and `All same: True`.

End-to-end depth accuracy remains pending until a trained aligned checkpoint is
available.  Once the checkpoint is produced, the same exporter should be used as
the calibration/board-run contract source.

## Window Attention Decomposition Probe

We also tested whether SDFormer window attention can be strictly decomposed and
offloaded to RHB piece by piece.

Probe roots:

```text
/root/demo/artifacts/output_sdformer_window_attn_probe_20260717
/root/demo/artifacts/output_sdformer_window_attn_probe_20260717_bwc
```

Full stage1/window16 batched-window tensors have shapes like:

```text
q/k/v: [1, 64, 8, 256]
qk:    [1, 64, 8, 8]
out:   [1, 64, 8, 256]
```

Those direct 4D probes failed compile/CModel for reshape, normalize, matmul,
softmax, and full branch core. The meaningful exact split is therefore one
window at a time:

```text
q/k/v: [1, 8, 256]
qk:    [1, 8, 8]
out:   [1, 8, 256]
```

With `BWC` layout:

| Submodel | CModel | Board |
| --- | --- | --- |
| `sdformer_test.window_one_qk_matmul_w16` | pass | fail at load: `Unsupported tensor layout: NDWW` |
| `sdformer_test.window_one_softmax_w16` | pass | runtime timeout after UIO interrupt polling timeouts |
| `sdformer_test.window_one_av_matmul_w16` | pass | runtime timeout after UIO interrupt polling timeouts; board SSH timed out after the run |
| `sdformer_test.window_one_l2_normalize_w16` | fail | not tested |
| `sdformer_test.window_one_core_w16` | fail | not tested |

Interpretation:

- `MatMul` and `Softmax` are not impossible at the compiler/CModel level when
  using one-window `BWC` tensors.
- Current board deployment is still not production-usable:
  - QK emits `NDWW`, which board `deploy.py`/`ac_driver.load_model()` rejects.
  - Softmax loads but hangs/timeout during inference.
  - AV MatMul also loads but hangs/timeout during inference.
  - L2 normalize does not compile in this decomposition.
- A strictly equivalent RHB attention path would require additional runtime
  support for `NDWW`, fixed matmul/softmax board runtime behavior for these
  tensor layouts, and a Host or rewritten normalize path. Until those are
  solved, keep the window-attention core on Host and only offload qkv/project
  Conv blocks.

## Exact Tokenized Window-Attention Translation

We also tested the stricter question: can SDFormer window attention be
translated exactly into the same token-style contract used by the accepted
CompletionFormer/PVT attention rules?

The exact translation is mathematically valid. For each window branch:

```text
Original SDFormer:
  q/k/v: [B, Nw, Cg, P]
  attn:  softmax(q @ k^T)        -> [B, Nw, Cg, Cg]
  out:   attn @ v                -> [B, Nw, Cg, P]

Tokenized equivalent:
  q/k/v: [B*Nw, Cg, P]
  attn:  softmax(q @ k^T)        -> [B*Nw, Cg, Cg]
  out:   attn @ v                -> [B*Nw, Cg, P]
```

where `Cg = D/3` and `P = dh*dw`. Host-side window pack/unpack restores the
original `[B, D/3, H, W]` branch output. The equivalence checker covers all
SDFormer HW128 window shapes:

```text
agentflow_rhb/scripts/check_sdformer_tokenized_attention_equivalence.py
/root/demo/artifacts/output_sdformer_tokenized_attention_matrix_20260720/equivalence.json
```

All checked cases produced `max_abs=0.0` and `mean_abs=0.0`, so the translation
does not change PyTorch semantics.

However, the translated tokenized core is still not RHB-safe under the current
compiler/runtime stack. Generated probe modules:

```text
agentflow_rhb/scripts/generate_sdformer_tokenized_attention_modules.py
/root/demo/models/sdformer_test/tokenized_generated
```

The fixed-layout batched-token compile/CModel matrix is:

```text
agentflow_rhb/scripts/run_sdformer_tokenized_attention_matrix.sh
/root/demo/artifacts/output_sdformer_tokenized_attention_matrix_fixedlayout_20260720/compile_cmodel_summary.tsv
```

Summary across the 12 unique `(Cg, P)` pairs used by SDFormer HW128:

| Tokenized core op | Cases | Compile/CModel result | Main marker |
| --- | ---: | --- | --- |
| `q @ k.transpose(-2, -1)` | 12 | fail | `Failed to parse op insts` |
| `attn @ v` | 12 | fail | `Failed to parse op insts` |
| `softmax(..., dim=-1)` | 12 | fail | `layout attr is not equal with the size of op shape` |

Interpretation:

- The exact tokenized form is useful as the Host implementation contract and as
  an AgentFlow analysis primitive.
- It is not a valid batched RHB offload contract yet. The current compiler does
  not generate usable CModel packages for the `[B*Nw, Cg, P]` tokenized QK/AV
  MatMul shapes, and Softmax fails layout-shape lowering even with the correct
  single-input `input0=WC` layout.
- Therefore SDFormer attention should still be allocated as:

```text
RHB:
  qkv Conv1x1
  project_out Conv1x1

Host:
  window pack/unpack
  L2 normalize
  qk matmul
  softmax
  av matmul
```

If a future compiler/runtime accepts `[B*Nw, Cg, P]` tokenized MatMul/Softmax
with CModel and board `All same=True`, AgentFlow can promote this translation
from Host-only to RHB-candidate without changing model semantics.

There is one narrower diagnostic path: force a single window into a real batch
of one and compile with `BWC`, not `WC`:

```text
q/k/v: [1, Cg, P]
layout: input0=BWC,input1=BWC
```

The reproducible runner is:

```text
agentflow_rhb/scripts/run_sdformer_onewindow_bwc_attention_probe.sh
/root/demo/artifacts/output_sdformer_onewindow_bwc_attention_probe_20260720/summary.tsv
```

For `Cg=8, P=256`, QK, Softmax, and AV all compile, pack, and complete CModel
in this one-window `BWC` form. This confirms that the `B*Nw` batched dimension
and layout contract are part of the compiler problem.

This still does not make the path production-usable on board:

- QK output lowers to `NDWW`, and `deploy.py` / `ac_driver.load_model()` rejects
  that output layout.
- Softmax and AV have prior board logs showing UIO/status timeout after loading.
- Running every SDFormer window as a separate RHB launch would be expensive even
  if those runtime issues were fixed.

So AgentFlow treats one-window `BWC` attention as a strict diagnostic/research
candidate, not as an accepted RHB partition.

## Contrast With Accepted PVT Attention Rules

Do not blindly reuse the CompletionFormer/PVT attention acceptance as a full
SDFormer attention acceptance. The accepted PVT-style attention path and
SDFormer window attention have different tensor contracts.

Accepted CompletionFormer/PVT token attention examples:

```text
/root/demo/models/completionformer_test/pvttiny_stage1_attn_tokens_srnorm_residual_28x38.py
/root/demo/models/completionformer_test/pvttiny_stage2_attn_tokens_srnorm_residual_14x19.py
/root/demo/models/completionformer_test/pvttiny_stage3_attn_tokens_srnorm_residual_7x9.py
/root/demo/models/completionformer_test/pvttiny_stage1_ln_fc1_relu_tokens_1064x24.py
```

Those use a token-friendly layout contract:

```text
x:    [B, N, C]
x_sr: [B, Nsr, C]
q/k/v Linear over C
attn: [B, heads, N, Nsr]
out:  [B, N, C]
```

SDFormer window attention instead uses a windowed channel-attention contract:

```text
x:     [B, D, H, W]
qkv:   [B, 3D, H, W]
split: three branches, each [B, D, H, W]
q/k/v: [B, D/3, H, W]
pack:  [B, Nw, D/3, P], P = dh * dw
attn:  [B, Nw, D/3, D/3]
out:   [B, D/3, H, W]
cat:   [B, D, H, W]
```

Operator-by-operator mapping:

| SDFormer op | Similar accepted rule | Difference | Current allocation |
| --- | --- | --- | --- |
| `qkv Conv1x1` | PVT q/k/v Linear / Conv1x1 | standard pointwise projection | RHB |
| `qkv_dwconv groups=3D` | none accepted for board | depthwise group Conv compiled but board mismatched | Host or retrain replacement |
| `split/chunk` | Host token glue | no arithmetic, layout-only | Host |
| window `reshape/permute` | Host token/image layout glue | emits window-layout tensors, not plain WC/BCHW | Host |
| `F.normalize(..., dim=-1)` | no board-proven equivalent | CModel fails in full one-window core | Host |
| `q @ k.transpose` | PVT token MatMul/Softmax research path | SDFormer output can become `NDWW`; board loader rejects it | Host |
| `softmax` | PVT token attention can compile in selected cases | SDFormer one-window softmax times out on board | Host |
| `attn @ v` | PVT token MatMul research path | SDFormer one-window AV times out on board | Host |
| inverse window unpack | Host layout glue | not useful as a standalone RHB op | Host |
| `project_out Conv1x1` | PVT output projection Linear/Conv | standard pointwise projection | RHB |

The safe reuse from PVT is therefore:

```text
RHB:
  pointwise q/k/v/qkv projections
  pointwise output projection

Host:
  attention score construction
  normalize/softmax
  layout reshape/permute
  residual/concat glue
```

To move more SDFormer attention work to RHB without changing semantics, the
next research target is not another PyTorch rewrite alone. It needs board-level
support for the one-window matmul/softmax contract: accepted output layout,
non-timeout runtime behavior, and a replacement or Host treatment for L2
normalize. If latency-first retraining is allowed, replace the whole
depthwise+window-attention core with board-proven Conv/ReLU style mixing and
retrain the SDFormer HW variant.

## Strict Runtime Partition

For each Transformer block:

1. Host: `LayerNorm`.
2. RHB: qkv pointwise Conv if the stage/channel shape has board proof.
3. Host: qkv depthwise Conv for strict no-retrain path.
4. Host: window split, normalize, attention matmul, softmax, inverse window.
5. RHB: attention `project_out` pointwise Conv.
6. Host: residual add and second `LayerNorm`.
7. RHB: FFN `project_in` as output-channel split chunks, preferably 72-wide.
8. Host: FFN depthwise Conv, chunk, GELU, tensor multiply.
9. RHB: FFN `project_out`.
10. Host: residual add.

Downsample and upsample:

- RHB runs the Conv before PixelUnshuffle/PixelShuffle.
- Host runs PixelUnshuffle/PixelShuffle and skip concat/crop.

Final head:

- Use exact input-channel split chunks.
- Host sums partial channel0 outputs, adds bias once if present, and applies
  `clamp(min=0)` / ReLU.

The executable skeleton for this partition is:

```text
agentflow_rhb/scripts/run_sdformer_hw128_orchestrated.py
```

It runs the 128x128 SDFormer-compatible Host path and emits the RHB allocation
contract. With random inputs and no checkpoint, the current trace writes:

```text
/root/demo/artifacts/output_sdformer_hw128_orchestrated_20260720/sdformer_hw128_orchestrated_outputs.npz
/root/demo/artifacts/output_sdformer_hw128_orchestrated_20260720/sdformer_hw128_orchestrated_schedule.json
```

Current allocation summary:

| Decision | Count |
| --- | ---: |
| `RHB_CONV_SINGLE_OUTPUT` | 112 |
| `HOST_LAYER_NORM` | 68 |
| `HOST_DEPTHWISE_GROUP_CONV` | 68 |
| `HOST_RHB_SPLIT_ATTENTION` | 34 |
| `RHB_EXACT_FFN_PROJECT_IN_OC_SPLIT72` | 34 |
| `HOST_RHB_SPLIT_FFN` | 34 |
| `HOST_RHB_SPLIT_TRANSFORMER_BLOCK` | 34 |
| `HOST_RHB_SPLIT_SHUFFLE_BLOCK` | 6 |
| `RHB_EXACT_FINAL_HEAD_IC_SPLIT` | 1 |

## HW128 Schedule Trace

The 128x128-compatible schedule trace runner is:

```text
agentflow_rhb/scripts/run_sdformer_hw128_schedule_trace.py
```

It runs the full SDFormer forward pass with 128x128-compatible window sizes and
writes:

```text
/root/demo/artifacts/output_sdformer_probe_20260717/sdformer_hw128_schedule_trace.json
```

Current trace summary:

| Decision | Count |
| --- | ---: |
| `RHB_CANDIDATE_BOARD_PROVEN_BY_PATTERN` | 146 |
| `HOST_LAYER_NORM` | 68 |
| `HOST_STRICT_FALLBACK_DEPTHWISE` | 68 |
| `COMPOSITE_HOST_RHB_SPLIT` | 102 |
| `COMPOSITE_RHB_CONV_HOST_PIXEL_SHUFFLE` | 6 |
| `RHB_EXACT_IC_SPLIT` | 1 |

This is a deployability trace, not an accepted accuracy result, because the
local upstream SDFormer tree does not include a checkpoint.

## Retraining-Friendly Variant

If latency is more important than strict pretrained semantics, train a
compiler-aligned SDFormer variant with:

- depthwise group Conv replaced by board-proven non-group Conv, separable
  pointwise Conv, or Host-sampled small Conv approximations;
- GELU/gate multiply replaced by ReLU-style blocks only when retrained;
- 128x128 window sizes chosen to divide every stage resolution;
- scale-aware boundary calibration using the shared NYU calibration32/128 data.

### Board-Proven Approximate Attention Replacement

The first accepted latency-first replacement for the unsupported SDFormer
window-attention core is a single shape-preserving Conv3x3:

```text
Host:
  optional LayerNorm / residual glue

RHB:
  Conv3x3 C -> C, stride=1, padding=1

Host:
  optional activation and residual add
```

This is not semantically equivalent to the pretrained SDFormer attention. It is
a retrain-required hardware-aligned variant, but it keeps the same tensor
contract `[B,C,H,W] -> [B,C,H,W]` and avoids depthwise group Conv, window
reshape, L2 normalize, MatMul, and Softmax.

Generator and runner:

```text
agentflow_rhb/scripts/generate_sdformer_approx_attention_modules.py
agentflow_rhb/scripts/run_sdformer_approx_attention_matrix.sh
```

Clean board matrix:

```text
/root/demo/artifacts/output_sdformer_approx_attention_matrix_clean_20260720/summary.tsv
```

Board results:

| Approx attention module | Board |
| --- | --- |
| `Conv3x3 24 -> 24 @ 128x128` | `All same: True` |
| `Conv3x3 48 -> 48 @ 64x64` | `All same: True` |
| `Conv3x3 96 -> 96 @ 32x32` | `All same: True` |
| `Conv3x3 192 -> 192 @ 16x16` | `All same: True` |
| `Conv3x3 72 -> 72 @ 128x128` | `All same: False` |
| `Conv3x3 72 -> 24 @ 128x128`, chunk 0 | `All same: True` |
| `Conv3x3 72 -> 24 @ 128x128`, chunk 1 | `All same: True` |
| `Conv3x3 72 -> 24 @ 128x128`, chunk 2 | `All same: True` |

The refinement/full-resolution `72 -> 72 @ 128x128` case must therefore use
exact output-channel split:

```text
RHB:
  Conv3x3 72 -> 24, output channels 0..23
  Conv3x3 72 -> 24, output channels 24..47
  Conv3x3 72 -> 24, output channels 48..71

Host:
  concatenate the three outputs along C
```

This split is exact for the Conv3x3 replacement because convolution output
channels are independent. A real checkpoint exporter should slice the trained
`72 -> 72` Conv weights by output channel. The failed full `72 -> 72` package
has allocator markers such as `Unable to allocate memory` and `No valid
allocation found`, followed by board `All same: False`, so it should be
blacklisted even though CModel completes.

Do not use the generated `Conv3x3 + torch.relu` form as an RHB subgraph. It
triggers the current MLIR `RankedTensorType` assertion. If activation is needed
for a retrained HW variant, keep it as Host glue after the RHB Conv unless a
specific activation wrapper is separately board-proven.

### Training-Ready HW Aligned Model

The probe result has been promoted into a training-ready SDFormer HW128 variant:

```text
agentflow_rhb/training/sdformer_aligned_hw/sdformer_aligned_hw.py
agentflow_rhb/scripts/run_sdformer_aligned_hw_smoke.py
```

This model keeps the upstream SDFormer U-Net style encoder/decoder topology,
but replaces both unsupported window attention and the original FFN
depthwise/GELU/gate core with the same shape-preserving Conv3x3 mixer:

```text
Transformer block:
  x = x + Conv3x3(LayerNorm(x))
  x = x + Conv3x3(LayerNorm(x))
```

The model is intentionally retrain-required. It is not expected to load an
upstream SDFormer checkpoint with strict key/shape compatibility.

Smoke test:

```bash
cd /root/demo
/opt/conda/bin/python artifacts/successful_pipelines/completionformer_hw128_ckpt00059_rhb_convonlycf_20260702/portable_runtime/agentflow_rhb/scripts/run_sdformer_aligned_hw_smoke.py
```

Latest smoke output:

```text
/root/demo/artifacts/output_sdformer_aligned_hw_smoke_20260720/sdformer_aligned_hw_smoke_report.json
```

Summary:

| Metric | Value |
| --- | ---: |
| input | `rgb [1,3,128,128]`, `dep [1,1,128,128]` |
| output | `pred [1,1,128,128]` |
| parameters | `8,884,164` |
| CPU smoke latency | about `90 ms` on the current host |
| Conv3x3 mixer single-RHB contracts | `64` |
| Conv3x3 mixer output-split contracts | `4` |

The 4 split contracts are exactly the two refinement blocks' attention and FFN
mixers:

```text
refinement.0.attn_mixer: 72 -> 72 @ 128x128, export as 3 x (72 -> 24)
refinement.0.ffn_mixer:  72 -> 72 @ 128x128, export as 3 x (72 -> 24)
refinement.1.attn_mixer: 72 -> 72 @ 128x128, export as 3 x (72 -> 24)
refinement.1.ffn_mixer:  72 -> 72 @ 128x128, export as 3 x (72 -> 24)
```

All other Conv3x3 mixers use one RHB launch each under the current board-proof
matrix. Host glue still handles LayerNorm, residual add, concat/crop,
PixelShuffle/Unshuffle, activation and final clamp.

Training entry point:

```text
agentflow_rhb/training/train_sdformer_aligned_hw.py
```

Expected remote command on the existing CSPN training host layout:

```bash
cd /workspace/CSPN/cspn_pytorch
python /path/to/portable_runtime/agentflow_rhb/training/train_sdformer_aligned_hw.py \
  --repo-root /workspace/CSPN/cspn_pytorch \
  --data-root /workspace/CSPN/cspn_pytorch \
  --train-list /workspace/CSPN/cspn_pytorch/datalist/nyudepth_hdf5_train.csv \
  --eval-list /workspace/CSPN/cspn_pytorch/datalist/nyudepth_hdf5_val.csv \
  --save-dir /workspace/CSPN/cspn_pytorch/output/agentflow_sdformer_aligned_hw128 \
  --image-size 128 \
  --epochs 80 \
  --batch-size 8 \
  --val-subset 512
```

The script writes:

```text
model_best.pt
model_latest.pt
log_train.txt
metadata.json
```

After training, the exporter must use the `rhb_export_hints()` contracts:

- every `RHBConvMixer` with `contract=single` exports as one Conv3x3 package;
- every `RHBConvMixer` with `contract=output_split` exports as three Conv3x3
  packages with output-channel slices `[0:24]`, `[24:48]`, `[48:72]`;
- Host concatenates the split outputs before the next Host glue op.

## Reproduce Probe Suite

```bash
cd /root/demo
bash artifacts/successful_pipelines/completionformer_hw128_ckpt00059_rhb_convonlycf_20260702/portable_runtime/agentflow_rhb/scripts/run_sdformer_probe_suite.sh
```

The script compiles, runs CModel, packs, and board-validates the current
SDFormer probe set.

Generate and CModel-check the full SDFormer HW128 Conv candidate matrix:

```bash
cd /root/demo
python artifacts/successful_pipelines/completionformer_hw128_ckpt00059_rhb_convonlycf_20260702/portable_runtime/agentflow_rhb/scripts/generate_sdformer_hw128_probe_modules.py
OUT=/root/demo/artifacts/output_sdformer_hw128_generated_compile_20260720 \
  bash artifacts/successful_pipelines/completionformer_hw128_ckpt00059_rhb_convonlycf_20260702/portable_runtime/agentflow_rhb/scripts/run_sdformer_hw128_compile_matrix.sh
```

Run the SDFormer HW128 partition/orchestration skeleton:

```bash
cd /root/demo
/opt/conda/bin/python artifacts/successful_pipelines/completionformer_hw128_ckpt00059_rhb_convonlycf_20260702/portable_runtime/agentflow_rhb/scripts/run_sdformer_hw128_orchestrated.py
```
