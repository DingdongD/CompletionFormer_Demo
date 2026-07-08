# CompletionFormer HW128 Real Image Board Compare

## What Changed

The ckpt decoder/head runner now supports real exported NYU features:

```text
/root/demo/tools/rhb_completionformer_decoder_system_runner_128x128_ckpt.py
```

New export and visualization scripts:

```text
/root/demo/tools/export_completionformer_hw128_real_features.py
/root/demo/tools/visualize_completionformer_board_real_pred.py
```

## Input

The feature export reuses the first sample from the existing reference visualization source:

```text
/root/demo/artifacts/visualizations/nyu_ref_model_hw_128x128_ckpt/nyu_first4_forward_outputs.npz
```

This ensures the RGB/sparse-depth input matches the first row of the previous reference image.

Exported real feature file:

```text
/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt_20260702/nyu_sample0_real_features_scale1.npz
```

Reference output consistency check:

```text
pred_ref_vs_source_maxdiff      4.2915344e-06
pred_init_vs_source_maxdiff     1.9073486e-06
```

So the exported feature file is aligned with the original reference visualization input.

## Board Run

Board output:

```text
/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt_20260702/board_decoder_system_128x128_ckpt_nyu_sample0_outputs.npz
```

Board log:

```text
/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt_20260702/board_decoder_system_128x128_ckpt_nyu_sample0.log
```

The board runner consumed real `fe1..fe7_i8` features:

```text
FEATURE_SOURCE: /tmp/nyu_sample0_real_features_scale1.npz
DECODER_SYSTEM_128X128_CKPT_EXECUTED: True
LATENCY decoder_system_128x128_ckpt_tracked_total: 1128.096 ms
```

No timeout/error marker was seen.

## Visualization

Generated comparison image:

```text
/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt_20260702/nyu_sample0_ref_vs_board_pred_scale1.png
```

Generated comparison tensors:

```text
/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt_20260702/nyu_sample0_ref_vs_board_pred_scale1.npz
```

Raw output ranges:

```text
ref_pred          min/max/mean  1.204878 / 5.737456 / 3.829129
board_init_depth  min/max/mean  0 / 53 / 2.765320
board_pred        min/max/mean  0 / 53 / 2.882127
raw_absdiff       mean/max      4.388147 / 47.983810
```

## Current Result

The board output image is not visually consistent with the reference prediction.

The immediate cause is the current compiler input quantization convention. The submodel graph inputs have `input_scale=1.0`, so the real float features were exported with round+clip to int8 using scale 1. That destroys low-magnitude encoder features:

```text
fe7_i8 nonzero 0 / 3072, min/max 0 / 0
fe6_i8 nonzero 92 / 6144, min/max 0 / 1
fe5_i8 nonzero 968 / 12288, min/max 0 / 2
```

This directly propagates to the first decoder stages:

```text
fd6 board min/max/mean 0 / 0 / 0
fd6 ref   min/max/mean 0 / 0.340136 / 0.009204

fd5 board min/max/mean 0 / 0 / 0
fd5 ref   min/max/mean 0 / 1.169033 / 0.052554
```

After `fe7` quantizes to all zeros, the board decoder/head can still execute, but it cannot reproduce the reference prediction.

## Reproduce

```bash
OUT=artifacts/output_completionformer_decoder_system_128x128_ckpt_20260702

python tools/export_completionformer_hw128_real_features.py \
  --sample-index 0 \
  --feature-scale 1.0 \
  --out "$OUT/nyu_sample0_real_features_scale1.npz"

sshpass -p root scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  tools/rhb_completionformer_decoder_system_runner_128x128_ckpt.py \
  root@192.168.115.122:/home/root/workspace/demo_vp_xj/packers/

sshpass -p root scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "$OUT/nyu_sample0_real_features_scale1.npz" \
  root@192.168.115.122:/tmp/nyu_sample0_real_features_scale1.npz

sshpass -p root ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  root@192.168.115.122 \
  "cd /home/root/workspace/demo_vp_xj/packers && \
   python3 rhb_completionformer_decoder_system_runner_128x128_ckpt.py \
   packer_decoder_system_128x128_ckpt_rram_false \
   --feature-npz /tmp/nyu_sample0_real_features_scale1.npz \
   --save-npz /tmp/decoder_system_128x128_ckpt_nyu_sample0_outputs.npz"

sshpass -p root scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  root@192.168.115.122:/tmp/decoder_system_128x128_ckpt_nyu_sample0_outputs.npz \
  "$OUT/board_decoder_system_128x128_ckpt_nyu_sample0_outputs.npz"

python tools/visualize_completionformer_board_real_pred.py \
  --feature-npz "$OUT/nyu_sample0_real_features_scale1.npz" \
  --board-npz "$OUT/board_decoder_system_128x128_ckpt_nyu_sample0_outputs.npz" \
  --out-png "$OUT/nyu_sample0_ref_vs_board_pred_scale1.png" \
  --out-npz "$OUT/nyu_sample0_ref_vs_board_pred_scale1.npz"
```

## Next Fix

To make the board prediction visually match the reference, the next step is not another runner change. We need compiler-aligned quantization for real feature inputs:

- export per-submodel activation scales from the compiler/ONNX quantization pass,
- quantize real `fe1..fe7` with those scales instead of fixed scale 1,
- update host glue to dequantize/requantize split sums with the same scales,
- then regenerate the board-vs-reference image.

## 2026-07-02 Activation-Scale-Aware v2

Implemented:

```text
/root/demo/tools/recalibrate_completionformer_ckpt_onnx_scales.py
/root/demo/tools/rhb_completionformer_decoder_system_runner_128x128_ckpt.py
```

The recalibration flow uses the real NYU sample's submodel inputs to rewrite
input/output scales in the ONNX files, then recompiles all 14 submodels.

Scale-aware output root:

```text
/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt_scaleaware_20260702
```

Scale table:

```text
/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt_scaleaware_20260702/onnx_models_patched_v2/activation_scales.csv
```

All 14 v2 submodels pass compile+cmodel:

```text
/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt_scaleaware_20260702/compile_cmodel_summary_v2_all.tsv
```

Board packer:

```text
/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt_scaleaware_20260702/packer_decoder_system_128x128_ckpt_scaleaware_v2_rram_false
```

Board output:

```text
/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt_scaleaware_20260702/board_decoder_system_128x128_ckpt_nyu_sample0_scaleaware_v2_outputs.npz
```

Visualization:

```text
/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt_scaleaware_20260702/nyu_sample0_ref_vs_board_pred_scaleaware_v2.png
```

Board execution:

```text
DECODER_SYSTEM_128X128_CKPT_EXECUTED: True
LATENCY decoder_system_128x128_ckpt_tracked_total: 2026.951 ms
```

Numerical improvement:

```text
previous raw/scale1 pred mean abs diff: 4.388147
previous glue-only scale-aware mean abs diff: 3.235972
activation-scale-aware v2 mean abs diff: 0.900617
activation-scale-aware v2 max abs diff: 5.678226
```

Representative layer errors after v2:

```text
fd6_f       mean/max abs diff 0.007671 / 0.340136
fd5_f       mean/max abs diff 0.059767 / 1.086304
fd4_f       mean/max abs diff 0.101222 / 1.891756
fd3_f       mean/max abs diff 0.177021 / 2.813306
fd2_f       mean/max abs diff 0.255552 / 7.828125
init_depth  mean/max abs diff 0.726582 / 5.950712
guide       mean/max abs diff 6.137292 / 74.03491
confidence  mean/max abs diff 0.575206 / 1.442495
```

Current interpretation:

- The real-image board output is now structurally meaningful and visually much closer to the reference.
- It is not yet numerically matched.
- The remaining gap is concentrated in full-resolution head outputs, especially `guide` and `confidence`.
- v2 uses one calibration sample and repeated host-side int8 requantization; both are too rough for final accuracy.

Recommended next step:

- Calibrate scales on the 32-sample representative set instead of one sample.
- Use per-edge scale metadata from the packer/ONNX in the runner rather than hardcoded constants.
- Reduce full-resolution host/RHB requantization points, especially around `head_in`, `dep/gd/cf` heads.

## 32-sample Representative Calibration

Implemented a 32-sample representative activation calibration pass for the current 128x128 compiler-aligned decoder/head pipeline.

Generated artifacts:

```text
artifacts/output_completionformer_decoder_system_128x128_ckpt_scaleaware32_20260702/onnx_models_patched_v32/activation_scales.csv
artifacts/output_completionformer_decoder_system_128x128_ckpt_scaleaware32_20260702/compile_cmodel_summary_v32_all.tsv
artifacts/output_completionformer_decoder_system_128x128_ckpt_scaleaware32_20260702/packer_decoder_system_128x128_ckpt_scaleaware_v32_rram_false/
artifacts/output_completionformer_decoder_system_128x128_ckpt_scaleaware32_20260702/board_decoder_system_128x128_ckpt_nyu_sample0_scaleaware_v32.log
artifacts/output_completionformer_decoder_system_128x128_ckpt_scaleaware32_20260702/board_decoder_system_128x128_ckpt_nyu_sample0_scaleaware_v32_outputs.npz
artifacts/output_completionformer_decoder_system_128x128_ckpt_scaleaware32_20260702/nyu_sample0_ref_vs_board_pred_scaleaware_v32.png
artifacts/output_completionformer_decoder_system_128x128_ckpt_scaleaware32_20260702/nyu_sample0_ref_vs_board_pred_scaleaware_v32.npz
```

All 14 patched ONNX submodels compile and cmodel pass. Board execution also completed without timeout or runtime error:

```text
DECODER_SYSTEM_128X128_CKPT_EXECUTED: True
LATENCY decoder_system_128x128_ckpt_tracked_total: 1392.921 ms
```

Per-stage board latency:

```text
dec6_rhb: 4.290 ms
dec5_rhb: 7.668 ms
dec4_rhb: 7.315 ms
dec3_rhb: 17.821 ms
dec2_resize_upconv_split_sum_host_relu: 129.548 ms
dec2_exact_block_conv0_rhb: 73.257 ms
dec2_exact_block_conv1_rhb: 74.927 ms
host_dec2_exact_residual_relu: 13.116 ms
dep_dec1_full_rhb: 152.836 ms
dep_dec0_rhb: 237.028 ms
gd_dec1_full_rhb: 152.743 ms
gd_dec0_rhb: 244.128 ms
cf_dec1_rhb: 140.396 ms
cf_dec0_rhb: 137.849 ms
```

32-sample activation scales:

```text
completionformer_test.decoder_tiny_dec6_resize_conv_basicblock_nocbam_4x4_to_8x8_ckpt,fe7,256.0,128.0
completionformer_test.decoder_tiny_dec5_resize_conv_basicblock_nocbam_8x8_to_16x16_ckpt,dec5_in,128.0,32.0
completionformer_test.decoder_tiny_dec4_resize_conv_basicblock_nocbam_16x16_to_32x32_ckpt,dec4_in,64.0,16.0
completionformer_test.decoder_tiny_dec3_resize_conv_basicblock_nocbam_32x32_to_64x64_ckpt,dec3_in,16.0,16.0
completionformer_test.decoder_tiny_dec2_resize_upconv_in80_chunk0_64x64_to_128x128_ckpt,dec2_chunk0_in,32.0,16.0
completionformer_test.decoder_tiny_dec2_resize_upconv_in80_chunk1_64x64_to_128x128_ckpt,dec2_chunk1_in,32.0,8.0
completionformer_test.decoder_tiny_dec2_exact_block_conv0_128x128_ckpt,dec2_up_ref,16.0,16.0
completionformer_test.decoder_tiny_dec2_exact_block_conv1_128x128_ckpt,dec2_block0_ref,64.0,32.0
completionformer_test.head_tiny_dep_dec1_conv_relu_128x128_ckpt,head_in_ref,32.0,8.0
completionformer_test.head_tiny_dep_dec0_conv_relu_128x128_ckpt,dep_dec0_in_ref,16.0,8.0
completionformer_test.head_tiny_gd_dec1_conv_relu_128x128_ckpt,head_in_ref,32.0,4.0
completionformer_test.head_tiny_gd_dec0_conv_128x128_ckpt,gd_dec0_in_ref,16.0,2.0
completionformer_test.head_tiny_cf_dec1_conv_relu_128x128_ckpt,head_in_ref,32.0,8.0
completionformer_test.head_tiny_cf_dec0_conv_sigmoid_128x128_ckpt,cf_dec0_in_ref,16.0,128.0
```

Numerical comparison on NYU sample0 against `ref_model_hw`:

```text
fd6_f abs mean/max: 0.007671 / 0.340136
fd5_f abs mean/max: 0.059767 / 1.086304
fd4_f abs mean/max: 0.110593 / 3.812500
fd3_f abs mean/max: 0.213999 / 3.149768
fd2_f abs mean/max: 0.157549 / 5.605440
init_depth_f abs mean/max: 0.810991 / 5.950712
guide_f abs mean/max: 4.410037 / 49.034908
confidence_f abs mean/max: 0.616563 / 1.379995
final pred abs mean/max: 1.014235 / 5.198215
```

Conclusion: the 32-sample representative calibration is functionally integrated and board-executable. It does not yet improve final sample0 prediction over the previous single-sample v2 calibration; the largest remaining mismatch is in the head outputs, especially confidence. This indicates the next useful work is targeted per-head calibration/output-scale tuning or head-local golden comparison, not pipeline connectivity.

## Layerwise Error Source Analysis for v32

Added a layerwise diagnostic script:

```text
tools/analyze_completionformer_head_error_v32.py
artifacts/output_completionformer_decoder_system_128x128_ckpt_scaleaware32_20260702/head_error_layerwise_v32.tsv
```

Also added and ran a board-side head-only probe that feeds golden `ref_model_hw` inputs directly into the head submodels:

```text
tools/rhb_completionformer_head_probe_128x128_ckpt.py
artifacts/output_completionformer_decoder_system_128x128_ckpt_scaleaware32_20260702/board_head_probe_v32_ref_inputs.log
artifacts/output_completionformer_decoder_system_128x128_ckpt_scaleaware32_20260702/head_probe_v32_ref_inputs_outputs.npz
```

Key decoder boundary errors from integrated board output:

```text
fd6 abs mean/max:       0.007671 / 0.340136
fd5 abs mean/max:       0.059767 / 1.086304
fd4 abs mean/max:       0.110593 / 3.812500
fd3 abs mean/max:       0.213999 / 3.149768
fd2 abs mean/max:       0.157549 / 5.605440
head_in abs mean/max:   0.052516 / 5.605440
```

The head input is not perfectly equal to ref, but it is not the dominant source. Feeding the board-derived `head_in` into PyTorch float heads gives much smaller dec1 errors than the real board dec1 outputs:

```text
float dep_dec1(board head_in) abs mean: 0.108267
float gd_dec1(board head_in)  abs mean: 0.227604
float cf_dec1(board head_in)  abs mean: 0.176833

board dep_dec1 abs mean: 0.408231
board gd_dec1  abs mean: 0.478902
board cf_dec1  abs mean: 0.799886
```

This means the error begins inside the board/compiler execution of the head submodels, not only from the preceding decoder feature error.

The head-only probe with golden ref inputs further separates dec1 and dec0:

```text
dep_dec1 golden input abs mean/max:        0.020298 / 0.686516
gd_dec1 golden input abs mean/max:         0.048333 / 1.012435
cf_dec1 golden input abs mean/max:         0.226825 / 10.935151

dep_dec0 golden input abs mean/max:        2.649969 / 6.614730
gd_dec0 golden input abs mean/max:         0.132177 / 3.495956
cf_dec0 sigmoid golden input abs mean/max: 0.597133 / 0.999992
```

Interpretation:

1. `dep_dec1` and `gd_dec1` are mostly healthy when given golden `head_in_ref`. Their large integrated errors are partly caused by accumulated `fd2/head_in` distribution shift plus board quantization.
2. `cf_dec1` is already less stable than dep/gd dec1 even with golden input, but it is not the largest final error source.
3. `dep_dec0` is a major independent failure point: even with golden `dep_dec0_in_ref`, board output mean is far below ref and abs mean is 2.65.
4. `cf_dec0_sigmoid` is also a major independent failure point: even with golden `cf_dec0_in_ref`, board output contains negative values after dequantization, which is incompatible with sigmoid semantics. This points to sigmoid/output-scale/output interpretation rather than upstream decoder error.
5. `gd_dec0` is healthy with golden input, so the problem is not a generic 96-channel full-resolution conv issue.

Input clipping under current v32 scales:

```text
head_in_ref scale=32 clip_hi=0.1178%
dep_dec0_in_ref scale=16 clip_hi=0.0081%
gd_dec0_in_ref scale=16 clip_hi=0.0004%
cf_dec0_in_ref scale=16 clip_hi=0.0877%
```

Clipping exists but does not fully explain `dep_dec0`, because its golden-input clipping rate is only 0.0081%. The most likely remaining causes are head-specific compiler quantization/output handling for `dep_dec0` and sigmoid/output interpretation for `cf_dec0`.

## Quantization Mode and Further Head Debug

Current quantization is compiler-side static PTQ metadata, not QAT. The flow is:

```text
PyTorch ckpt -> ONNX float -> util.quant_onnx adds compiler quantization attributes -> compile.py/ACompiler
```

Important details:

```text
activation/input/output: tensor-level scale attributes (`input_scale`, `output_scale`)
conv weights: per-output-channel `weight_ch_scales`
bias: marked by model `tensor_16 = [".bias"]`, so bias is not plain int8
scale selection: power-of-two style scale from calibration tensor range; original code uses random calibration/percentile, our v32 pass rewrites scales from 32 NYU representative samples
runtime host glue: float -> int8 by round(x * input_scale), board output float = int8 / output_scale
```

For example v32 head metadata:

```text
dep_dec0 Conv: input_scale=16, output_scale=8, weight_ch_scales=[814.1691]
dep_dec0 Relu: output_scale=8
cf_dec0 Conv:  input_scale=16, output_scale=8, weight_ch_scales=[157.8766]
cf_dec0 Sigmoid lowered as HardSwish/PWL: output_scale=128, slopes_scale=512, intercepts_scale=1
```

This means `cf_dec0` is not a native sigmoid in the compiled graph. It is rewritten into a compiler `HardSwish` op carrying PWL parameters generated from sigmoid. The board result containing negative values after this node is therefore a concrete red flag for this lowering/output interpretation path.

Extra debug variants were added under `models/completionformer_test/`:

```text
head_tiny_dep_dec0_conv_only_128x128_ckpt.py
head_tiny_cf_dec0_conv_only_128x128_ckpt.py
head_tiny_dep_dec0_conv_relu_128x128_ckpt_pad8.py
head_tiny_cf_dec0_conv_sigmoid_128x128_ckpt_pad8.py
head_tiny_cf_dec0_conv_only_128x128_ckpt_pad8.py
head_tiny_dep_dec0_conv_relu_128x128_ckpt_pad16.py
head_tiny_cf_dec0_conv_sigmoid_128x128_ckpt_pad16.py
head_tiny_cf_dec0_conv_only_128x128_ckpt_pad16.py
```

Results:

```text
conv-only 1-channel variants: compile fails in layout/MLIR pass with ArrayRef invalid index
pad8/pad16 output variants: compile also fails in layout/MLIR pass with ArrayRef invalid index
initial pad variants with zero-filled channels created `weight_ch_scales=inf`, confirming quant_onnx has no zero-channel guard
```

Interpretation:

1. The current compiler path is fragile around 1-channel full-resolution head outputs.
2. `dep_dec0 conv+ReLU` compiles and runs, but its golden-input numerical error is large, so this is not just upstream decoder drift.
3. `cf_dec0 conv+Sigmoid` compiles and runs, but sigmoid is lowered to `HardSwish`/PWL and board output can be negative, so this path is not numerically reliable.
4. Simple padded-output workarounds are not currently viable because the compiler crashes before cmodel.

Near-term safe scheduling implication: keep `dep_dec0` and `cf_dec0/sigmoid` on Host, while keeping the healthier RHB pieces (`decoder`, `dep_dec1`, `gd_dec1`, `gd_dec0`, and possibly `cf_dec1`) on RHB. This avoids the two confirmed unreliable board subgraphs without changing model semantics.

## ReLU-specific Check

Added ReLU debug files:

```text
models/completionformer_test/relu_only_1ch_128x128.py
models/completionformer_test/relu_only_8ch_128x128.py
tools/export_completionformer_dep_dec0_preact.py
tools/rhb_relu_probe.py
artifacts/output_completionformer_relu_debug_20260702/dep_dec0_preact_ref.npz
```

Standalone ReLU-only subgraphs do not compile in the current layout pass:

```text
ONNXLayoutPass.cpp:2356: Assertion `out_str.size() >= resShape.size() && "layout attr is not equal with the size of op shape."' failed.
```

However, the important semantic check is conclusive for the current NYU sample0:

```text
dep_dec0 pre-activation min/max/mean: 0.328497 / 6.614730 / 3.664237
negative percentage: 0.0%
max(abs(ReLU(preact) - init_depth_raw_ref)): 0.0
```

Therefore `dep_dec0` ReLU is an identity for this sample. The large board error in `dep_dec0` cannot be caused by ReLU clipping or wrong ReLU semantics on the reference path. Since `dep_dec1` and `gd_dec1` are also Conv+ReLU and are mostly healthy with golden input, ReLU itself is unlikely to be the root cause. The remaining likely source for `dep_dec0` is the preceding 1-output-channel Conv quantization/execution/output handling before ReLU.

## Error Reduction Analysis

Checked whether `dep_dec0` is merely an output-scale interpretation problem by fitting board raw int8 output to ref float output.

Golden-input `dep_dec0` board raw vs ref:

```text
corr(raw, ref): -0.278243
best affine fit abs mean/max: 1.079606 / 3.764959
raw / 8 abs mean/max: 2.649969 / 6.614730
```

This is not a simple `output_scale` mismatch. The raw spatial pattern itself is poorly correlated with ref for `dep_dec0` under golden input.

Golden-input `cf_dec0` board raw vs ref:

```text
corr(raw, ref): 0.767585
best affine fit abs mean/max: 0.128522 / 0.839775
raw / 128 abs mean/max: 0.597133 / 0.999992
```

`cf_dec0` has meaningful correlation before dequantization, so the problem is more consistent with sigmoid/PWL/output-scale interpretation than a completely wrong conv pattern.

Tried output padding workarounds to avoid 1-channel output:

```text
head_tiny_dep_dec0_conv_relu_128x128_ckpt_pad8var
head_tiny_cf_dec0_conv_only_128x128_ckpt_pad8var
head_tiny_cf_dec0_conv_sigmoid_128x128_ckpt_pad8var
```

These still fail in the compiler layout/MLIR pass with `ArrayRef invalid index`. Therefore padded-output is not currently a viable workaround.

Hybrid scheduling counterfactuals were generated:

```text
artifacts/output_completionformer_decoder_system_128x128_ckpt_scaleaware32_20260702/hybrid_host_dec0_only_outputs.npz
artifacts/output_completionformer_decoder_system_128x128_ckpt_scaleaware32_20260702/hybrid_host_dec0_only_outputs_pred.png
artifacts/output_completionformer_decoder_system_128x128_ckpt_scaleaware32_20260702/hybrid_host_full_heads_outputs.npz
artifacts/output_completionformer_decoder_system_128x128_ckpt_scaleaware32_20260702/hybrid_host_full_heads_outputs_pred.png
```

Head-level improvements using current board `fd2`:

```text
dep board all abs mean/max:                         0.810991 / 5.950712
dep host dec0 only from board dep_fd1 abs mean/max: 0.806307 / 5.907313
dep host dec1+dec0 from board fd2 abs mean/max:     0.280422 / 4.679377

cf board all abs mean/max:                          0.616563 / 1.379995
cf host dec0 only from board cf_fd1 abs mean/max:   0.097853 / 0.960495
cf host dec1+dec0 from board fd2 abs mean/max:      0.145784 / 0.998653
```

Final NLSPN pred comparison:

```text
current v32 board pred abs mean/max:      1.014235 / 5.198215
hybrid host dec0-only pred abs mean/max:  1.016175 / 5.198214
hybrid host full-heads pred abs mean/max: 0.502634 / 4.170636
```

Interpretation:

1. Moving only dec0 to Host is not enough because `dep_fd1` from board already contains substantial error.
2. Moving full dep/gd/cf heads to Host from board `fd2` cuts final pred mean error by about 50%.
3. The remaining error after Host heads is mainly inherited from RHB decoder `fd2` quantization error.

Recommended low-risk scheduling to reduce error now:

```text
RHB:
  backbone/former/decoder through fd2

Host:
  dep_dec1 -> dep_dec0
  gd_dec1 -> gd_dec0
  cf_dec1 -> cf_dec0 + sigmoid
  NLSPN
```

This keeps the currently unreliable full-resolution head outputs off RHB and gives the largest measured accuracy gain without changing model semantics.

## model_00059.pt Check and Quantization Status

Current quantization configuration is already the intended mixed granularity:

```text
activation/input/output: per-tensor scale (`input_scale`, `output_scale`)
conv weights: per-output-channel scale (`weight_ch_scales`)
bias: 16-bit via `tensor_16 = [".bias"]`
```

The requested best ckpt path was checked:

```text
/root/demo/CompletionFormer/ref_model_hw/model_00059.pt
```

It is currently not a valid complete PyTorch checkpoint. Symptoms:

```text
size(model_00059.pt): 786432 bytes
size(model_00030.pt): 4763935 bytes
torch.load(model_00059.pt): PytorchStreamReader failed reading zip archive: failed finding central directory
unzip -t model_00059.pt: End-of-central-directory signature not found
PK end signature: missing in model_00059.pt, present in model_00030.pt
```

Therefore `model_00059.pt` appears truncated or incompletely transferred. It cannot be used for strict model loading, ONNX export, calibration, cmodel, or board validation yet.

Once a complete `model_00059.pt` is available, the correct rerun flow is:

```bash
export COMPLETIONFORMER_HW_CKPT=/root/demo/CompletionFormer/ref_model_hw/model_00059.pt

# Re-export the 14 ckpt-backed submodels so their weights come from model_00059.pt.
# Then rerun 32-sample representative calibration, compile+cmodel, pack, board run,
# and the same layerwise/hybrid error analysis.
```

Expected optimization baseline from model_00030 remains:

```text
RHB through fd2 + board heads:           final pred abs mean 1.014235
RHB through fd2 + Host full heads:       final pred abs mean 0.502634
```

Thus the current best low-risk accuracy alignment is to keep RHB offload through `fd2`, then execute `dep/gd/cf` full heads on Host. The next ckpt experiment should test whether `model_00059.pt` reduces the remaining `fd2` quantization error after the checkpoint file is corrected.

## model_00059.pt + Train32 Calibration Board Run

`model_00059.pt` was rechecked and is now valid:

```text
/root/demo/CompletionFormer/ref_model_hw/model_00059.pt
size: 4.6M / 4763935 bytes
zip test: OK
torch strict load: OK
```

Used calibration set:

```text
/root/demo/CompletionFormer/ref_model_hw/nyu_train_calibration_128_128x128.zip
extracted to: artifacts/nyu_train_calibration_128_128x128
calibration samples used: first 32 train samples
```

Generated artifacts:

```text
artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/export_summary.tsv
artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/onnx_models_patched_train32/activation_scales.csv
artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/compile_cmodel_summary_train32_all.tsv
artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/packer_decoder_system_128x128_ckpt00059_traincalib32_rram_false/
artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/board_decoder_system_128x128_ckpt00059_train32.log
artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/board_decoder_system_128x128_ckpt00059_train32_outputs.npz
artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/nyu_sample0_ref_vs_board_pred_ckpt00059_train32.png
artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/head_error_layerwise_ckpt00059_train32.tsv
```

All 14 ckpt-backed submodels were re-exported with `model_00059.pt`, calibrated with Train32 scales, and passed compile+cmodel:

```text
14 / 14 compile_ok=True
14 / 14 cmodel_ok=True
```

Board execution completed without timeout:

```text
DECODER_SYSTEM_128X128_CKPT_EXECUTED: True
LATENCY decoder_system_128x128_ckpt_tracked_total: 1394.617 ms
```

Train32 activation scales:

```text
dec6 input/output:        256 / 128
dec5 input/output:        128 / 32
dec4 input/output:         64 / 16
dec3 input/output:         16 / 32
dec2 up chunk0:            32 / 8
dec2 up chunk1:            32 / 8
dec2 block0:               32 / 16
dec2 block1:               64 / 64
dep_dec1:                  32 / 16
dep_dec0:                  16 / 16
gd_dec1:                   32 / 4
gd_dec0:                   32 / 2
cf_dec1:                   32 / 8
cf_dec0 sigmoid:           16 / 128
```

Full board output against `model_00059.pt` ref on NYU validate sample0:

```text
fd6 abs mean/max:        0.014208 / 0.391482
fd5 abs mean/max:        0.044467 / 0.916024
fd4 abs mean/max:        0.107597 / 3.812500
fd3 abs mean/max:        0.182788 / 2.907824
fd2 abs mean/max:        0.309218 / 14.703125
head_in abs mean/max:    0.103073 / 14.703125
init_depth abs mean/max: 0.714093 / 7.354456
guide abs mean/max:      6.701910 / 76.700348
confidence abs mean/max: 0.640629 / 1.460029
final pred abs mean/max: 0.865831 / 7.952108
```

Comparison with previous `model_00030.pt` v32 board run:

```text
model_00030 v32 final pred abs mean/max:       1.014235 / 5.198215
model_00059 train32 final pred abs mean/max:   0.865831 / 7.952108
```

So `model_00059.pt` improves average final pred error, but has a worse max error and visible top-band artifact in the board output.

Hybrid tests for `model_00059.pt`:

```text
full board pred abs mean/max:         0.865831 / 7.952108
Host cf_dec0 only pred abs mean/max:  0.865831 / 7.952108
Host dec0-only pred abs mean/max:     0.909380 / 11.169670
Host full heads pred abs mean/max:    0.974928 / 13.134056
```

Unlike `model_00030.pt`, moving full heads to Host does not help for `model_00059.pt`. The board `fd2/head_in` error is larger and Host heads amplify this shifted feature distribution. For this checkpoint, the best measured option on sample0 is the full board-head pipeline, despite known confidence/sigmoid issues.

Current interpretation:

1. The intended quantization granularity is in use: activation per-tensor, weight per-output-channel.
2. Train32 calibration works mechanically and produces a board-runnable model.
3. `model_00059.pt` improves average final pred error versus `00030`, but the decoder `fd2` error is now larger (`0.309` vs `0.158` mean), and this limits host/RHB split benefits.
4. Further accuracy work should target decoder `fd2` quantization first, especially `dec2` split/upconv/block scales, before revisiting head placement.

### Validate32 Scale Coverage with Train32 Calibration

Added validation-set scale coverage analysis:

```text
artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/validate32_input_scale_coverage_ckpt00059_train32.csv
```

This uses `model_00059.pt`, the Train32-calibrated input scales, and the 32 validate representative samples. Clipping rates are low but nonzero:

```text
fe7 scale=256 clip_hi=0.0824%
dec2_up_ref scale=32 clip_hi=0.0719%
gd_dec0_in_ref scale=32 clip_hi=0.0878%
head_in_ref scale=32 clip_hi=0.0350%
dep_dec0_in_ref scale=16 clip_hi=0.0190%
cf_dec0_in_ref scale=16 clip_hi=0.0370%
```

The Train32 scales broadly cover the Validate32 distribution, but a few high-activation tails still clip. The largest board-side error on sample0 is not explained by large overall clipping percentages alone; decoder accumulation and `dec2/fd2` scale choices remain the main target.

## ckpt00059 dec2/fd2 split-sum follow-up

Focus was moved to `dec2/fd2` quantization scale and split-sum glue. A standalone `dec2` probe showed that the calibrated scales are broadly usable: when `dec2` receives reference `fd3`, `fd2` is close to float reference (`abs mean/max = 0.0281 / 2.2530`). When it receives board `fd3`, standalone `dec2` is still reasonable (`abs mean/max ~= 0.14 / 3.2`).

The full board runner exposed a separate runtime/glue issue: in the continuous pipeline, `DEC2_RSZ_UP0` can produce a corrupted first output after the previous decoder stages. The same saved `fd3` input run through the standalone dec2 probe produces a correct `DEC2_RSZ_UP0` result, so this is not explained by input shape or scale alone. `DEC2_RSZ_UP1` is independent of `fd3` and was stable in the original trace.

A temporary mitigation was added to `tools/rhb_completionformer_decoder_system_runner_128x128_ckpt.py`:

```bash
--warmup-dec2-up0
--warmup-dec2-all
```

`--warmup-dec2-all` runs each dec2 RHB submodel once and discards the result before using its real output. On sample0 with `model_00059.pt` and Train32 calibration, this improved the key metrics:

```text
baseline full board fd2 abs mean/max: 0.3092 / 14.7031
warmup-all board fd2 abs mean/max:   0.1137 / 3.0551
baseline final pred abs mean/max:    0.8658 / 7.9521
warmup-all final pred abs mean/max:  0.7469 / 6.3855
```

The tradeoff is latency: tracked total increased to about `2160 ms`, because dec2 submodels are launched twice. This is useful for correctness diagnosis, but the production fix should avoid duplicate launches, likely by addressing the submodel-switch/driver-state issue around `DEC2_RSZ_UP0`.

Key artifacts:

```text
artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/board_decoder_system_128x128_ckpt00059_train32_trace_outputs.npz
artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/board_decoder_system_128x128_ckpt00059_train32_warmupall_outputs.npz
artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/nyu_sample0_ref_vs_board_pred_ckpt00059_train32_warmupall.png
```

## dec2 state-pollution scheduling modes

Additional scheduling tests confirm the dec2 corruption is a first-call/submodel-switch issue, not a fixed `DEC2_RSZ_UP0` arithmetic error:

- `normal`: first dec2 call is `DEC2_RSZ_UP0`; `up0` is corrupted.
- `reverse`: first dec2 call is `DEC2_RSZ_UP1`; corruption moves to `up1`.
- `dummy-y1`: a discarded `up1` call absorbs the first-call corruption; real `up0` becomes stable and `fd2` improves.
- `warmup-all`: discarding the first call for each dec2 submodel is currently the most stable full-pipeline workaround, but increases latency.

Mode summary is saved at:

```text
artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/dec2_state_pollution_mode_summary.tsv
```

## First-frame pollution fix

The first-frame/runtime pollution is not limited to `DEC2_RSZ_UP0`. Layerwise comparison showed the first significant corruption starts at `dec4`:

```text
baseline fd4 abs mean/max: 0.1772 / 8.8945
second-run fd4 abs mean/max: 0.0539 / 0.9632
```

A diagnostic mode was added to `tools/rhb_completionformer_decoder_system_runner_128x128_ckpt.py`:

```bash
--use-second-run
--second-run-mode from-dec4
```

`--second-run-mode from-dec4` reruns only `dec4` and all downstream submodels with the same input and uses the second output. It matches full `--use-second-run` numerically, while avoiding unnecessary reruns of `dec6/dec5`.

Best current correctness mode:

```bash
python3 rhb_completionformer_decoder_system_runner_128x128_ckpt.py \
  packer_decoder_system_128x128_ckpt00059_traincalib32_rram_false \
  --feature-npz /tmp/nyu_sample0_real_features_float_ckpt00059.npz \
  --second-run-mode from-dec4 \
  --save-npz /tmp/decoder_system_128x128_ckpt00059_train32_second_fromdec4_outputs.npz
```

Result on NYU sample0:

```text
fd2 abs mean/max:       0.0374 / 1.2243
pred_init abs mean/max: 0.0663 / 0.6798
final pred mean/max:    0.2782 / 2.4146
latency:                2621 ms
```

This establishes the error after first-frame pollution is removed. The remaining dominant gap is no longer `dec2/fd2`; it is mostly final guide/confidence quantization, especially confidence/sigmoid scale behavior.

Summary table:

```text
artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/first_frame_pollution_fix_summary.tsv
```

## wr_done clear root-cause fix

A controlled PL register sweep found the clear mechanism for stale `wr_done`:

```text
PL base:       0x800e0000
wr_done state: PL[0x20] bit0
clear strobe:  write PL[0x00] = 0x100
```

Sweep result:

```text
artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/wr_done_clear_sweep.log
```

Minimal dec4 validation:

```text
without clear: dec4 first != dec4 second
with clear:    dec4 first == dec4 second
```

The full runner now supports:

```bash
--clear-wr-done-before-run
```

This clears stale `wr_done` before every `ac_driver.run_inference()` by writing `PL[0x00]=0x100`. It removes the need for `second-run` and matches the `second-run from-dec4` numerical result.

NYU sample0 result:

```text
baseline pred abs mean/max:     0.865831 / 7.952108
second-run from-dec4 mean/max:  0.278156 / 2.414595
clear-wr-done mean/max:         0.278156 / 2.414595
second-run from-dec4 latency:   2621.167 ms
clear-wr-done latency:          1486.739 ms
```

Artifacts:

```text
artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/board_decoder_system_128x128_ckpt00059_train32_clearwrdone_outputs.npz
artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/nyu_sample0_ref_vs_board_pred_ckpt00059_train32_clearwrdone.png
```
