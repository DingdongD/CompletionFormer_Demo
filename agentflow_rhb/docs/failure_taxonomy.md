# Failure Taxonomy

Use this taxonomy to classify each failed probe or deployment attempt.

## compile_fail

The compiler cannot produce artifacts.

Common causes:

- unsupported operator
- unsupported attribute
- unsupported layout
- shape inference/export issue
- compiler crash

Framework response:

- mark pattern as `host`, `rewrite`, or `probe` with narrower shape constraints
- keep evidence log

## cmodel_fail

Compilation succeeds but CModel does not match expected output or cannot complete.

Common causes:

- quantization scale mismatch
- layout mismatch
- unsupported fusion despite successful compile

Framework response:

- split subgraph
- compare layerwise tensors
- add quant/layout risk rule

## board_timeout

Board execution hangs or reports status wait timeout.

Common causes:

- runtime state not cleared
- unsupported instruction sequence
- shape/channel pressure
- stale DMA/status register

Framework response:

- ensure `clear_wr_done_before_run`
- test smaller subgraph
- split input/output channels
- mark exact shape as board-fail if repeated

## board_stale_output

Board returns a previous tensor or unmodified input-like tensor.

Common causes:

- stale `wr_done`
- output buffer not refreshed
- wrong output address
- instruction did not run

Framework response:

- clear status before launch
- check counters and frame counts
- force output buffer invalidation if runtime supports it

## fail_accuracy

Board runs and returns output, but numeric error is too high.

Common causes:

- RHB-internal Slice/sample/downsample before Conv
- activation semantic mismatch
- branch fusion with incompatible scales
- approximate rewrite without retraining
- Host glue scale mistake

Framework response:

- compare layerwise tensors
- isolate first divergent submodel
- if the divergent submodel contains `x[:, :, ::2, ::2]`, `Slice`, or `Gather`, move sample/gather to Host and re-export RHB with sampled tensor input
- move activation/glue to Host
- split fused branches
- recalibrate scales

## outlier_driven_activation_compression

Board execution can be correct enough to avoid timeout and still lose accuracy
because rare high-magnitude activations are not represented by the accepted int8
boundary contract.

Failure signature:

- early boundaries have high correlation and comparable ranges
- a later boundary suddenly shows high-value compression:
  `board_absmax / ref_absmax < 0.35`, or `board_p99.9 / ref_p99.9 < 0.45`
- downstream decoder/refine/head tensors have low correlation even when ReLU
  outputs remain non-negative
- output is visually plausible but misses high-depth/high-guidance structures

Board-proven CSPN evidence:

- QAT epoch004 val23 accepted split, unit-scale path:
  - `s3`: corr `0.9389`, board max `6.776`, ref max `8.841`
  - `s4`: corr `0.5473`, board max `5.683`, ref max `24.318`
  - `dec2`: corr `0.1980`, board max `19.881`, ref max `110.411`
  - `refined`: corr `0.1016`, board max `30.896`, ref max `189.775`
  - `guidance`: corr `0.1659`, board max `14.744`, ref max `132.774`
- ReLU was not the primary fault in this case:
  - ReLU-bearing boundary minima stayed at `0`
  - major residual `add+relu` glue was executed on Host in the runner
  - the observed signature was range compression, not sign leakage

Framework response:

- run `analyze-outliers` after board/ref layerwise comparison
- report `p99`, `p99.9`, max, compression ratios, top-0.1% energy share, and
  int8 saturation rate for each boundary
- do not attribute this signature to ReLU unless negative leakage or a direct
  board-vs-csim ReLU mismatch is observed
- train/QAT against the exact accepted split contract, including every Host/RHB
  boundary and Host add/relu boundary
- evaluate max-scale and p99.9-scale candidates by end-to-end board-vs-ref,
  because percentile-only scales may preserve normal texture while clipping
  structural high-depth outliers
- if widening the scale destroys normal-range resolution, split the region or
  move the boundary to reduce activation dynamic range before RHB execution

## scale_mismatch_runner_only

Changing only the runner to multiply inputs by calibrated activation scale and divide outputs by calibrated output scale is valid only when the compiled submodel was exported with matching scale compensation.

Failure signature:

- local float Q/DQ simulation is accurate
- board output range is compressed or inflated immediately at the first scaled boundary
- `board_raw / output_scale` diverges, while `board_raw` itself matches the old qscale=1 interpretation

Board-proven CSPN evidence:

- original CSPN `stem_ckpt_inscale_fold` already folds only the image input scale into the first Conv
- its board raw output range max was about `36`, matching local `s1` float max about `37`
- dividing that output by calibrated `s1_scale=3.425` incorrectly compressed `s1` to max about `10.5`
- full calibration-scale runner produced `board_vs_gt L1=3.145 RMSE=3.345`
- unit-scale runner, matching current compiled packers, recovered the old baseline but remained inaccurate: `board_vs_gt L1=1.075 RMSE=1.509`
- local ideal boundary Q/DQ with the same calibration scales was good: ref `L1=0.186 RMSE=0.361`, Q/DQ `L1=0.196 RMSE=0.373`

Framework response:

- do not use alternate checkpoint probes as a shortcut unless they are trained and validated with the same per-boundary scale contract
- do not apply CompletionFormer-style runner scales to qscale=1 packers
- for real scale-aware execution, re-export each RHB submodel with compensation:
  `submodel_scaled(x_i8) = module(x_i8 / input_scale) * output_scale`
- prefer folding `1 / input_scale` into the first Conv weights when the first op is Conv
- fold `output_scale` only through compiler-safe forms; CSPN probe showed direct graph `Div/Mul`, BN gamma/beta scaling, and post-ReLU 1x1 scaling all triggered compile/quant failures for the tested stage2 probe
- if scale compensation cannot compile for a submodel, keep that boundary at unit scale or move the boundary until the compiler accepts the compensated graph

## board_effective_scale_mismatch

The compiled submodel and runner agree on a nominal activation scale contract,
but real board output behaves as if a different output scale was used.

Failure signature:

- board-vs-reference correlation is high, but amplitude is consistently too
  small or too large;
- `output_i8` looks like a scaled version of quantized reference output;
- least-squares effective scale differs from nominal scale by more than 20%;
- CModel may pass, because the mismatch appears only in board/runtime behavior
  or in real-feature board comparison.

Board-proven CSPN evidence:

- `stage2_b0_main`: nominal `5.6245`, effective `2.8006`, ratio `0.4979`
- `stage3_b1_conv2`: nominal `9.7195`, effective `19.4419`, ratio `2.0003`
- `dec1_conv0`: nominal `1.2539`, effective `0.6264`, ratio `0.4995`
- `depth_head_conv1_padded`: nominal `6.5144`, effective `0.7777`, ratio `0.1194`

Framework response:

- run a real-boundary effective-scale probe;
- estimate per-submodel effective scales with saturated samples excluded;
- only use per-output effective-scale overrides if ratios are stable across
  representative samples;
- if ratios are unstable or head layers amplify residual error, move the
  boundary to Host instead of stacking more compensation.

## head_error_amplification

A prediction head compiles and may even have high single-layer correlation, but
it amplifies upstream feature error enough that end-to-end output is poor.

Failure signature:

- upstream feature such as `refined` is highly correlated with reference;
- head output has lower correlation or large visible range error;
- moving only the final Conv to Host gives little improvement;
- moving the whole head to Host gives a large end-to-end improvement.

Board-proven CSPN evidence:

- per-output effective scales recovered `refined` to corr `0.9866`
- RHB depth + RHB guidance: final pred corr `0.8975`
- Host final depth Conv only: final pred corr `0.9039`
- Host full depth head + RHB guidance: final pred corr `0.9525`
- Host full depth + Host full guidance: final pred corr `0.9779`

Framework response:

- test `RHB body -> Host head` as a first-class allocation candidate;
- keep small/high-sensitivity prediction heads on Host when accuracy dominates
  the extra Host compute cost;

## partial_sum_post_scale_mismatch

Input-channel split Conv can pass compile, CModel, and board execution while the
Host partial-sum tensor is quantized with the wrong scale before a downstream
BN/ReLU/post submodel.

Failure signature:

- one partial branch has saturation or large amplitude error;
- correcting that partial output scale alone makes end-to-end output worse;
- the next post submodel has a very low input scale because it was calibrated
  on the old partial-sum amplitude;
- jointly tuning partial output scale and post input scale sharply improves the
  head or final prediction.

Board-proven NLSPN evidence:

- `id_dec1_partial_oc0_64_ic0_64` had about `5.4%` saturated raw values on
  sample00 and large partial L1 error.
- Changing only its corrected output scale worsened sample00 final pred L1 from
  `0.407606` to `0.936520`.
- Jointly using `corrected_output_scale=7.20130917` for that partial and
  `input_scale=4.0` for `id_dec1_post_bn_relu_oc0_64` improved val32 final pred
  L1 mean from `0.366076` to `0.139718`, with worst sample `0.210052`.

Framework response:

- when a Host sum feeds an RHB post op, create a joint scale-search task;
- sweep post input scale candidates on representative samples;
- rank by real end-to-end board metrics, not just local layerwise L1;
- promote the joint tuple only after val32/128 confirms it is stable.
- prefer RHB for the wide Conv body up to a stable semantic feature boundary
  such as `refined`.

## layerwise_head_divergence

A head block is not uniformly bad: early layers may match Host closely, while a
later layer causes most of the output error.

Failure signature:

- the upstream boundary tensor is already accurate;
- the first head layer has high correlation and low relative error;
- the full head output drops sharply;
- moving only the first bad layer may improve the local tensor, but the final
  metric still needs downstream sensitivity checks.

Board-proven CSPN evidence:

- `depth_head conv0` from board `refined`:
  `L1=0.0974`, `RMSE=0.1699`, `corr=0.9993`
- full `depth_head` from the same board `refined`:
  `L1=1.5384`, `RMSE=1.8353`, `corr=0.8950`
- therefore the first depth-head divergence is the final Conv, not the
  ConvBNReLU layer.

Framework response:

- compare each candidate layer against Host using the same real boundary input;
- mark the first divergent layer as a candidate split point;
- still evaluate final task output, because downstream propagation may amplify
  even high-correlation head errors.

## channelwise_scale_mismatch

The whole tensor may look well correlated, but individual output channels have
different scale/offset errors. This matters for guidance, affinity, attention,
or other tensors used as control weights.

Failure signature:

- full tensor correlation is high;
- several channels have materially lower correlation or shifted mean/std;
- per-channel scale/bias correction improves the downstream metric more than a
  global correction.

Board-proven CSPN evidence:

- board guidance vs Host guidance: overall `corr=0.9949`
- weaker channels included `ch3 corr=0.8887`, `ch4 corr=0.9304`,
  `ch7 corr=0.9486`
- final pred:
  - original board depth + board guidance: `corr=0.8975`
  - board depth + per-channel corrected guidance: `corr=0.9505`
  - full Host depth + Host guidance: `corr=0.9779`

Framework response:

- run per-channel diagnostics for multi-channel control tensors;
- accept channel-wise Host correction only if stable across calibration and
  validation;
- otherwise move the control/guidance head to Host.

## rhb_internal_slice_downsample

Spatial Slice/Gather/downsample inside an RHB submodel can compile and pass random CModel/board smoke tests but fail badly on real feature distributions.

Board-proven response:

- do exact sample/gather/indexing on Host
- feed the already-sampled tensor to RHB
- keep only Conv/BN/ReLU in the RHB submodel
- use `run_repeat=2` or clear runtime state until the driver-side first-run issue is fixed

Evidence:

- CSPN stage2 original `stage2[0].conv1(s1)` and `stage2[0].shortcut(s1)` included `x[:, :, ::2, ::2]` inside RHB and produced real-feature corr about `0.03-0.04`
- PyTorch vs float ONNX matched (`corr ~= 1.0`), so BN folding/export was not the cause
- CPU 8-bit folded-weight simulation stayed high-correlation (`corr ~= 0.94`)
- Host sample + RHB sampled `[1,24,64,64] -> 1x1 Conv+BN(+ReLU)` restored b0 main/shortcut to `corr ~= 0.998`, max diff `1`
- Full stage2 sampled runner improved final stage2 from `corr ~= 0.06` to `corr ~= 0.807`

## conv_bn_relu_fusion_accuracy

Some Conv+BN+ReLU fusions can compile and pass CModel but fail board-vs-csim exactness for specific parameter/channel ranges.

Board-proven response:

- split the fused region at the quantization boundary
- run only the Conv projection on RHB
- run BN+ReLU on Host with exported BN parameters
- treat this as an execution-correctness repair, not automatically as an accuracy repair

Evidence:

- CSPN stage3 `b0_main` fused `sampled_s2 -> Conv1x1(48->96)+BN+ReLU` failed board-vs-csim: `corr ~= 0.894`, `maxdiff=127`
- output-channel split into 48 and then 24 channels did not fix the first half
- projection-only `sampled_s2 -> Conv1x1(48->96)` passed board-vs-csim exactly
- replacing fused stage3 main with `RHB Conv1x1 + Host BN/ReLU` made downstream 3x3 stages board-stable, but inserted an extra int8 boundary and did not improve end-to-end prediction accuracy without an accepted scale-compensated boundary

Required follow-up:

- propagate calibrated activation scale across the new Conv->Host BN boundary
- prefer exact compiler-aligned rewrites plus real-feature board validation; retraining is only for deliberate approximate rewrites
- compare board to cmodel for execution, and quantized graph to float ckpt for accuracy; do not use naive PyTorch rounding as final proof

## capacity_or_channel_pressure

Submodel exceeds packed-weight budget or triggers large-channel issues.

Framework response:

- exact input-channel split
- exact output-channel split only if necessary
- avoid extra launch count if input split is sufficient

## conv_stride2_timeout

Stride-2 Conv can compile and pass CModel but timeout on board, especially when it produces 32 or more output channels.

Board-proven exact response:

- keep the Conv semantics exact by moving only offset/sample gather to Host
- run each sampled kernel offset as RHB Conv1x1 partials
- im2col is the flattened/batched form of the same offset decomposition
- for current CSPN down20, use output chunk 8 and flat im2col input chunk 8
- Host sums partial outputs and adds the original bias

Evidence:

- CSPN down20 original stride-2 Conv failed on board
- exact im2col+Conv1x1 with flat chunk 8 passed
- flat chunks 16, 32, 48, 72, and 144 timed out
- CompletionFormer used the equivalent per-offset sample gather + Conv1x1 rule for SRConv

Retrainable low-launch response:

- replace stride-2 3x3 Conv with fixed Host even-grid sample plus RHB 1x1 Conv
- keep output shape unchanged: `[B,16,16,16] -> [B,32,8,8]`
- do not run the full `[16 -> 32, 8x8]` 1x1 Conv as one RHB submodel; it was board-inaccurate
- split the 1x1 Conv by input channel into two `[8 -> 32, 8x8]` partials
- Host sums the two partials, adds bias, then applies ReLU

Evidence:

- `models/cspn_test/failed_board/cspn_hwdown20_sample_1x1.py`: compile/cmodel pass, board `All same: False`
- `models/cspn_test/cspn_hwdown20_sample_1x1_ic8.py`: compile/cmodel pass, board `All same: True`
- expected down20 launches drop from 18 exact offset partials to 2 retrained 1x1 partials

## small_spatial_high_channel_conv

Small spatial Conv at 8x8 with 32 channels can compile and pass CModel but timeout on board when the input-channel block is too wide.

Board-proven exact response:

- split Conv by input channels and output channels
- use input chunk 8
- output chunk can be 16 or 32 for the tested CSPN down21/down22 blocks
- Host sums input-channel partial outputs, adds the original bias, then concatenates output chunks

Evidence:

- output-only split still timed out
- input chunk 8 with output chunk 16 or 32 passed
- input chunk 16 or 32 timed out, even with smaller output chunks
