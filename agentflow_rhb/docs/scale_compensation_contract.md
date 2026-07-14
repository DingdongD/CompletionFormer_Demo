# Scale Compensation Contract

CompletionFormer-style scale-aware execution has two parts:

1. the compiled RHB submodel must implement the same scale contract used by the runner
2. the runner must quantize/dequantize only according to that validated contract

Runner-only scaling is not valid for an existing qscale=1 packer.

## Contract

For an RHB submodel with float operation `module(x)`, input activation scale `Si`, and output activation scale `So`, the compiled submodel must satisfy:

```text
y_i8 ~= round(module(x_i8 / Si) * So)
y_float = y_i8 / So
```

Only after this compile-time contract is validated may the runner do:

```text
x_i8 = round(x_float * Si)
y_float = rhb(x_i8) / So
```

## Compiler-Safe Candidate Order

Do not generate explicit ONNX `Div`/`Mul` around the submodel by default. For this compiler path, that is a high-risk pattern.

Candidate order:

1. `conv_input_fold_conv_output_fold`
   - fold `1 / Si` into the first Conv weights/bias
   - fold `So` into the final Conv weights/bias
   - use when the submodel starts and ends with Conv-compatible operators

2. `conv_input_fold_bn_output_fold`
   - fold `1 / Si` into the first Conv weights/bias
   - fold `So` into terminal BN gamma/beta
   - mathematically valid before ReLU for positive `So`, but must pass compile/cmodel/board

3. `conv_input_fold_host_output`
   - fold `1 / Si` into the first Conv weights/bias
   - do not fold output scale in RHB
   - this is a fallback contract; output scale is native/unit unless a downstream compiled consumer absorbs it

4. `unit_scale`
   - keep qscale=1 for this boundary
   - use only as a known baseline or when no scale-compensated candidate compiles

## Single-Conv Bias Rule

For a single Conv submodel implementing:

```text
y = W * x + b
```

and the desired compiled contract:

```text
y_i8 ~= round((W * (x_i8 / Si) + b) * So)
```

the weight and bias transform is:

```text
W' = W * So / Si
b' = b * So
```

Do not divide the bias by `Si` for `single_conv_io_fold`.

This differs from a multi-op graph where the first Conv's bias participates in
the intermediate activation before later ops. In those cases, folding `1/Si`
into the first Conv bias can be valid, but only if the overall graph contract is
then validated by compile/cmodel/board.

Board-proven CSPN evidence:

- `depth_head_conv1_padded16` old and bias-corrected packers had nearly the same
  final error, so bias was not the dominant CSPN failure.
- The rule is still mathematically required and should be used for future
  generated single-Conv wrappers.

## Single-Output Conv Head Exception

For high-sensitivity final prediction heads such as:

```text
Conv2d(C -> 1, kernel=3, padding=1, bias=True)
```

the normal `single_conv_io_fold` contract can still fail on real features even
when export, compile, CModel, and board execution all complete. CSPN
`depth_head[1]` showed this exact case:

```text
plain24 single_conv_io_fold: board range [-1,2], L1=1.1078, corr=0.8631
explicit output Mul:         board range [-1,2], L1=0.9355, corr=0.8985
weight gain 8/32/64:         no effect; compiler normalized int params
```

The board output matched the CModel integer arithmetic exactly, so this is a
compile-time integer scale-contract problem, not a runtime/driver issue.

Preferred exact rewrite:

```text
RHB:
  partial_i = Conv2d(x[:, group_i], W[:, group_i], bias=False)

Host:
  y = sum_i dequant(partial_i, effective_scale_i) + original_bias
```

For CSPN `depth_head[1]`, six group4 partial Conv submodels produced:

```text
L1=0.1396, RMSE=0.2557, corr=0.9965
```

against the float final Conv on a real `depth_head_mid` tensor. This rewrite is
strictly linear/exact before quantization and should be considered before
moving the whole head to Host. The per-partial effective scales must be
estimated on representative calibration tensors and checked for stability.

Follow-up 32/128-sample calibration showed the effective/input ratios are
stable and close to 0.25, but group4 does not beat the padded16 final-depth
candidate in the current CSPN QAT004 path:

```text
train128 LS ratios:
[0.24967800, 0.24851626, 0.24891064, 0.24936840, 0.24890906, 0.25080055]

val0 pred board-vs-ref:
padded16          L1=0.169498, corr=0.983091
group4 train128  L1=0.185707, corr=0.980819
```

Therefore group4 is a validated fallback/diagnostic, not the default
accuracy-first replacement for padded16. Also ensure generated packers use the
same checkpoint as the reference tensors; an early group4 calibration used an
older default checkpoint and had to be rebuilt with `CSPN_HW_CKPT` set to the
QAT004 checkpoint.

## Validation Gate

A generated wrapper is not usable until all gates pass:

```text
export -> compile -> cmodel -> board-vs-csim -> real-feature layerwise error
```

The makefile may ignore compile failures, so the gate must check for generated `op_insts.bin`, absence of compiler assertions, and valid cmodel completion.

For CSPN NCHW image-like submodels, use `layout=input0=BCHW` or an empty layout. `layout=input0=CHW` can trigger a compiler assertion even for an otherwise valid original Conv-only projection.

## Partial-Sum Post Input Scale

For input-channel split Conv blocks, the scale contract has an extra boundary:

```text
RHB partial_i = Conv(x[:, ic_i])
Host acc      = sum_i dequant(partial_i, partial_output_scale_i)
RHB post      = BN/ReLU/affine(acc)
```

Do not calibrate only `partial_output_scale_i`. The downstream post submodel's
`input_scale` must be calibrated against the actual Host partial-sum tensor that
will be fed to RHB.

Board-proven NLSPN evidence:

```text
path: id_dec1 partial-sum -> post BN/ReLU -> pred_init

limited val32 scales:
  val32 pred_l1 mean 0.366076, worst 0.511881
  sample00 pred_init_l1 0.700597

enable only id_dec1_partial_ic0 corrected_output_scale=7.20130917:
  sample00 pred_l1 0.936520
  sample00 pred_init_l1 1.483736

joint boundary scale:
  id_dec1_partial_oc0_64_ic0_64 corrected_output_scale=7.20130917
  id_dec1_post_bn_relu_oc0_64 input_scale=4.0

result:
  val32 pred_l1 mean 0.139718, worst 0.210052
  sample00 pred_init_l1 0.267775
```

AgentFlow should therefore search the tuple:

```text
(partial output scale candidates, post input scale candidates)
```

and rank by end-to-end board-vs-reference metrics. This is especially important
for prediction-init and guidance heads where moderate boundary error can
dominate final depth.

## Calibrated Host Affine Glue

When a boundary remains biased after RHB scale-contract fixes, a calibrated Host
affine can be a valid deployment glue step:

```text
y_corrected = clip(alpha * y_board + beta, valid_min, valid_max)
```

Use this only when:

- the error is stable across representative calibration samples;
- the affected tensor is already crossing Host/RHB boundary;
- the correction is explicit in the runner configuration;
- val32/128 confirms end-to-end improvement.

Board-proven NLSPN evidence:

```text
baseline after pred-init and guidance joint scale:
  val32 pred_l1 mean 0.137466, worst 0.185438

pred_init Host affine:
  alpha = 0.8865426182746887
  beta  = 0.45262056589126587

after affine:
  val32 pred_l1 mean 0.101815, worst 0.155142
  board sample00 pred_l1 0.094380
```

This rule is not a replacement for fixing a broken RHB submodel. It is a final
boundary-compensation candidate after layerwise traces have identified a stable
amplitude/bias residual.

## Outlier-Aware Scale Selection

Activation scale selection must include outlier diagnostics, not only global
min/max or a single calibration sample.

For every Host/RHB boundary, collect at least:

```text
p99, p99.9, absmax
board_absmax / ref_absmax
board_abs_p99.9 / ref_abs_p99.9
top-0.1% absolute energy ratio
int8 saturation rate
```

Recommended interpretation:

- `board_absmax / ref_absmax < 0.35` means the board path is probably
  compressing structural outliers.
- `board_abs_p99.9 / ref_abs_p99.9 < 0.45` means the high tail is compressed,
  even if max is noisy.
- high top-0.1% energy means outliers are not ignorable; clipping them can
  change the prediction, not just a few pixels.
- visible ReLU non-negativity is not enough to clear a boundary; range
  compression can happen while all ReLU outputs remain `>= 0`.

Candidate scales must be selected by end-to-end board-vs-ref and layerwise
error, not by calibration histograms alone:

```text
scale_max      = 127 / max(abs(x_ref))
scale_p999     = 127 / percentile(abs(x_ref), 99.9)
scale_p99      = 127 / percentile(abs(x_ref), 99)
```

Use `scale_max` when high-value structures are semantically important and
outlier energy is high. Use percentile scales only when layerwise and
end-to-end validation show that the clipped tail is harmless.

AgentFlow command:

```text
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py analyze-outliers \
  --ref-npz REF_TENSORS.npz \
  --board-npz BOARD_TENSORS.npz \
  --keys s4_float,dec3_float,dec2_float,dec1_float,refined_float,depth_float,guidance_float
```

For historical layerwise CSVs:

```text
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py analyze-outliers \
  --boundary-csv path/to/boundary_error_compare.csv
```

## CSPN 2026-07-09 Evidence

Local ideal Q/DQ using calibrated scales was accurate:

```text
ref pred: L1=0.186 RMSE=0.361
Q/DQ pred: L1=0.196 RMSE=0.373
```

Runner-only scale on existing qscale=1 packers failed:

```text
calibration-scale runner: board_vs_gt L1=3.145 RMSE=3.345
unit-scale runner:        board_vs_gt L1=1.075 RMSE=1.509
```

Generated CSPN stage2 Conv+BN+ReLU candidates currently fail compile:

```text
conv_input_fold_bn_output_fold:   LLVM ArrayRef assertion
conv_input_fold_host_output:      LLVM ArrayRef assertion
```

Conv-only projection candidates with `single_conv_io_fold` pass the full initial gate:

```text
stage3 main projection: board All same True, real-feature L1=0.006363, corr=0.999517
stage4 main projection: board All same True, real-feature L1=0.002449, corr=0.999763
```

Accepted contract list:

```text
artifacts/rhb_auto_config_framework/reports/cspn_convonly_scale_compensation_accepted_contracts_20260709.tsv
```

Therefore CSPN should grow scale-aware execution outward from accepted Conv-only boundaries rather than applying runner-only scales to the full qscale=1 stagewise packer.

## Empirical Effective Scale Gate

Some RHB submodels compile and pass CModel, but board output uses a different
effective scale than the nominal compiler-aligned contract. AgentFlow must
measure the effective scale on real boundary tensors before accepting a
pipeline.

For each RHB submodel, record:

```text
input_float
input_i8
output_i8
nominal_output_scale
reference_float_output
```

Estimate:

```text
effective_scale = argmin_s || output_i8 / s - reference_float_output ||
```

Use a robust estimate that excludes near-saturated int8 samples when saturation
is present. A boundary should be marked unstable if:

- effective / nominal ratio is outside `[0.8, 1.25]`, unless the runner records
  an explicit per-output effective-scale override;
- the ratio is not stable across representative samples;
- per-output overrides improve one sample but degrade the validation set.

Board-proven CSPN val23 pattern:

```text
stage2 outputs:          ratio ~= 0.5
stage3_b0 outputs:       ratio ~= 1.0
stage3_b1_conv2:         ratio ~= 2.0
stage4_b1_conv1:         ratio ~= 1.95
dec1/refine/head_mid:    ratio ~= 0.5
depth_head_conv1_padded: ratio ~= 0.119
```

Per-output effective scales improved CSPN val23 `pred` corr from `0.2277` to
`0.8975`, but remaining head sensitivity still required a better Host/RHB
boundary.

## Head Boundary Rule

Prediction heads should not automatically remain on RHB just because their Conv
subgraphs compile. They often amplify upstream quantization and residual
boundary error.

Move a head to Host when:

- head output correlation is much lower than the preceding feature tensor;
- the head has a very small or unstable effective-scale ratio;
- moving only the final Conv to Host does not recover accuracy;
- moving the full head to Host significantly improves end-to-end metrics.

CSPN evidence on val23:

```text
RHB depth + RHB guidance:             pred corr 0.8975
Host final depth conv only:           pred corr 0.9039
Host full depth head + RHB guidance:  pred corr 0.9525
Host full depth + Host full guidance: pred corr 0.9779
```

Recommended CSPN boundary:

```text
RHB:  stem -> stage2 -> stage3 -> stage4 -> dec3 -> dec2 -> dec1 -> refine
Host: depth_head -> guidance_head -> CSPN/NLSPN
```
