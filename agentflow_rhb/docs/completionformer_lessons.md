# CompletionFormer Lessons Encoded Into Rules

This document summarizes deployment lessons from `/root/demo/models/completionformer_test` and the successful HW128 ckpt00059 pipeline.

## Accepted Pipeline

```text
RHB:
  dec6 -> dec5 -> dec4 -> dec3
  dec2 resize/upconv split chunks
  dec2 exact block conv0
  dec2 exact block conv1
  dep_dec1 -> dep_dec0
  gd_dec1 -> gd_dec0
  cf_dec1 -> cf_dec0 Conv only

Host:
  concat
  split-sum glue
  residual add / ReLU glue
  cf_dec0 sigmoid
  NLSPN
```

## Rules Proven Useful

### Runtime State Must Be Cleared

The board showed first-frame pollution when switching submodels. The fix was:

```text
write PL[0x00] = 0x100 before each run_inference()
```

In runner form:

```text
--clear-wr-done-before-run
```

### Sigmoid Must Stay Host

`cf_dec0` as `Conv + Sigmoid` compiled into an incorrect activation shape. The successful rule is:

```text
RHB:  cf_dec0 Conv only
Host: true sigmoid
```

This reduced final prediction error from about `0.218` to about `0.021` on sample0.

### Input-channel Split Is Exact

For Conv2d:

```text
Conv([x0, x1], W, b) = Conv(x0, W0, b) + Conv(x1, W1, 0)
```

Use this when a large Conv times out or violates channel/weight constraints.

### Fusing Can Harm Quantization

The fused `dep/gd/cf dec1` head compiled and ran, but accuracy degraded:

```text
separate heads: final pred mean error about 0.021
fused dec1:     final pred mean error about 0.12
```

Rule:

```text
Do not fuse independently calibrated branches without re-running calibration and layerwise error checks.
```

### Approximate Rewrites Need Retraining

Large-stride PVT `srconv` and ConvTranspose replacements may be useful hardware-aligned variants, but they are not always exact.

Use categories:

```text
rewrite_exact: safe with glue scale checks
rewrite_approx: requires retraining or explicit acceptance
```

## Failure Taxonomy From Model Directory

Observed path categories:

- `failed_acsim`: compile/CModel failure or unsupported pattern.
- `archived_board_failed`: compiled or CModel-passed but failed board behavior.
- `failed_accuracy`: board ran but output was numerically unacceptable.
- `failed_full_channels`: channel/capacity pressure requiring split.

The scanner uses these directory names as weak labels.

## Reusable Success Patterns

- Conv3x3 + ReLU blocks.
- Resize + Conv for upsampling in hardware-aligned model variants.
- Full-res head Conv as separate submodels.
- Host-side glue for concat, residual add, split-sum, sigmoid, NLSPN.
- 32-sample activation calibration.
