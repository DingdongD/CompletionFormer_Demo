# Operator Optimization Rules

This document records the rules used by the accepted CompletionFormer HW128 RHBLite deployment.

## Deployment Boundary

Accepted board pipeline:

- RHB: decoder `dec6 -> dec5 -> dec4 -> dec3`
- RHB + Host glue: `dec2` split resize/up-conv, exact block conv0/conv1, residual add and ReLU
- RHB: separate `dep_dec1`, `dep_dec0`
- RHB: separate `gd_dec1`, `gd_dec0`
- RHB: separate `cf_dec1`
- RHB: `cf_dec0` Conv only
- Host: `cf_dec0` sigmoid
- Host: NLSPN refinement and final comparison

The bundle is a runtime test package. It contains the compiled packer and does not require the compiler or Model-Packer.

## Stable RHB Patterns

- Conv2d 3x3 with padding=1 and ReLU.
- Resize + Conv2d as a hardware-aligned replacement for ConvTranspose2d.
- BasicBlock without CBAM when expressed as Conv/ReLU/Conv plus Host residual glue if needed.
- Separate full-resolution decoder heads when each head has its own activation scale.
- Conv-only confidence head followed by Host sigmoid.

## Host Glue Rules

- Keep concat on Host unless it is already inside a compiler-aligned submodel.
- For split Conv2d, dequantize partial outputs to float, sum in Host, then requantize for the next RHB submodel.
- Keep final sigmoid on Host for `cf_dec0`.
- Keep NLSPN on Host in this runtime package.
- Use Host for tensor gate multiply if the RHB tensor-tensor multiply pattern is not already validated for the exact shape.

## Exact Split Rules

- Input-channel split is mathematically exact for Conv2d:
  `Conv([x0, x1], W, b) = Conv(x0, W0, 0) + Conv(x1, W1, b)`.
- Put the bias in exactly one chunk; all other chunks must be bias-free.
- Sum partial outputs in float or in a calibrated fixed-point glue path.
- Output-channel split is also exact, but avoid it if input-channel split already passes, because it increases launch count.

## Quantization Rules

- Use 32-sample representative calibration for activation scales.
- Keep per-submodel input and output scales in `activation_scales.csv`.
- Host glue must respect the scale boundary:
  `int8 -> dequant(output_scale) -> glue op -> quant(input_scale)`.
- Do not fuse submodels unless the fused output scale has been re-calibrated and layer error is rechecked.

## Known Avoidance Rules

- Do not use fused `dep/gd/cf dec1` for ckpt00059. It compiles and runs but produces unacceptable quantization error.
- Do not use hardware sigmoid for `cf_dec0`; use Conv-only on RHB and true sigmoid on Host.
- Avoid `rram_only=true` for this pipeline.
- Clear stale `wr_done` before each `ac_driver.run_inference()` to avoid first-frame pollution.

## Runtime Rule

The accepted runner uses:

```text
--cf-dec0-host-sigmoid
--clear-wr-done-before-run
```

`--clear-wr-done-before-run` writes `PL[0x00] = 0x100` through `/dev/mem` before each submodel call.
