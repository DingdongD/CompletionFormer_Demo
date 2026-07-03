# CompletionFormerTiny HW-Aligned Reference

This package is the training-side reference for the current compiler-aligned Host/RHB deployment path.
It is intentionally separate from `CompletionFormer/ref_model` so the original model stays unchanged.

## Entry Point

```python
from ref_model_hw import CompletionFormerHWTiny
```

Smoke test:

```bash
python tools/check_completionformer_hw_forward.py
```

Expected output includes:

```text
COMPLETIONFORMER_HW_FORWARD_OK: True
pred (1, 1, 128, 128)
guidance (1, 8, 128, 128)
confidence (1, 1, 128, 128)
```

## Hardware-Aligned Changes

Compared with the original CompletionFormerTiny:

- Decoder `ConvTranspose2d + BN + CBAM BasicBlock` is replaced by `bilinear resize + Conv2d + NoCBAM residual block`.
- Default HW-aligned image size is `128x128`; decoder feature sizes are `8x8`, `16x16`, `32x32`, `64x64`, and `128x128`.
- `dec2` uses the same topology as the validated opt2 split path: resize+up-conv, block conv0, block conv1, residual ReLU. Deployment splits resize+up-conv by input channels.
- `dep_dec1` and `gd_dec1` use the validated full `96->32` head convs instead of output-channel chunks.
- `cf_dec1/cf_dec0` remain direct small heads.
- NLSPN is included inside `CompletionFormerHWTiny` as `self.prop_layer` when `prop_time > 0`. It is a Host-side PyTorch implementation using a differentiable `grid_sample` fallback instead of the external DCN extension.

## Encoder Note

When `timm/mmcv` are installed, `BackboneHWTiny` uses the original `PVTTiny` encoder implementation.
In the current demo environment those dependencies are missing, so a lightweight fallback encoder is used for smoke tests only.
The fallback preserves feature shapes/channels but is not the final trainable encoder for quality experiments.

For real retraining, install the original CompletionFormer training dependencies and use the real `PVTTiny` path.

## Remaining Work Before Training Checkpoint Export

1. Move the currently validated stage1-4 RHB/Host alignment rules into a trainable PVTTiny-HW encoder, especially SR downsample policy, ReLU/GELU choice, and CBAM Host policy.
2. Train `CompletionFormerHWTiny` or the final PVTTiny-HW version from scratch/fine-tune.
3. Export trained weights into compiler-aligned submodel files.
4. Run stage-by-stage golden against board outputs under quantization tolerance.
5. Keep NLSPN inside the model as a Host-side PyTorch stage unless the CUDA DCN extension is available and explicitly selected.
