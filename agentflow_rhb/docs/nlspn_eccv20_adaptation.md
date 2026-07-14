# NLSPN ECCV20 Compiler-aligned Adaptation

This note maps `/root/demo/NLSPN_ECCV20` onto the current AgentFlow rules.

## Source Structure

Main model:

```text
/root/demo/NLSPN_ECCV20/src/model/nlspnmodel.py
```

Important components:

```text
NLSPNModel
  conv1_rgb, conv1_dep
  ResNet layer1..layer4 encoder
  conv6
  dec5..dec2 ConvTranspose decoder
  id_dec1/id_dec0 init-depth head
  gd_dec1/gd_dec0 guidance head
  cf_dec1/cf_dec0 confidence head
  NLSPN propagation layer
```

## Allocation Decision

### RHB candidates

Use RHB for Conv-heavy image submodels after compile, cmodel, board, and
real-feature checks:

```text
conv1_rgb / conv1_dep
ResNet encoder blocks
conv6 if small-spatial high-channel checks pass or after split
decoder ConvTranspose replacement: Host resize + RHB Conv2d
id/gd/cf Conv heads as separate scale-aware submodels
```

Use `input0=BCHW` for image-like tensors.

### Host by default

Keep the actual NLSPN propagation on Host:

```text
ModulatedDeformConvFunction.apply
custom deformconv CUDA extension
dynamic offset construction
tanh / abs / sum affinity normalization
confidence modulation through deformable sampling
preserve-input mask
final clamp
```

This is not a temporary compiler issue; it is a semantic mismatch with the
current RHB rule set. The board demos have no accepted dynamic deformable
sampling primitive.

## Hardware-aligned Model Variant

Local implementation:

```text
/root/demo/models/nlspn_test/nlspn_hw_aligned.py
```

For a retrainable NLSPN-HW variant:

1. Replace `ConvTranspose2d` decoder blocks with:

   ```text
   Host bilinear/nearest resize -> RHB Conv2d(+BN)+ReLU
   ```

2. Replace `LeakyReLU(0.2)` with ReLU, or keep LeakyReLU on Host. ReLU needs
   retraining if used as an approximation.

3. Keep confidence Sigmoid on Host unless a fresh board-specific Sigmoid proof
   exists.

4. Replace deformable propagation only if retraining is allowed:

   ```text
   fixed 8-neighbor shift/gather on Host
   affinity/guidance Conv on RHB
   affinity normalization on Host
   weighted neighbor sum on Host or split RHB Conv if exact and useful
   ```

   This is a CSPN-like fixed-neighbor propagation, not exact NLSPN.

The current local aligned model follows this route. It exports a board-facing
single-output head tensor:

```text
[pred_init, guidance, confidence_logits] -> [B, 10, 128, 128]
```

Host then applies sigmoid and fixed-neighbor propagation. This keeps the RHB
graph Conv/ReLU-centric and avoids custom deformable sampling.

Local smoke output:

```text
/root/demo/artifacts/nlspn_hw_aligned/nlspn_hw_aligned_heads.onnx
```

AgentFlow reports:

```text
/root/demo/artifacts/rhb_auto_config_framework/reports/onnx_summary_nlspn_hw_aligned_heads.txt
/root/demo/artifacts/rhb_auto_config_framework/reports/onnx_annotation_nlspn_hw_aligned_heads.tsv
```

## First Deployable Split

The pragmatic first pipeline should be:

```text
RHB:
  RGB/depth stem
  encoder
  decoder
  pred_init head Conv
  guidance head Conv
  confidence head Conv-only

Host:
  concat/crop/resize glue
  confidence sigmoid
  NLSPN propagation
  clamp
```

This mirrors the successful CompletionFormer pattern: RHB for Conv-heavy
feature generation, Host for sensitive nonlinear propagation.

## Risks

- ResNet layer4 and `conv6` can hit small-spatial high-channel timeout risk.
  Apply IC/OC split or smaller block boundaries.
- Direct `ConvTranspose2d` should not be sent to RHB.
- Direct multi-output export of `pred_init/guidance/confidence` should be split
  into one output per submodel.
- Layerwise real-feature validation is required; compile/cmodel pass alone is
  not enough.

## Next Probe Order

1. Build minimal wrappers for:

   ```text
   stem_rgb
   stem_dep
   resnet layer1/layer2/layer3/layer4
   conv6
   dec5..dec2 resize+conv variants
   id/gd/cf heads
   ```

2. Compile/cmodel each wrapper with BCHW layout.
3. Board-test with `rram_only=false` and wr_done clear.
4. Stitch with Host glue and run NYU 128x128 validation samples.
5. Decide whether propagation remains Host or becomes a retrainable fixed
   neighbor aligned variant.
