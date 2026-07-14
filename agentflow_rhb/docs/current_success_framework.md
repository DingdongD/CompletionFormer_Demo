# Current Successful RHB AgentFlow Framework

This document is the current accepted rule layer for AgentFlow after the
CompletionFormer HW128 and CSPN HW128 board demos.

The framework is intentionally board-first:

```text
compile pass -> cmodel pass -> board pass -> real-feature layerwise check -> end-to-end visualization
```

CModel pass alone is not enough to promote a rule.

## Accepted Demo Cases

### CompletionFormer HW128 ckpt00059

Accepted runtime:

```text
/root/demo/artifacts/successful_pipelines/completionformer_hw128_ckpt00059_rhb_convonlycf_20260702
```

RHB allocation:

```text
decoder dec6 -> dec3
dec2 split resize/up-conv chunks
dec2 exact block conv0/conv1
dep_dec1 -> dep_dec0
gd_dec1 -> gd_dec0
cf_dec1
cf_dec0 Conv only
```

Host allocation:

```text
feature input/export
concat
split-sum dequant/requant glue
residual add
cf_dec0 sigmoid
NLSPN
visualization / app IO
```

Important rejected paths:

```text
RHB Sigmoid for cf_dec0
fused dep/gd/cf dec1
rram_only=true
run-repeat as a stale-output fix
```

### CSPN ResNetTiny HW128 aligned

Accepted package:

```text
/root/demo/artifacts/rhb_auto_config_framework/work/deployment_packages/cspn_resnettiny_hw128_w24_step8_stagewise_v3_hostsample
```

Accepted app data:

```text
/root/demo/artifacts/successful_pipelines/completionformer_hw128_ckpt00059_rhb_convonlycf_20260702/portable_runtime/outputs/cspn_unified_input
```

The CSPN app now uses the same NYU source as CompletionFormer:

```text
data/nyu_val32_source_128x128.npz
```

RGB is denormalized from the CompletionFormer RGB tensor, while sparse depth
and GT are reused directly. This prevents cross-demo color/input mismatches.

RHB allocation:

```text
BCHW image-like Conv submodels
Host-sampled downsample projections feeding RHB 1x1 Conv
stagewise Conv/refine blocks accepted by the v3 hostsample runner
padded16 final depth head as the current validated app path
```

Host allocation:

```text
spatial sample/downsample before stride2 replacements
concat/add/residual/fan-out glue
scale-aware dequant/sum/requant
CSPN propagation or high-sensitivity tail work where scheduled by runner
visualization / point cloud / BEV
```

Important rejected paths:

```text
fakequant epoch004 / QAT004 as deployment default
im2col stride-conv replacement as deployment default
internal Slice/Gather downsample inside RHB
multi-output RHB submodel
runner-only scale compensation on a qscale=1 packer
```

### NLSPN ECCV20 adaptation baseline

NLSPN is currently an onboarding target, not a promoted board-success demo.
The accepted first allocation is conservative:

```text
RHB candidates:
  RGB/depth stem
  ResNet encoder blocks
  conv6 after small-spatial high-channel checks
  ConvTranspose replacements as Host resize + RHB Conv2d
  init-depth / guidance / confidence Conv heads as separate submodels

Host by default:
  concat/crop/resize glue
  LeakyReLU unless retrained to ReLU
  confidence Sigmoid
  ModulatedDeformConvFunction propagation
  offset/affinity tanh/abs/sum normalization
  preserve-input mask and final clamp
```

Exact full offload is not supported by the current rule set because NLSPN
propagation depends on dynamic deformable sampling. A full hardware-aligned
variant would need a retrainable fixed-neighbor propagation replacement.

## Current Rule Principles

1. Clear board state before every launch.

   Use the PL wr_done clear strobe before `ac_driver.run_inference()`.
   The accepted fix is state clearing, not running the same submodel twice.

2. Use `rram_only=false` by default.

   Earlier `rram_only=true` tests failed even for simple subgraphs. Promote it
   only with direct board evidence for the exact package.

3. Keep image Conv blocks on RHB when board-proven.

   Use large enough contiguous Conv/ReLU regions to amortize launch overhead,
   but promote only after real-feature boundary validation.

4. Keep glue on Host unless a fused submodel is proven.

   Concat, residual add, split-sum, fan-out, and dequant/requant are cheap and
   sensitive to scale contracts. Host glue is the default accepted route.

5. Treat unsupported nonlinearities as Host or retrainable approximations.

   Sigmoid is Host in the accepted CompletionFormer pipeline. GELU->ReLU and
   LeakyReLU->ReLU are retrainable hardware-aligned variants. ConvTranspose
   replacement must be treated as a hardware-aligned variant unless the exact
   resize/Conv decomposition is proven for the target layer.

6. Avoid monolithic small-spatial high-channel Conv.

   For H/W <= 8 with mid/high channel counts, use IC/OC split, smaller
   boundaries, or Host fallback. Compile/CModel pass is not enough.

7. Do not keep downsample Slice/Gather inside RHB.

   CSPN success uses Host sample/downsample and feeds the sampled tensor into
   RHB Conv submodels.

8. Do not fuse independent heads by default.

   Separate head scales are part of the successful CompletionFormer path. Fused
   heads must pass layerwise and end-to-end checks before promotion.

9. Validate scale contracts with real features.

   Calibration must be implemented in compiled wrappers or explicit Host glue.
   Runner-only scaling on old packers is not a valid deployment proof.

10. Archive failed explorations as evidence, not default rules.

    Fakequant, QAT004, im2col, scalar-gain head fixes, and stale-output
    run-repeat paths are now superseded unless a future experiment produces
    fresh board evidence.

11. Keep custom deformable sampling on Host.

    `ModulatedDeformConvFunction` and related dynamic offset propagation are
    not part of the current board-proven operator set. Use Host propagation or
    redesign/retrain as fixed-neighbor propagation before considering RHB.

## Required Promotion Checklist

```text
[ ] ONNX/PyTorch submodel export is reproducible
[ ] compile passes
[ ] cmodel passes
[ ] packer config uses accepted runtime settings
[ ] board run passes without run-repeat correctness dependency
[ ] real-feature layerwise board-vs-host check passes
[ ] end-to-end sample-set metrics are generated
[ ] visualization is added to the app/static demo when user-facing
[ ] rule DB is updated only with board-backed evidence
```
