# Production SPN Deployment Rules

This note consolidates the CompletionFormer, CSPN, and NLSPN deployment policy
into one rule set for future SPN-style depth completion models.

## Semantic Policy

Use this priority order:

1. Exact compiler-aligned runtime rewrite.

   Examples: input/output channel split, pad-and-slice, single-output split,
   Host gather/sample plus RHB 1x1 when it is mathematically identical, Host
   partial-sum plus bias/BN/ReLU represented by a validated RHB affine/ReLU
   block.

   Requirement: software reference equivalence, compile/cmodel pass, board pass,
   and real-feature boundary trace.

2. Approximate compiler-aligned model rewrite.

   Examples: GELU to ReLU, ConvTranspose to resize plus Conv, stride Conv to
   Host sample plus RHB 1x1 when it changes the operator, deform propagation to
   fixed-neighbor or simplified propagation.

   Requirement: the approximation must be present in the model source and
   checkpoint. It cannot be hidden in the runner. Retrain or QAT is required.

3. Host fallback.

   Use Host for dynamic indexing, deformable sampling, unsupported nonlinear
   normalization, gate multiply, or propagation when exact decomposition is not
   proven and approximate retraining is not in scope.

## Quantization Gate

Before board promotion, dump calibration32/64/128 boundary features and run:

```bash
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py quant-diagnostics \
  --npz <calibration_features.npz> \
  --channel-axis 1
```

Inspect:

- absmax vs p99/p99.9 scale gap
- QDQ L1/RMSE under absmax and percentile scales
- int8 saturation rate
- skewness and kurtosis
- top 0.1% absolute energy ratio

High-risk tensors need one of these decisions before board work continues:

- split high-energy channels
- fuse or move a boundary
- keep the fragment on Host
- add QAT or retrain with the exact split contract
- collect real-feature board traces and fit a model-owned scale contract

## Maximal RHB Subgraph Search

The search objective is not "every Conv on RHB"; it is "largest board-validated
RHB regions with minimal Host/RHB round trips."

Search loop:

1. Import ONNX and annotate nodes using the rule DB.
2. Grow RHB regions through allowed Conv/ReLU/BN-fold/token-friendly blocks.
3. Stop growth on unsupported ops, multi-output board limits, 8MB packer limits,
   high quant risk, or known timeout patterns.
4. Apply exact split/pad/single-output rewrites first.
5. Compile, run CModel, pack, and board-test candidates.
6. Run real-feature boundary validation, not only `csim_input`.
7. Promote only if end-to-end val32 visualization and metrics pass.

## CSPN/NLSPN Notes

CSPN and NLSPN should follow CompletionFormer's scale-aware runner discipline:

- no stale ONNX or stale packer artifacts
- no hidden runtime affine compensation
- `rram_only=false` by default
- clear `wr_done` before each board launch
- single-output RHB submodels unless a multi-output board path is proven
- propagation stays Host unless exact decomposition or retrained approximation is
  accepted

For NLSPN, current priority remains pred-init/id-decoder boundary scale and
large-spatial high-channel Conv splitting. For CSPN, current priority is reducing
small launches while keeping exact or checkpoint-owned semantics.

## Remote Training Template

Approximate rewrites must be trained on the remote GPU server, then fetched back
before export/board validation. The default profiles live in:

```text
artifacts/rhb_auto_config_framework/configs/remote_training_profiles.json
```

Render and review commands first:

```bash
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py remote-train \
  --profile cspn_resnettiny_hw128_sample1x1 \
  --action plan
```

Then the standard lifecycle is:

```bash
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py remote-train --profile <profile> --action submit
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py remote-train --profile <profile> --action status
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py remote-train --profile <profile> --action fetch
```

The template uses:

```text
ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -p 2222 root@100.104.115.57
```

Fetched checkpoints are stored under:

```text
artifacts/rhb_auto_config_framework/remote_ckpts/
```

After fetching, rerun the production loop from fresh export. Do not compare board
outputs against stale ONNX or stale packer artifacts.
