# Deployment Playbook For A New Model

Use this playbook when bringing a new model to RHB.

## 1. Establish Baseline

- Identify input shape and representative calibration/eval samples.
- Run FP32 or reference PyTorch output.
- Export a small set of real intermediate tensors.

## 2. Import And Canonicalize

- Export ONNX.
- Normalize layout conventions.
- Replace training-only modules.
- Mark dynamic or unsupported control flow for Host.

## 3. First Rule Annotation

Use the rule DB:

```bash
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py summarize-rules
```

Annotate nodes:

- `allow`: known RHB-safe
- `host`: CPU fallback
- `rewrite_exact`: semantics preserving rewrite
- `rewrite_approx`: hardware-aligned variant requiring retraining/acceptance
- `probe`: unknown
- `forbid`: known bad

## 4. Build Candidate Submodels

Prefer these boundaries:

- whole residual block
- whole Conv-heavy decoder block
- full head branch with independent scale
- split Conv chunks when exact input-channel split applies

Avoid:

- tiny single-op RHB islands
- frequent Host/RHB ping-pong
- fusing branches with different activation scales
- putting spatial indexing/downsample (`Slice`, `Gather`, `x[:, :, ::2, ::2]`) inside an RHB submodel

For sample/downsample blocks:

- Host performs exact sample/gather/indexing first.
- RHB submodel input shape must already be the sampled tensor shape.
- RHB then runs Conv/BN/ReLU on that sampled tensor.
- Do not rely on compile/cmodel pass for an RHB-internal Slice downsample; CSPN stage2 passed random csim but failed on real features.

## 5. Compile And CModel

For each candidate submodel:

1. Export model file.
2. Compile.
3. Run CModel.
4. Parse logs and save artifacts.

CModel pass is only a filter. It is not final proof.

## 6. Board Validation

Run on board with:

```text
rram_only=false
clear_wr_done_before_run=true
```

Record:

- board status
- output equality or error metrics
- runtime counters
- status timeout markers
- latency per submodel

## 7. Localize Failure

If a region fails:

- split region by graph boundary
- test smallest failing subgraph
- if the first failing block contains `Slice`/stride sample before Conv, re-export with Host sample and sampled RHB input before investigating BN/residual add
- classify failure:
  - compile unsupported
  - CModel mismatch
  - board timeout
  - board output stale
  - quantization/activation semantic error
  - capacity/channel pressure

## 8. Apply Rewrite

Use the lowest-risk rewrite:

1. Host fallback.
2. Exact split.
3. Layout-only rewrite.
4. RHB Conv + Host activation.
5. Approximate hardware-aligned rewrite with retraining.

When applying `RHB Conv + Host BN/ReLU`, keep two separate pass criteria:

- execution criterion: RHB Conv output must match packer/csim exactly on board
- accuracy criterion: the added int8 boundary must be covered by an accepted scale-compensated boundary or an explicitly retrained approximate rewrite

Do not assume moving BN/ReLU to Host improves final accuracy just because it fixes board-vs-csim exactness.

## 9. Package Runtime

A deployable runtime should contain:

- compiled packer
- scheduler/runner
- activation scales
- representative input source
- reference comparison script
- board demo image
- manifest and checksums

See the CompletionFormer portable runtime as the concrete template.
