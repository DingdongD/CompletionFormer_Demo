# Framework Design

This framework implements a reusable version of AGENT-flow v2 for RHB-style black-box accelerators.

## Goal

Given a new model, produce a deployable CPU/RHB schedule:

```text
Original model
  -> graph import and shape annotation
  -> rule-based compatibility annotation
  -> capacity-aware partition
  -> submodel export
  -> compile/CModel validation
  -> board validation
  -> rule feedback and re-plan
  -> portable runtime/demo package
```

## Core Assumption

The RHB stack is a black box:

- compiler cannot be modified
- runtime cannot be modified
- operator support must be discovered empirically
- board behavior overrides compiler/CModel behavior

## Components

### Rule DB

`rule_db/rhb_blackbox_rules.seed.json` stores rules with:

- pattern
- compile status
- board status
- decision
- action
- evidence
- risk

Rules are intentionally board-centric. `compile_status=pass` is not sufficient when `board_status=fail_accuracy` or `board_status=fail_timeout`.

### Inventory Scanner

`scan-models` classifies historical model files by directory and filename:

- accepted candidate
- failed board
- failed compile/CModel
- failed accuracy
- experiment

This turns the existing `models/` history into a reusable weakly-labeled dataset.

### Planner

The current planner renders an initial schedule from a case spec and rules. The next step is graph-level planning:

1. Parse ONNX/PyTorch graph.
2. Annotate each node as `allow`, `host`, `rewrite_exact`, `rewrite_approx`, `probe`, or `forbid`.
3. Build maximal contiguous RHB regions.
4. Split by packed-weight budget and risky shape/layout rules.
5. Add Host boundary glue.
6. Emit submodel exporter and scheduler plan.

## Decision Lattice

```text
forbid
  Highest priority. Never place on RHB.

host
  Keep on Host unless an exact board-pass rule exists for this shape.

rewrite_exact
  Semantics preserving rewrite. Safe if quant glue is calibrated.

rewrite_approx
  Hardware-aligned approximation. Requires retraining or explicit accuracy acceptance.

allow
  Candidate for RHB region.

probe
  Unknown. Generate micro-test, compile, cmodel, board, then update rule DB.
```

## Capacity Model

Every RHB region must satisfy:

```text
packed_weight_bytes <= effective_budget
```

Default:

```text
8MB * 0.9 = 7.55MB
```

Use packed artifacts, not PyTorch FP32 parameter size, when available.

## Validation Loop

The framework should treat failures as structured data:

```text
compile failure:
  parse unsupported op/layout/shape
  add or refine rule

cmodel failure:
  localize subgraph mismatch
  add quant/layout risk

board failure:
  parse timeout/counter/output mismatch
  override CModel rule with board-fail evidence

accuracy failure:
  compare layerwise outputs
  identify activation/fusion/scale/glue root cause
```

## CompletionFormer-derived Priorities

1. Prefer large block-level RHB regions over single-op offloads.
2. Keep expensive full-res Conv heads on RHB when scales are independent.
3. Do not fuse heads unless calibration and layerwise error pass.
4. Use Host for cheap nonlinearities if RHB lowering is semantically wrong.
5. Clear runtime state before every board launch.

## Current Successful Demo Layer

The active rule DB is now based on two board-visible demos:

- CompletionFormer HW128 ckpt00059 conv-only confidence-head pipeline.
- CSPN ResNetTiny HW128 stagewise v3 hostsample pipeline with unified NYU
  inputs.

Rules promoted from old exploratory work must satisfy the current promotion
checklist in `docs/current_success_framework.md`. The following are no longer
default strategies:

```text
fakequant/QAT004 deployment defaults
im2col stride-conv replacement defaults
stale-output run-repeat
runner-only scale compensation on qscale=1 packers
fusing independent heads before layerwise/end-to-end validation
```

These paths are retained only as archived evidence under
`archive/superseded_experiments_20260710/`.
