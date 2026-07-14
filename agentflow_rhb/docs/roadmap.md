# Roadmap

## v0: Seed Framework

- Seed RHB rule DB from CompletionFormer.
- Inventory scanner for historical model experiments.
- Case-spec based deployment plan renderer.
- Documents and templates.

Status: implemented.

## v1: Graph-aware Planner

- Import ONNX graph.
- Run shape inference.
- Match nodes against rule DB.
- Emit compatibility table.
- Build candidate RHB regions.

Status:

- ONNX graph importer is implemented with shape inference and packed-weight estimates.
- Initial rule-based node annotation is implemented.
- Region-building from annotated nodes is implemented.
- Effective-budget splitting is implemented for contiguous regions.
- Deployment graph emission is implemented.

## v2: Compile/Board Loop

- Generate submodel source files.
- Invoke existing ONNX compile/CModel commands.
- Package candidate packer.
- Run board probe.
- Parse logs and metrics.
- Emit rule update suggestions.

Status:

- Compile/CModel wrapper is implemented.
- Model-Packer wrapper is implemented.
- Board runner wrapper is implemented.
- RHB region ONNX export is implemented.
- Manual ONNX export fallback handles compiler-extended attributes such as `weight_ch_scales`.
- Failure localizer is implemented for pass, timeout, accuracy, compile, and cmodel classes.
- Retry planner is implemented for runtime retry, binary split, host fallback, activation split, and scale audit actions.
- PyTorch-source submodel generation and automatic execution of retry actions are pending.

## v3: Cost-aware Search

- Estimate CPU/RHB boundary cost.
- Estimate launch and transfer latency.
- Penalize ping-pong and tiny RHB islands.
- Search among exact splits and Host fallback choices.

Status:

- First-pass latency and boundary scoring is implemented.
- Directory-level ONNX candidate ranking is implemented.
- Result JSON cost calibration is implemented.
- Deep-search now enumerates exact, single-output split, and latency-oriented approximate-rewrite strategies.
- Production policy now prefers the largest board-validated RHB region, with boundary-count and launch-count penalties.
- Remaining gap: automated branch-aware global search still needs to execute compile/cmodel/board probes while expanding candidate regions, rather than only scoring static ONNX candidates.

## v4: Training-aware Variant Generation

- Mark approximate rewrites.
- Generate hardware-aligned model variant.
- Export retraining recipe for deliberate approximate rewrites.
- Validate against reference and board.

Status:

- Rewrite suggestions mark exact vs approximate and whether retraining is required.
- Source-tree profiler is implemented for early risk detection in new PyTorch models.
- New-case generator is implemented.
- Production-plan gate separates strict semantic rewrites from approximate rewrites that require retraining or QAT.
- Software quant-diagnostics is implemented for trained compiler-aligned models and calibration feature dumps.
- Remaining gap: automatic source-code generation for full HW-aligned variants is still case-specific; CSPN/NLSPN should reuse the CompletionFormer pattern but still need model-family adapters.

## v5: New Model Onboarding

- Generate case specs.
- Profile PyTorch source tree.
- Score existing ONNX probes.
- Optimize candidate ONNX graphs and export RHB regions.

Status:

- CSPN source profile and ONNX portfolio smoke tests pass.
- Depth-Anything-V2 source profile smoke test passes.
- CompletionFormer, CSPN, and NLSPN now share the same production gates: source profile, max-RHB search, exact rewrite proof, optional approximate retrain, quant diagnostics, compile/cmodel/board, real-feature trace, val32 visualization.
- Full automatic PyTorch-to-ONNX export remains pending for arbitrary repositories.

## v6: Production SPN Deployment Loop

- Treat the board as the final correctness source, but catch quantization failures earlier in software.
- Preserve original model semantics with exact runtime rewrites whenever possible.
- Move approximate changes into the compiler-aligned training graph when exact equivalence is impossible.
- Search for maximal RHB subgraphs to reduce Host/RHB launch overhead.
- Record per-boundary scale contracts, outlier risk, saturation, effective board scale, and validation metrics.

Status:

- CompletionFormer is the reference production case.
- CSPN and NLSPN have deployable Host/RHB splits and app-level demos, but still need more automated retrain/export recipes and more aggressive max-subgraph search to reduce launch count.
- New `production-plan` and `quant-diagnostics` commands provide the shared gate structure for the next refinement pass.
