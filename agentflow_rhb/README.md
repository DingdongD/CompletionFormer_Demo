# RHB Black-box Auto Configuration Framework

This directory turns the current CompletionFormer deployment experience into a reusable workflow for future RHB deployments.

## Portable Snapshot In This Repository

This `agentflow_rhb/` directory is a source-only snapshot of the local
`/root/demo/artifacts/rhb_auto_config_framework` workspace. It intentionally
excludes generated work directories, packer binaries, board outputs, reports,
and checkpoints.

From this repository root, use:

```bash
python agentflow_rhb/rhb_auto_config/cli.py summarize-rules
python agentflow_rhb/rhb_auto_config/cli.py production-plan \
  --model cspn_hw128 \
  --case agentflow_rhb/examples/cspn_hw128_current_success_case.json
python agentflow_rhb/rhb_auto_config/cli.py plan \
  --case agentflow_rhb/examples/dyspn_hw128_probe.json
python agentflow_rhb/rhb_auto_config/cli.py remote-train \
  --profile nlspn_eccv20_hw128_resnet18 \
  --action plan
python agentflow_rhb/rhb_auto_config/cli.py remote-train \
  --profile dyspn_hw128_resnet18 \
  --action plan
```

The original `/root/demo/artifacts/rhb_auto_config_framework/...` commands
below are preserved for the lab machine where the full generated workspace
exists.

Current deployment reports:

- `docs/subgraph_load_reduction_status.md`: CompletionFormer/CSPN/NLSPN bundle partition status, accepted packers, and rejected candidates.
- `docs/inference_only_latency_breakdown.md`: CPU vs Host/RHB latency context, load timing, packer switch first-run timing, and steady inference bottlenecks.
- `docs/nlspn_fullres_launch_reduction.md`: NLSPN 128x128 full-resolution fusion probes, strict all-RHB head default, and optional Host-post latency experiment.
- `docs/dyspn_hw128_adaptation.md`: DySPN HW128 compiler-aligned scaffold, NYU Depth V2 adapter, and first RHB board probe.

DySPN status:

- `dyspn_test.dyspn_hw_offset_aff_compile_probe` passes compile/CModel/packer/board with `All same: True`.
- Full `dyspn_hw_guide` exports to ONNX but still needs Host resize/concat/add + RHB Conv partitioning because the current compiler path rejects nearest `Resize`.
- A trained HW-aligned checkpoint is required before end-to-end board validation because ConvTranspose-style decoder behavior is replaced by Host resize + Conv compiler-aligned glue.

NLSPN strict semantic validation entrypoint on the lab machine:

```bash
LIMIT=32 artifacts/rhb_auto_config_framework/scripts/run_nlspn_val32_strict_board.sh
```

The framework is designed for a black-box accelerator flow:

- compiler/runtime cannot be modified
- supported operators/layouts are discovered by probing
- CModel pass is necessary but not sufficient
- board output is the final source of truth
- submodels must obey the RHB 8MB packed-weight budget
- unsupported or risky fragments are scheduled on Host

## What This Contains

```text
rhb_auto_config_framework/
  configs/                 Hardware and flow defaults.
  docs/                    Design, rulebook, playbook, lessons.
  examples/                CompletionFormer case and known schedule.
  reports/                 Generated reports.
  rhb_auto_config/         Python CLI, ONNX optimizer, runners, and deployment planner.
  rule_db/                 Seed black-box RHB rules.
  scripts/                 Convenience entrypoints.
  templates/               New-model case and rule-update templates.
```

## Quick Start

From `/root/demo`:

```bash
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py summarize-rules
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py scan-models --models-root /root/demo/models/completionformer_test
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py plan --case artifacts/rhb_auto_config_framework/examples/completionformer_hw128_case.json
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py plan --case artifacts/rhb_auto_config_framework/examples/cspn_hw128_current_success_case.json
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py plan --case artifacts/rhb_auto_config_framework/examples/nlspn_eccv20_compiler_aligned_case.json
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py plan --case artifacts/rhb_auto_config_framework/examples/dyspn_hw128_probe.json
```

Closed-loop adapters:

```bash
# ONNX graph import and op histogram
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py import-onnx --onnx onnx_models/<model>.onnx

# Initial per-node RHB/Host annotation
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py annotate-onnx --onnx onnx_models/<model>.onnx

# Production closed-loop plan for a model family/case
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py production-plan \
  --model cspn_hw128 \
  --case artifacts/rhb_auto_config_framework/examples/cspn_hw128_current_success_case.json

# Render SSH-based remote training commands for an aligned model.
# Use submit/status/fetch after reviewing the generated plan.
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py remote-train \
  --profile cspn_resnettiny_hw128_sample1x1 \
  --action plan

python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py remote-train \
  --profile nlspn_eccv20_hw128_resnet18 \
  --action plan

python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py remote-train \
  --profile dyspn_hw128_resnet18 \
  --action plan

# After remote-train --action fetch, record checkpoint/export/packer hashes before validation.
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py artifact-manifest \
  --name nlspn_hw128_after_fetch \
  --paths artifacts/rhb_auto_config_framework/remote_ckpts/nlspn_eccv20_hw128_resnet18 \
          onnx_models/<fresh_export>.onnx \
          artifacts/rhb_auto_config_framework/work/<fresh_packer_dir>

# ACSim-style tile/padding/layout risk analysis
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py analyze-layout \
  --onnx onnx_models/<model>.onnx

# Full graph optimization pass:
# import -> annotate -> layout analysis -> maximal RHB regions -> Host/RHB deployment graph
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py optimize-onnx \
  --onnx onnx_models/<model>.onnx \
  --export-submodels

# Enumerate deeper Host/RHB search candidates and rewrite recipes
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py deep-search \
  --onnx onnx_models/<model>.onnx

# The default policy is latency-first and boundary-minimizing:
# configs/deployment_policy_latency_first.json

# Generate a deployment package from the selected deep-search candidate
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py generate-package \
  --deep-search-json artifacts/rhb_auto_config_framework/reports/deep_search_<model>.json

# Generate an explicit package contract for an existing package
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py generate-contract \
  --package-dir artifacts/rhb_auto_config_framework/work/deployment_packages/<package>

# Score many existing ONNX candidates and rank larger/cleaner RHB regions first
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py score-onnx-dir \
  --onnx-root onnx_models \
  --glob 'completionformer_test.*128*.onnx'

# Local export/compile/CModel through the existing Makefile
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py compile-cmodel --model completionformer_test.<submodel> --layout "input0=BWC"

# Model-Packer wrapper
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py pack \
  --compile-output-dir /path/to/output/<model> \
  --packer-output-dir /path/to/packer_<model>

# Board runner wrapper
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py board-run \
  --packer-dir /path/to/packer_<model>

# One-shot local compile+cmodel+pack, with optional board run
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py deploy-loop \
  --model completionformer_test.<submodel> \
  --layout "input0=BWC" \
  --run-board

# Batch validate model modules through compile/cmodel/pack/board
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py validate-models \
  --name cspn_board_smoke_bchw \
  --models cspn_test.cspn_tiny_aligned_backbone_ch4 \
  --layout "input0=BCHW" \
  --skip-cv-model \
  --run-board

# Convert a compile/board/deploy result JSON into a rule-update draft
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py suggest-rule \
  --result-json artifacts/rhb_auto_config_framework/work/reports/deploy_loop_<model>.json

# Classify a failed board/compile result and generate retry actions
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py localize-failure \
  --result-json artifacts/rhb_auto_config_framework/work/reports/<result>.json
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py plan-retry \
  --localization-json artifacts/rhb_auto_config_framework/reports/failure_localization_<result>.json

# Execute supported retry actions, currently including multi-output region splitting
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py execute-retry \
  --retry-plan-json artifacts/rhb_auto_config_framework/reports/retry_plan_<result>.json \
  --region-plan-json artifacts/rhb_auto_config_framework/reports/region_plan_<model>.json

# Profile a new PyTorch source tree before ONNX export
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py profile-source \
  --source-root /root/demo/CSPN/cspn_pytorch

python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py profile-source \
  --source-root /root/demo/NLSPN_ECCV20/src

# Generate a new model case spec
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py new-case \
  --case-name cspn_nyu_probe \
  --model-family CSPN \
  --source-root /root/demo/CSPN/cspn_pytorch \
  --input-shape 1,3,128,128

# Collect measured board latency/counters from result JSONs
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py calibrate-costs \
  --result-root artifacts/rhb_auto_config_framework/work/reports

# Summarize compile/cmodel/pack/board health across generated result JSONs
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py health-report

# Convert health failures into prioritized rewrite/retry actions
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py rewrite-backlog

# Software-side RHB int8 quantization, outlier, saturation, and kurtosis diagnostics.
# Use this on calibration feature dumps before deciding scale/split/QAT strategy.
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py quant-diagnostics \
  --npz artifacts/rhb_auto_config_framework/work/<case>/calibration_features.npz \
  --channel-axis 1

# Execute supported backlog actions: exact Conv IC/OC split export, or multi-output split retry
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py execute-backlog \
  --category conv_ic_oc_split \
  --validate \
  --limit 1

python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py execute-backlog \
  --category split_multi_output \
  --limit 1

# Generate an exact Host/RHB Conv split contract with post-sum affine/ReLU.
# This captures the deployment glue needed after partial Conv export.
python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py make-split-contract \
  --name nlspn_dec2_bn_relu_ic64 \
  --source models/nlspn_test/nlspn_hw_aligned.py::dec2 \
  --in-channels 192 \
  --out-channels 64 \
  --height 32 \
  --width 32 \
  --input-chunk-channels 64 \
  --post bn_relu
```

The generated reports are written to:

```text
artifacts/rhb_auto_config_framework/reports/
```

## Current Maturity

This is now a graph-aware deployer and candidate optimizer, not a fully automatic compiler replacement.

The current default rule layer has been reset to the board-proven
CompletionFormer HW128 and CSPN HW128 app demos. Superseded explorations
including fakequant/QAT004, im2col stride-conv replacement, stale-output
run-repeat, fused-head defaulting, and runner-only scale compensation are
archived under:

```text
archive/superseded_experiments_20260710/
```

Use `docs/current_success_framework.md` as the first reference before adding or
promoting rules for a new model.

Implemented now:

- rule schema and seed RHB capability database
- model experiment inventory scanner
- initial rule-driven planner
- ONNX graph import with shape inference and packed-weight estimates
- rule-driven node annotation
- ACSim-inspired layout/tile/padding risk analysis
- maximal contiguous RHB region builder
- effective 8MB budget splitter
- Host/RHB deployment graph emitter
- rewrite and host-fallback suggestion engine
- RHB region ONNX submodel exporter, with a manual fallback for compiler-extended ONNX attributes
- directory-level candidate scorer for existing submodel portfolios
- source-tree profiler for new PyTorch models
- new-case generator for onboarding CSPN/Depth-Anything-like models
- failure localizer and retry planner
- retry executor for supported automatic actions, currently multi-output RHB region splitting
- deep-search strategy enumerator for exact, split-output, and exploratory rewrite-heavy plans
- deployment package generator with schedule, RHB ONNX exports, Host kernels, calibration plan, retraining plan, and explicit package contract
- measured result collector for cost calibration
- CompletionFormer-derived deployment playbook
- failure taxonomy and mitigation templates
- compile/CModel, Model-Packer, board-run, deploy-loop wrappers
- result-to-rule feedback draft generator
- framework health reporter for pass/fail/risky submodel summaries
- rewrite backlog generator for repeated timeout, multi-output, and Conv split failures
- backlog executor for supported automatic rewrite actions: Conv IC/OC split export plus optional validation, and multi-output region split retry execution
- exact Conv split contract generator for Host partial-sum plus RHB post-sum affine/ReLU schedules
- production-plan generator that standardizes exact rewrite, approximate retraining, compile/cmodel/board, real-feature trace, and visualization gates
- software-side RHB int8 quant diagnostics for calibration/boundary NPZs, including outlier energy, kurtosis, saturation, and QDQ error
- SSH remote training profile generator/submitter/fetcher for CSPN/NLSPN compiler-aligned retraining
- artifact manifest generator to bind checkpoint, fresh ONNX export, compile output, and packer directory before board accuracy analysis

Expected next integrations:

- PyTorch-source submodel generation for models that cannot compile directly from extracted ONNX.
- automatic end-to-end execution of split contracts after failure localization.
- learned/benchmarked cost model for CPU/RHB boundary and launch overhead.
- hardware-aware retraining recipe generation for approximate rewrites.

## Main Documents

- `docs/framework_design.md`: framework architecture and decision lattice.
- `docs/current_success_framework.md`: current accepted rules distilled from the CompletionFormer and CSPN board demos.
- `docs/completionformer_lessons.md`: rules distilled from the CompletionFormer deployment.
- `docs/nlspn_eccv20_adaptation.md`: first-pass NLSPN ECCV20 Host/RHB allocation and unsupported-op boundaries.
- `docs/deployment_playbook.md`: step-by-step workflow for a new model.
- `docs/failure_taxonomy.md`: compile/CModel/board/accuracy failure classes.
- `docs/integration_contracts.md`: JSON contracts for future graph, compiler, and board adapters.
- `docs/roadmap.md`: staged implementation plan.

## Templates

- `templates/new_model_case.template.json`
- `templates/rule_update.template.json`

## Key CompletionFormer Rule Outcome

Accepted schedule:

```text
RHB:
  decoder dec6-dec3
  dec2 split resize/up-conv + exact block convs
  dep_dec1, dep_dec0
  gd_dec1, gd_dec0
  cf_dec1
  cf_dec0 Conv only

Host:
  concat / split-sum / residual / quant glue
  cf_dec0 sigmoid
  NLSPN
```

Runtime:

```text
--clear-wr-done-before-run
```

Avoid:

```text
rram_only=true
hardware Sigmoid for cf_dec0
fused dep/gd/cf dec1 for ckpt00059
```

## Key CSPN Rule Outcome

Accepted case:

```text
examples/cspn_hw128_current_success_case.json
```

Accepted package:

```text
work/deployment_packages/cspn_resnettiny_hw128_w24_step8_stagewise_v3_hostsample
```

Accepted app input/output source:

```text
../successful_pipelines/completionformer_hw128_ckpt00059_rhb_convonlycf_20260702/portable_runtime/outputs/cspn_unified_input
```

Use:

```text
layout=input0=BCHW
Host sample/downsample before stride2 replacements
RHB Conv submodels with explicit scale-aware Host glue
padded16 final depth head only as a validated candidate/default for the current CSPN app
```

Avoid:

```text
internal Slice/Gather downsample inside RHB
multi-output RHB submodels
fakequant/QAT004 as deployment default
im2col as deployment default
runner-only scale on qscale=1 packers
run-repeat=2 as a correctness fix
```
