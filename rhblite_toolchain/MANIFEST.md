# RHBLite Toolchain Manifest

This directory contains source code and documentation needed to reproduce the
CompletionFormer/CSPN RHBLite flow from PyTorch submodules to board execution.

It intentionally excludes generated ONNX files, compiled outputs, packer
binaries, checkpoints, board logs, and validation datasets.

## Scripts

- `scripts/cv_onnx.py`: PyTorch submodel to ONNX export, simplify, quantize.
- `scripts/compile.py`: ACompiler entrypoint.
- `scripts/test_rt.py`: ONNXRuntime vs ACSim/cmodel comparison.
- `scripts/run_compile_sim.sh`: one-command export, compile, cmodel, optional ACTSim.
- `scripts/pack_and_board.sh`: Model-Packer and simple board deploy.
- `scripts/export_completionformer_hw128_real_features.py`: export real ckpt features for staged board execution.
- `scripts/rhb_completionformer_decoder_system_runner_128x128_ckpt.py`: multi-submodel board runner using `ac_driver`.
- `scripts/visualize_completionformer_board_real_pred.py`: compare and visualize reference vs board outputs.
- `scripts/recalibrate_completionformer_ckpt_onnx_scales*.py`: activation scale calibration helpers.
- `scripts/compare_cspn_*_golden.py`: CSPN equivalence checks.

## Models

- `models/completionformer_test/`: accepted ckpt-backed CompletionFormer 128x128 RHB submodels.
- `models/cspn_test/`: compiler-aligned CSPN backbone and propagation stages.

## Docs

- `README.md`: full pipeline analysis and usage.
- `docs/compiler_actsim_rules.md`: detailed operator adaptation notes.
- `docs/*completionformer*.md`: board validation and error attribution reports.

## Configs

- `configs/activation_scales.csv`: accepted board runner input/output scales.
- `configs/config.yaml`: accepted pipeline configuration snapshot.
