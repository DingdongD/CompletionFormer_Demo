# DySPN HW128 Compiler-aligned Adaptation, 2026-07-14

## Scope

This pass adds the initial DySPN compiler-aligned inference path and NYU Depth V2 128x128 data adapter.

The implementation keeps strict deployment semantics:

- Conv/BN/ReLU and Conv heads are RHB candidates.
- Resize is Host glue before RHB Conv, following the CompletionFormer/NLSPN rules.
- DySPN `grid_sample` propagation remains Host-side.
- No manual runtime affine compensation is used.

## Added Files

Model modules:

```text
/root/demo/models/dyspn_test/dyspn_hw_aligned.py
/root/demo/models/dyspn_test/dyspn_hw_guide.py
/root/demo/models/dyspn_test/dyspn_hw_offset_aff.py
/root/demo/models/dyspn_test/dyspn_hw_offset_aff_compile_probe.py
```

Data / inference scripts:

```text
/root/demo/artifacts/rhb_auto_config_framework/scripts/prepare_dyspn_nyu128_npz.py
/root/demo/artifacts/rhb_auto_config_framework/scripts/run_dyspn_hw_nyu128_inference.py
```

Agentflow case:

```text
/root/demo/artifacts/rhb_auto_config_framework/examples/dyspn_hw128_probe.json
```

## NYU Depth V2 Adapter

Generated DySPN-specific 128x128 val32 inputs:

```text
/root/demo/artifacts/rhb_auto_config_framework/work/dyspn_nyu_val32_raw_128x128.npz
/root/demo/artifacts/rhb_auto_config_framework/work/dyspn_nyu_val32_imagenet_128x128.npz
```

DySPN reference code comments out ImageNet normalization, so the raw `[0,1]` RGB npz is the default DySPN input contract.

Smoke inference output:

```text
/root/demo/artifacts/visualizations/dyspn_hw_aligned_nyu128_raw/
```

Current metrics are from an untrained/random HW-aligned model and are only a data-path smoke test:

```text
sample,l1,rmse,max_abs,pred_min,pred_max
0,3.457499,3.690502,5.755812,0.001937,3.876190
1,2.770655,3.108776,6.020856,0.003791,4.202377
2,2.146235,2.292078,3.622453,0.003450,2.486122
3,1.274602,1.326955,2.027985,0.002498,1.446195
```

## Board Probe

The DySPN offset/aff Conv shape was validated through compile/CModel/packer/board using a no-checkpoint nonzero compile probe:

```text
model: dyspn_test.dyspn_hw_offset_aff_compile_probe
compile_status: pass
cmodel_status: pass_with_warnings
pack_returncode: 0
pack_has_config: True
board_returncode: 0
board_status: pass
board_all_same: True
```

Why a compile probe exists:

The strict untrained DySPN initialization sets `conv_offset_aff` weights and bias to zero. The legacy quantizer divides by activation range and fails on all-zero output. A real trained `DYSPN_HW_CKPT` should use the strict `dyspn_hw_offset_aff.py` module, not the probe.

## Full Guide Graph Status

The full RGBD -> guide ONNX export succeeds:

```text
/root/demo/onnx_models/dyspn_test.dyspn_hw_guide.onnx
```

But compiling the full guide graph as one RHB submodel is not currently accepted:

```text
model: dyspn_test.dyspn_hw_guide
compile_status: pass_with_warnings
cmodel_status: fail
```

Compiler log root cause:

```text
Resize mode not supported: nearest
```

This is expected. The accepted framework rule is to keep resize in Host and send the following Conv/BN/ReLU to RHB. The agentflow region plan therefore splits the guide graph into RHB Conv regions with Host concat/resize/add glue.

Generated plans:

```text
/root/demo/artifacts/rhb_auto_config_framework/reports/source_profile_DySPN.md
/root/demo/artifacts/rhb_auto_config_framework/reports/deployment_graph_dyspn_test.dyspn_hw_guide.md
/root/demo/artifacts/rhb_auto_config_framework/reports/region_plan_dyspn_test.dyspn_hw_guide.md
```

## Current Workload Split

Initial DySPN split:

```text
RHB:
  - Conv/BN/ReLU subgraphs in RGB/depth stem, ResNet blocks, decoder convs, guidance heads
  - DySPN conv_offset_aff guide -> offset/aff Conv

Host:
  - resize before decoder convs
  - concat / residual add glue unless included in a board-validated single-output submodel
  - sigmoid / softmax / slice / narrow
  - DySPN grid_sample propagation loop
```

## Next Step

To complete board-grade DySPN deployment:

1. Train or fetch a `DYSPN_HW_CKPT` for this compiler-aligned architecture.
2. Generate stage/decoder/head split modules from the `deployment_graph_dyspn_test.dyspn_hw_guide` plan.
3. Validate each Conv region with compile/CModel/board, using strict all-RHB Conv post behavior.
4. Build a DySPN board runner equivalent to the NLSPN runner:
   Host resize/concat/add -> RHB Conv bundle -> Host DySPN propagation.
5. Run val32 NYU visualizations from `dyspn_nyu_val32_raw_128x128.npz`.
