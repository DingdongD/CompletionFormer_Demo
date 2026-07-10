# RHB Board Depth Viewer

Small zero-dependency web UI for the successful RHB board depth pipelines.

Supported model views:

- `CompletionFormer HW128`: ckpt00059 conv-only confidence-head board pipeline.
- `CSPN ResNetTiny HW128`: stagewise scale-aware board pipeline with padded16 final depth head.

## Run

```bash
cd /root/demo/artifacts/successful_pipelines/completionformer_hw128_ckpt00059_rhb_convonlycf_20260702/portable_runtime
python apps/completionformer_board_viewer/app.py --host 0.0.0.0 --port 7861
```

Open:

```text
http://127.0.0.1:7861
```

The app reads the portable runtime from:

```text
/root/demo/artifacts/successful_pipelines/completionformer_hw128_ckpt00059_rhb_convonlycf_20260702/portable_runtime
```

Override with:

```bash
CF_PORTABLE_DIR=/path/to/portable_runtime python apps/completionformer_board_viewer/app.py
```

The CSPN package defaults to:

```text
/root/demo/artifacts/rhb_auto_config_framework/work/deployment_packages/cspn_resnettiny_hw128_w24_step8_stagewise_v3_hostsample
```

Override with:

```bash
CSPN_PACKAGE_DIR=/path/to/cspn_package \
CSPN_VAL32_INPUT_DIR=/path/to/cspn_inputs \
CSPN_VAL32_BOARD_DIR=/path/to/cspn_board_outputs \
python apps/completionformer_board_viewer/app.py
```

CSPN inputs are generated from the same CompletionFormer NYU source NPZ:

```text
data/nyu_val32_source_128x128.npz
```

The generated CSPN RGBD inputs and board outputs live under:

```text
outputs/cspn_unified_input/inputs
outputs/cspn_unified_input/board_outputs
```

## Board Run

The `Run Board` button calls:

```bash
BOARD=root@192.168.115.122 BOARD_PASS=root ./run_board_single_sample.sh <sample_index>
```

The UI then loads:

```text
outputs/sample<idx>/nyu_val<idx>_ref_vs_board_convonlycf_hostsigmoid.npz
```

For CSPN, the app uses the unified input source and executes one of the 32 packaged NYU samples:

```bash
python3 cspn_resnettiny_hw128_w24_step8_board_runner_stagewise_v3_scaleaware.py \
  <board_package> \
  --input-npz app_inputs/cspn_valXX_input.npz \
  --save app_outputs/cspn_valXX_board_padded16_clearwr_run1.npz \
  --scales-csv cspn_stagewise_scales_orig_val0_20260709.csv \
  --unit-scales \
  --use-scaled-simple \
  --use-scaled-fullsplit \
  --use-padded-depth-head
```

## Interfaces

- NYU packaged samples: implemented.
- RGBD upload: endpoint reserved; current board runner expects packaged 128x128 NYU NPZ samples.
- ToF: endpoint reserved for later live-device integration.
