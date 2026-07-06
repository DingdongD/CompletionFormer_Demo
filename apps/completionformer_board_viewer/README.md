# CompletionFormer Board Viewer

Small zero-dependency web UI for the successful CompletionFormer HW128 ckpt00059 board pipeline.

## Run

```bash
cd /root/demo/apps/completionformer_board_viewer
python app.py --host 0.0.0.0 --port 7861
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
CF_PORTABLE_DIR=/path/to/portable_runtime python app.py
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

## Interfaces

- NYU packaged samples: implemented.
- RGBD upload: endpoint reserved; current board runner expects packaged 128x128 NYU NPZ samples.
- ToF: endpoint reserved for later live-device integration.
