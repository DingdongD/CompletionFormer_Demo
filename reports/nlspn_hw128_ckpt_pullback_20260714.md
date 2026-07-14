# NLSPN HW128 Checkpoint Pullback

Pulled from remote training host on 2026-07-14.

Remote source:

```text
root@100.104.115.57:/workspace/CSPN/cspn_pytorch/output/nyu_nlspn_hw_aligned_128_20260713/model_best.pt
```

The original training checkpoint is 184.7 MiB and includes optimizer/scheduler state. An inference-only checkpoint was generated on the remote host with `state_dict`, `metrics`, `args`, and `epoch` only, then copied locally.

Local files:

```text
/root/demo/artifacts/nlspn_hw_aligned_ckpts/nyu_nlspn_hw_aligned_128_20260713/model_best_infer_state.pt
/root/demo/artifacts/nlspn_hw_aligned_ckpts/nyu_nlspn_hw_aligned_128_20260713/metadata.json
/root/demo/artifacts/nlspn_hw_aligned_ckpts/nyu_nlspn_hw_aligned_128_20260713/log_train.txt
```

SHA256:

```text
4f2c6e5d5312e2c13498cf9425be51e9e3c05c25bd593553bc06890401c88be2  model_best_infer_state.pt
2c40c6001132959a28d730fb6cf7822114321faa1ae5386abaa8c884350bc218  metadata.json
35d1aa666028714cf89db12a6852bb2f348a0c2a20235a41e9577d21e66929c8  log_train.txt
```

Strict load result against `models.nlspn_test.nlspn_hw_aligned.NLSPNHWAlignedModel`:

```text
keys: 172
missing: []
unexpected: []
```

Training best metrics embedded in checkpoint:

```json
{"val_l1": 0.10124755278229713, "val_pred_init_l1": 0.10945911658927798, "val_rmse": 0.26286692218855023, "epoch": 50, "train_loss": 0.1066361660913132, "lr": 0.00015, "elapsed_sec": 45.80579972267151, "best_l1": 0.10124755278229713}
```

CPU latency report updated in:

```text
reports/cpu_vs_rhb_latency_20260714.json
```
