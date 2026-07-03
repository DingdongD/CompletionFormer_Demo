# File Layout

```text
portable_runtime/
  CompletionFormer/ref_model_hw/       Minimal ref_model_hw implementation and ckpt00059.
  data/                                32-sample preprocessed source npz.
  docs/                                Rules and migration notes.
  models/completionformer_test/        Accepted submodel source definitions.
  packer/                              Compiled RHBLite packer.
  scripts/                             Host export/visualization and board runner scripts.
  validation/                          Known 32-sample metrics and visual summary.
```

The package is intended for runtime validation on another host plus the RHBLite board. It is not a full compiler environment.

Required host Python packages:

- `torch`
- `numpy`
- `matplotlib`
- `sshpass` command-line tool for board transfer

Required board files/environment:

- `/home/root/workspace/demo_vp_xj/ac_driver.py`
- `/dev/mem` access
- RHBLite runtime dependencies already installed on the board
