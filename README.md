# Portable Runtime: CompletionFormer HW128 ckpt00059

This is the minimal movable runtime package for the accepted RHBLite CompletionFormer 128x128 decoder/head pipeline.

It includes:

- compiled RHBLite packer
- ckpt00059 and minimal `ref_model_hw`
- 32-sample source npz
- accepted submodel source files
- board runner, feature exporter, visualization script
- operator optimization rules
- known validation metrics

It does not include the compiler, Model-Packer, or the full historical experiment tree.

## Quick Test

From inside this directory on the host:

```bash
./run_board_single_sample.sh 0
```

Override board settings if needed:

```bash
BOARD=root@192.168.115.122 BOARD_PASS=root ./run_board_single_sample.sh 0
```

Run all 32 representative samples:

```bash
./run_board_val32.sh
```

## Expected Output

For sample0, the script writes:

```text
outputs/sample0/
  nyu_val0_features_ckpt00059.npz
  board_val0_convonlycf_hostsigmoid.log
  board_val0_convonlycf_hostsigmoid_outputs.npz
  nyu_val0_ref_vs_board_convonlycf_hostsigmoid.png
  nyu_val0_ref_vs_board_convonlycf_hostsigmoid.npz
```

The accepted 32-sample reference metrics are stored under `validation/`.

## Important Flags

The board run must use:

```text
--cf-dec0-host-sigmoid
--clear-wr-done-before-run
```

The first flag keeps sigmoid on Host. The second flag clears stale `wr_done` before each submodel launch.

## Operator Rules

See:

`docs/operator_optimization_rules.md`
# CompletionFormer_Demo
