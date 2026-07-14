# Runtime Outlier Root Cause and Mitigation

Date: 2026-07-14

## Root Cause Evidence

The CompletionFormer rerun with `dec4_rhb = 1019.741 ms` was not slow because the RHB NPU computation took 1 second.

Relevant log interval:

```text
10:30:19.949717 Running inference on dec4
10:30:20.963260 Sending data to accelerator with 28704 bytes
10:30:20.963654 Checked wr_done after 244 loops
10:30:20.964332 sta_npu_opt_cnt = 25290
```

The delay happened before the input DMA send started. Once the data was sent, `wr_done`, output receive, and NPU counters were normal.

Normal dec4 runs show:

```text
Running inference -> Sending data: about 1-4 ms
dec4_rhb total: about 7-10 ms
```

The outlier run showed:

```text
Running inference -> Sending data: about 1013 ms
dec4_rhb total: 1019.741 ms
```

Therefore the most likely source is Host/runtime jitter inside `ac_driver.run_inference()` before DMA submission:

- Linux scheduling interruption of the Python/ac_driver process;
- CPU frequency/userspace governor state;
- page fault or memory residency jitter before the C++ driver touches the NumPy buffer;
- temporary board CPU contention;
- less likely: RHB instruction/NPU execution, because `sta_npu_opt_cnt` and `wr_done` were normal.

## Tested Mitigations

### Full stabilization

Command options:

```text
--stabilize-runtime
```

This enables:

- CPU max frequency
- CPU3 affinity
- SCHED_FIFO priority 20
- `mlockall`

Result:

- `dec4_rhb = 10.246 ms`, so the 1s spike disappeared.
- total latency increased to `2028.318 ms`.

Conclusion: pinning to CPU3 and SCHED_FIFO reduced the observed dec4 spike, but hurt full-resolution transfer/head throughput. Do not use this as the default mode.

### Preferred mitigation

Command options:

```text
--lock-cpu-max-freq --mlockall
```

Result:

- `dec4_rhb = 7.570 ms`
- total latency `1492.605 ms`
- no 1s pre-DMA stall
- latency is close to the normal steady-state range.

Conclusion: the default runtime mitigation should be CPU max frequency plus `mlockall`, without CPU affinity or SCHED_FIFO.

## Code Updates

Updated runner:

- `artifacts/successful_pipelines/completionformer_hw128_ckpt00059_rhb_convonlycf_20260702/portable_runtime/scripts/rhb_completionformer_decoder_system_runner_128x128_ckpt.py`
- `artifacts/successful_pipelines/completionformer_hw128_ckpt00059_rhb_convonlycf_20260702/portable_runtime/rhblite_toolchain/scripts/rhb_completionformer_decoder_system_runner_128x128_ckpt.py`

Added CLI controls:

- `--lock-cpu-max-freq`
- `--mlockall`
- `--cpu-affinity`
- `--realtime-priority`
- `--stabilize-runtime`

Updated default board script:

- `artifacts/successful_pipelines/completionformer_hw128_ckpt00059_rhb_convonlycf_20260702/portable_runtime/run_board_single_sample.sh`

It now uses:

```text
--clear-wr-done-before-run --lock-cpu-max-freq --mlockall
```

## Practical Rule

For future board runners:

1. Always clear `wr_done` before each submodel.
2. Set CPU frequency to max before benchmark/inference.
3. Use `mlockall` if running as root.
4. Avoid CPU affinity/SCHED_FIFO by default; only use them as a debug mode.
5. If a spike remains, inspect driver log timing:
   - delay before `Sending data`: Host/runtime scheduling or memory residency;
   - delay between `Sending data` and `Checked wr_done`: DMA/NPU/status path;
   - high `wr_done` loop count or timeout: PL/driver status handling.
