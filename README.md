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

## AgentFlow RHB Framework

The reusable Host/RHB deployment framework is included under:

```text
agentflow_rhb/
```

It packages the rule database and automation used to generalize the
CompletionFormer deployment flow to CSPN and NLSPN:

- ONNX graph import, annotation, layout risk analysis, and deep-search planning
- exact compiler-aligned rewrite contracts such as IC/OC split and pad/slice
- approximate rewrite policy that requires compiler-aligned retraining
- software-side int8 outlier, saturation, kurtosis, and boundary-scale diagnostics
- SSH remote training templates for CSPN/NLSPN aligned checkpoints
- compile/CModel/Model-Packer/board-run wrappers for the lab environment

Quick smoke commands:

```bash
python agentflow_rhb/rhb_auto_config/cli.py summarize-rules
python agentflow_rhb/rhb_auto_config/cli.py production-plan \
  --model nlspn_eccv20_hw128 \
  --case agentflow_rhb/examples/nlspn_eccv20_compiler_aligned_case.json
python agentflow_rhb/rhb_auto_config/cli.py remote-train \
  --profile cspn_resnettiny_hw128_sample1x1 \
  --action plan
```

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

## Board Output Demo

The image below stitches 32 validation samples. Each group shows the reference prediction, the RHBLite board prediction, and the absolute error.

![32-sample board/ref/error demo](validation/nyu_val32_board_ref_error_triplet_contact_sheet.png)

Summary metrics from `validation/val32_metrics.csv`:

- final pred abs mean: `0.022418`
- final pred p95 mean: `0.096996`
- pred_init abs mean: `0.062948`
- confidence abs mean: `0.042193`
- latency mean: `1552.956 ms`
- latency mean excluding one observed scheduling outlier: `1488.789 ms`

## CompletionFormer / CSPN / NLSPN Comparison

The GitHub Pages demo and local viewer now expose the same comparison flow:

- RGB, sparse depth, GT, reference prediction, board prediction, and absolute error maps
- full-resolution point cloud with RGB, oblique, top, and BEV views
- per-sample fine-grained top-operator latency pie and top-op table when the runner packaged `LATENCY` markers
- model-level summary cards computed from the saved board outputs

Current published sample summary from `docs/data/manifest.json`:

| Model | Saved board samples | Mean board/ref L1 | Mean board/ref RMSE | Latency | Main tracked bottleneck |
| --- | ---: | ---: | ---: | --- | --- |
| CompletionFormer HW128 | 15 | 0.0222 | 0.0550 | 1515.6 ms median / 1975.9 ms mean | RHB decoder/head compute 92.4%, Host glue 7.6% |
| CSPN ResNetTiny HW128 | 32 | 0.1089 | 0.1479 | 16684.1 ms median / mean | Submodel load/switch 93.5%, RHB compute 4.7% |
| NLSPN HW128 | 32 | 0.0779 | 0.1456 | 8817.4 ms median / 8750.8 ms mean | Split decoder/head launches; top dec5/dec4/dec3 launches include switch-mixed overhead |

The bottleneck interpretation is:

- CompletionFormer is numerically the strongest of the three packaged demos. Runtime is dominated by full-resolution decoder/head RHB conv blocks; the remaining Host cost comes from resize, split/sum glue, and boundary requantization.
- CSPN is functionally packaged, but its current latency is dominated by repeated packer load/submodel switch overhead rather than arithmetic. The production direction is fewer RHB launches, persistent package reuse, and larger compiler-aligned fused blocks.
- NLSPN now packages per-sample `LATENCY` markers from the split-decoder/head runner. The current log granularity mixes model switch/load overhead into the first `RUN` after each switch, so the fine-grained pie identifies expensive launches, while a lower-level runtime trace is still needed to split pure arithmetic from packer-switch cost.

### CPU vs Host/RHB Latency

CPU baselines were measured on this host with PyTorch eval/no-grad, batch=1,
`128x128` NYU val inputs, 4 CPU threads, 2 warmup samples, and 5 measured
samples. The Host/RHB numbers are from the packaged board traces in
`docs/data/manifest.json`.

| Model | CPU compiler-aligned PyTorch median | Current Host/RHB board median | Observation |
| --- | ---: | ---: | --- |
| CompletionFormer HW128 | 35.4 ms | 1515.6 ms | Board path is dominated by full-res decoder/head launches and data/glue overhead. |
| CSPN ResNetTiny HW128 | 15.1 ms | 16684.1 ms | Current board path is dominated by repeated submodel load/switch, not arithmetic. |
| NLSPN HW128 | 77.7 ms topology-only | 8817.4 ms | Full trained NLSPN ckpt was not present locally; topology CPU timing uses the same compiler-aligned model with random weights. Board path is split-launch heavy. |

The CPU numbers are not an accuracy comparison; they are a host-side latency
baseline for the compiler-aligned model structures. The current RHB deployments
are useful for validating board-compatible scheduling, quantization contracts,
and visual output, but their latency is dominated by launch granularity and
runtime/model-switch overhead. The main optimization target is therefore larger
RHB subgraphs, persistent model loading, and fewer Host/RHB round trips.

## GitHub Pages Demo

A static GitHub Pages demo is available under `docs/`. It uses saved RHBLite board outputs and does not require Python, SSH, or board access.

Local preview:

```bash
python -m http.server 8091 --directory docs
```

Open:

```text
http://127.0.0.1:8091
```

The static demo includes the current CompletionFormer, CSPN, and NLSPN saved board-output samples, depth/error maps, latency metrics, fine-grained top-operator latency pie charts, and full `128x128 = 16384` point point-cloud data per sample. To publish it on GitHub, enable GitHub Pages from the repository `docs/` directory on this branch or after merging to `main`.

## Web Viewer

A local board-output viewer is included under `apps/completionformer_board_viewer/`. It can load saved board outputs, trigger `run_board_single_sample.sh`, show depth/error maps, report latency parsed from board logs, and render a point-cloud view.

Run from this portable runtime directory:

```bash
python apps/completionformer_board_viewer/app.py --host 0.0.0.0 --port 7861
```

Open:

```text
http://127.0.0.1:7861
```

The app keeps RGBD upload and ToF source endpoints reserved for later live input integration.


## Full Toolchain Source

The source-level compile and simulation workflow is preserved under:

```text
rhblite_toolchain/
```

That directory documents and includes the code for:

```text
PyTorch submodel -> ONNX export/quantization -> ACompiler compile
  -> ACSim/cmodel functional simulation -> optional ACTSim timing simulation
  -> Model-Packer packaging -> board ac_driver/deploy execution
```

Start with:

```text
rhblite_toolchain/README.md
rhblite_toolchain/MANIFEST.md
```

The toolchain directory is source-only. Generated ONNX files, compiler outputs,
logs, NPZ outputs, and packer binaries remain excluded by its local `.gitignore`.

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
