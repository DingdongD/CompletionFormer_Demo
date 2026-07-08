# CompletionFormer/CSPN RHBLite Toolchain

This package documents and preserves the current working toolchain:

```text
PyTorch submodel
  -> ONNX export and quantization
  -> ACompiler compile
  -> ACSim/cmodel functional simulation
  -> optional ACTSim timing simulation
  -> Model-Packer packaging
  -> board ac_driver/deploy execution
```

The package is source-only. It does not include generated ONNX files, compiled
outputs, checkpoints, packer binaries, validation datasets, or board logs.

## Directory Layout

```text
rhblite_toolchain/
  scripts/
    cv_onnx.py
    compile.py
    test_rt.py
    run_compile_sim.sh
    pack_and_board.sh
    export_completionformer_hw128_real_features.py
    rhb_completionformer_decoder_system_runner_128x128_ckpt.py
    visualize_completionformer_board_real_pred.py
    compare_cspn_official_shift_golden.py
  models/
    completionformer_test/
    cspn_test/
  configs/
    activation_scales.csv
    config.yaml
  docs/
```

## Required Workspace

The scripts assume the historical RHBLite workspace layout:

```text
/root/demo/
  ACompiler import path available in Python
  ACTransformer_DS_toolChain_docker/
  Model-Packer/
  arch_16.yaml
  arch_256.yaml
  arch/rhb_arch.yaml
  onnx_models/
  output*/
```

For CompletionFormer ckpt-backed models, set:

```bash
export COMPLETIONFORMER_HW_CKPT=/root/demo/CompletionFormer/ref_model_hw/model_00059.pt
```

If unset, the model helper falls back to the path encoded in
`models/completionformer_test/_ckpt_hw128_common.py`.

## Model Module Contract

Every PyTorch submodel consumed by `cv_onnx.py` must expose:

```python
class Model(torch.nn.Module): ...
ifmap_sz = [[C, H, W], ...]
input_layouts = ["CHW", ...]
```

Optional fields:

```python
op_version = 18
batch_size = 1
tensor_16 = [".bias"]
tensor_4 = []
tensor_flt = []
batch_onnx = False
```

`cv_onnx.py` imports `models.<MODEL>`, creates dummy inputs from `ifmap_sz`,
exports ONNX, simplifies it, and calls `quant_onnx()`. The final compiler input
is expected at:

```text
onnx_models/<MODEL>.onnx
```

## Standard Compile and Simulation Flow

Run one submodel through ONNX export, ACompiler, and ACSim/cmodel:

```bash
cd /root/demo
MODEL=completionformer_test.decoder_tiny_dec6_resize_conv_basicblock_nocbam_4x4_to_8x8_ckpt \
  ROOT=/root/demo/CompletionFormer/rhblite_toolchain \
  /root/demo/CompletionFormer/rhblite_toolchain/scripts/run_compile_sim.sh
```

Important variables:

```bash
MODEL=completionformer_test.head_tiny_cf_dec0_conv_only_128x128_ckpt
OUTPUT_ROOT=output_rhblite_toolchain
ARCH_PATH=arch_16.yaml,arch_256.yaml
LAYOUT=""                 # empty means use model.input_layouts
SEED=1
RUN_ACTSIM=0             # set to 1 for ACTSim
ACTSIM_ARCH=arch/rhb_arch.yaml
```

The cmodel step is the functional gate. It is not a prerequisite for ONNX
export, but it should pass before ACTSim, Model-Packer, or board execution.

Expected cmodel success marker:

```text
ThreadID 0 completed the simulation
```

Failure markers include:

```text
Segmentation fault
dumped core
Assertion
Traceback
Check failed
Aborted
Failed
```

## Optional ACTSim Timing Simulation

Enable timing simulation after cmodel:

```bash
RUN_ACTSIM=1 ACTSIM_TIMEOUT=120s MODEL=<model> \
  ROOT=/root/demo/CompletionFormer/rhblite_toolchain \
  /root/demo/CompletionFormer/rhblite_toolchain/scripts/run_compile_sim.sh
```

Expected ACTSim success marker:

```text
ACTSim Ends
```

ACTSim is stricter than ACSim. A graph can export, compile, and pass cmodel, but
still fail ACTSim because generated per-op YAML metadata, `op_idx`, dependency
waits, or RRAM load scheduling are not compatible with the timing simulator.

## Packaging and Board Deploy

After cmodel passes:

```bash
MODEL=<model> OUTPUT_ROOT=output_rhblite_toolchain \
  /root/demo/CompletionFormer/rhblite_toolchain/scripts/pack_and_board.sh
```

This runs:

```text
Model-Packer/main_packer.py
  -> upload packer to board
  -> python3 deploy.py <packer> <model>
```

Board variables:

```bash
BOARD_HOST=192.168.115.122
BOARD_USER=root
BOARD_PASS=root
BOARD_WORKDIR=/home/root/workspace/demo_vp_xj
```

For the accepted CompletionFormer multi-submodel pipeline, use the specialized
runner instead of the generic deploy wrapper.

## CompletionFormer Pipeline

The accepted 128x128 CompletionFormer path is staged. It does not compile the
original model as one monolithic ONNX graph.

RHB submodels:

```text
dec6 -> dec5 -> dec4 -> dec3
dec2 resize/up-conv chunk0
dec2 resize/up-conv chunk1
dec2 exact block conv0
dec2 exact block conv1
dep_dec1 -> dep_dec0
gd_dec1 -> gd_dec0
cf_dec1 -> cf_dec0 Conv only
```

Host responsibilities:

```text
feature export
concat/resize glue
dequant and requant between submodels
dec2 chunk sum and residual ReLU
cf_dec0 sigmoid
final NLSPN/ref-model comparison
```

The board runner is:

```bash
python3 scripts/rhb_completionformer_decoder_system_runner_128x128_ckpt.py \
  <packer_path> \
  --feature-npz <features.npz> \
  --save-npz <board_outputs.npz> \
  --cf-dec0-host-sigmoid \
  --clear-wr-done-before-run
```

The two flags are part of the accepted schedule:

- `--cf-dec0-host-sigmoid`: RHB runs only the confidence Conv. Host applies true sigmoid.
- `--clear-wr-done-before-run`: clears stale PL `wr_done` before each submodel launch.

To generate real features from the reference checkpoint:

```bash
python scripts/export_completionformer_hw128_real_features.py \
  --ckpt /root/demo/CompletionFormer/ref_model_hw/model_00059.pt \
  --source-npz /root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/nyu_val32_source_128x128.npz \
  --sample-index 0 \
  --out outputs/sample0/features.npz
```

## CompletionFormer Adaptation Rules

The direct official graph is not the current reliable compiler target because it
contains:

- `ConvTranspose2d`
- unfused `BatchNorm2d`
- PVT transformer blocks
- CBAM attention and reductions
- sigmoid confidence path
- high-channel 3x3 decoder convolutions

Current replacements:

- use `Resize + Conv2d` instead of `ConvTranspose2d`
- fold BN into Conv or remove it for compiler tests
- split PVT/CBAM into independent compiler targets
- split large decoder blocks by stage and sometimes by input channel chunk
- run unsupported glue on Host when it preserves semantics more reliably

## CSPN Pipeline

CSPN is also staged. The official propagation pads and shifts depth/guidance
for eight neighbor directions. The compiler-aligned version replaces that with
a fixed sparse 3x3 convolution over the eight direction channels.

Validated equivalence check:

```bash
python scripts/compare_cspn_official_shift_golden.py
```

Recommended compiled stage:

```text
models/cspn_test/cspn_official_prop1_shift_conv_anchor.py
```

Inputs:

```text
raw_depth [1,H,W]
gate8     [8,H,W]
center    [1,H,W]
mask      [1,H,W]
inv_mask  [1,H,W]
```

Computation:

```python
propagated = raw_depth * center + fixed_shift_sum(raw_depth * gate8)
out = inv_mask * propagated + mask * raw_depth
```

Deployment rule:

- keep gate normalization `abs + sum + div` on Host, or train/export the head to
  produce normalized `gate8` and `center`
- use one compiled anchored propagation step repeatedly from Host for
  `prop_time > 1`
- avoid unrolling multiple anchored propagation steps in one ACompiler graph

## What to Commit

Commit this directory as source documentation:

```text
CompletionFormer/rhblite_toolchain/
```

Do not commit:

```text
*.onnx
output*/
packer*/
*.pt
*.zip
*.npz
*.log
*.bin
*.hex
```

Generated artifacts should be published as release assets or stored in external
artifact storage, not in the source repository.
