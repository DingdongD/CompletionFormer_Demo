# RHB Subgraph Load Reduction Status, 2026-07-14

## Question

Can CSPN, NLSPN, and CompletionFormer reduce repeated subgraph loading like the
`vp_test_ok` batch example, where one Model-Packer package is loaded once and
multiple `run_inference()` calls are issued against resident submodels?

## Short Answer

Yes, this is already the right direction and has already been applied where it
is board-exact:

| model | current load strategy | current latency | further all-in-one status |
| --- | --- | ---: | --- |
| CompletionFormer HW128 | one Model-Packer package, 14 submodels, one `load_model()` | stable `~1486-1492 ms` | already single package |
| CSPN ResNetTiny HW128 | two exact packages: stem/backbone + decoder/head | stable `2121.742 ms` tracked, `2682.677 ms` wall | all-in-one loads but is not board-exact |
| NLSPN HW128 | two packages: dec5+dec4 + rest | stable `4282.089 ms` tracked | all-in-one pack fails due `params_wei` offset overflow |

## Current Evidence

### CompletionFormer

Package:

```text
/root/demo/artifacts/successful_pipelines/completionformer_hw128_ckpt00059_rhb_convonlycf_20260702/portable_runtime/packer/packer_decoder_system_128x128_ckpt00059_traincalib32_convonlycf_rram_false
```

Packer contents:

```text
submodels: 14
instructions lines: 156
params_ch lines: 122
params_wei lines: 3204
packed binary bytes: 825472
```

Status: already one load for all accepted RHB decoder/head submodels. Remaining
latency is mostly full-resolution board compute plus Host resize/split/sum glue,
not repeated packer load.

### CSPN

Accepted exact 2-load partition:

```text
packer_cspn_stem_backbone_16pack_20260714     submodels=16
packer_cspn_decoder_heads_15pack_20260714     submodels=15
```

Revalidated latency:

```text
LATENCY_TOTAL_MS 2197.879
LATENCY_WALL_MS 2756.434
load::bundle::packer_cspn_stem_backbone_16pack_20260714 533.628 ms
load::bundle::packer_cspn_decoder_heads_15pack_20260714 530.635 ms
```

Stable accepted headline:

```text
tracked median: 2121.742 ms
wall median: 2682.677 ms
```

Rejected all-in-one:

```text
packer_cspn_padded16_allinone_20260714
submodels=31
packed binary bytes=1146256
```

It loads and runs faster, but is not board-exact:

```text
depth_float mean_abs diff: 1.910499
s2_float mean_abs diff: 0.742950
```

Follow-up all-in-one bisect:

```text
packer_cspn_all31_exactstem_20260714
submodels=31
instructions=198
params_ch=248
params_wei=4440
core binary bytes=1146256
```

The stem submodel has the same config offsets in the accepted 16-pack and in
the all31 pack:

```text
stem offsets: instructions=0, params_ch=0, params_wei=0
stem sizes:   instructions=18, params_ch=12, params_wei=90
```

However, the all31 board output drifts immediately at the first RHB boundary:

```text
input0_i8 exact
s1_i8 max_abs=33 mean_abs=1.53657 nonzero=57.038%
```

The no-stem 30-pack drifts at its first bundled stage:

```text
s1 exact from standalone stem
s2_float max_abs=16.2324 mean_abs=0.658018 nonzero=70.091%
```

A discarded warmup run after loading the all31 package did not fix correctness:

```text
--warmup-after-load-first-run --unit-scales
first nonzero tensor: s1_i8
max_abs=33 mean_abs=1.53657 nonzero=57.038%
```

Decision: keep the 2-load exact partition. The CSPN all-in-one candidate is
rejected by board-exact tensor comparison, not by the nominal 8 MB package
budget. The likely fault class is a Model-Packer/runtime global-load boundary
issue for very large resident bundles; it is not a scale contract error and not
decoder accumulation.

Cleanup: historical CSPN runner experiments, per-submodel packers, repeat2
outputs, and rejected no-stem/all-in-one variants were archived under:

```text
work/deployment_packages/cspn_resnettiny_hw128_w24_step8_stagewise_v3_hostsample/
  archive/legacy_before_bundle_rule_20260714/
```

### NLSPN

Accepted 2-pack partition:

```text
packer_nlspn_final_dec5_dec4_20260714
  submodels=62
  params_wei lines=8160
  packed binary bytes=2107904

packer_nlspn_final_rest
  submodels=27
  params_wei lines=2284
  packed binary bytes=591280
```

Stable accepted headline:

```text
tracked median: 4282.089 ms
accepted runs: 4468.575 ms, 4095.602 ms
```

All-in-one attempt:

```text
submodels=89
pack failed: Value 8232 out of range for bits 298:286
```

Decision: keep the 2-pack partition. The first packer is close to the observed
`params_wei` offset field limit; all-in-one is blocked by packer/instruction
encoding, not by the nominal 8 MB package size.

## Runtime Rule

The runner must cache the currently loaded packer and call `load_model()` only
when the requested submodel maps to a different package. This is the same
principle as the `vp_test_ok` batch runner, generalized to multiple resident
submodels inside each package.

Current implementation status:

- CSPN scale-aware runner uses `self.loaded` and `multi_packer_map`.
- NLSPN runner uses `CURRENT_MODEL_PATH` and `SUBMODEL_BUNDLES`.
- CompletionFormer runner loads one package once.

## Remaining Optimization Space

1. CSPN: root-cause why all-in-one is not board-exact. If fixed, the best case
   removes one `load_model()` call, saving roughly `~530 ms` wall/tracked load
   overhead.
2. NLSPN: all-in-one requires either packer/instruction offset extension or a
   different partition that keeps `params_wei` offsets below the observed
   failing threshold. Current dec5+dec4 pack already reaches `8160` lines, close
   to the observed failure at `8232`.
3. All models: persistent board service can avoid Python process startup and
   repeated accelerator init across multiple input frames. This is separate from
   Model-Packer bundling and should be implemented as a long-running board
   runner for streaming/batch inference.
