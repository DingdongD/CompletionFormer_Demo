# Inference-only Latency Breakdown, 2026-07-14

Input: NYU 128x128, batch=1.

## Summary

| model | accepted total tracked | load tracked | inference + host glue | main conclusion |
| --- | ---: | ---: | ---: | --- |
| CompletionFormer HW128 | 1486.7 ms median | not included in val32 tracked CSV | 1486.7 ms | inference itself dominates; single packer is already used |
| CSPN ResNetTiny HW128 | 2045.6 / 2197.9 ms | ~1064.3 ms | 981.2 / 1133.6 ms | load is about half, but inference/glue is still ~1.0-1.1s |
| NLSPN HW128 | 4095.6 / 4468.6 ms | not timed in `LATENCY_TOTAL_MS` | 4095.6 / 4468.6 ms | inference/submodel launch dominates; packer load is not the main bottleneck |

## CompletionFormer

`validation/val32_latency_by_stage.csv` contains per-stage tracked execution after the resident decoder package is used. Median total:

```text
1486.6995 ms
```

Median grouped cost:

```text
RHB submodel execution: ~1336.8 ms
Host glue:              ~149.9 ms
Full-resolution heads:  ~1144.0 ms
```

Largest median stages:

```text
dep_dec1_full_rhb                       270.247 ms
gd_dec1_full_rhb                        267.037 ms
dep_dec0_rhb                            236.460 ms
cf_dec1_rhb                             136.263 ms
dec2_resize_upconv_split_sum_host_relu  134.135 ms
gd_dec0_rhb                             129.925 ms
cf_dec0_rhb                             104.104 ms
```

Interpretation: packer load is already amortized. Further gains require reducing full-resolution head launches, moving more glue into board-exact fused subgraphs, or changing the compiler-aligned model/training design.

## CSPN

Accepted exact 2-load runs:

```text
run A total=2045.604 ms load=1064.440 ms inference+glue=981.164 ms
run B total=2197.879 ms load=1064.263 ms inference+glue=1133.616 ms
```

Representative split:

```text
Host glue: ~294-296 ms
RHB execution and per-submodel transfer: ~685-839 ms
```

Largest non-load stages:

```text
host::dec1_input                         ~144 ms
cspn dec1_conv0                          ~101-108 ms
host::dec2_input                         ~56 ms
dec1_conv1                               ~48-92 ms
stage2_b1 conv spikes in one revalidate  ~67-72 ms
refine/depth/guidance full-res convs      ~30-36 ms each
```

Interpretation: fixing all-in-one would save roughly one 530 ms load, but the remaining inference/glue is still about 1 second. After the load issue, optimize full-res dec1/refine/head and host concat/resize glue.

## NLSPN

Accepted 2-pack logs now report three separate runtime buckets:

```text
latest profiled run tracked inference total = 4388.238 ms
PACKER_LOAD_TOTAL_MS                     = 1177.525 ms
  packer_nlspn_final_dec5_dec4_20260714  = 625.337 ms
  packer_nlspn_final_rest                = 552.188 ms
PACKER_SWITCH_FIRST_RUN                  = 595.722 ms
```

The first submodel after switching to `packer_nlspn_final_rest` is:

```text
dec3_partial_oc0_64_ic0_64 = 595.722 ms
```

The remaining same-shape `dec3` partials are only `43.6-44.7 ms`. If the first-run-after-switch is accounted as runtime state cost and replaced by the steady median `44.098 ms`, NLSPN steady inference is approximately:

```text
4388.238 - 595.722 + 44.098 = 3836.614 ms
```

Revalidate grouped cost:

```text
dec3       858.253 ms
dec2       767.003 ms
id_dec1    458.284 ms
gd_dec1    456.140 ms
dec4       314.516 ms
cf_dec1    312.200 ms
confidence 240.647 ms
guidance   239.402 ms
pred_init  229.025 ms
dec5       220.130 ms
```

Interpretation: NLSPN is launch/transfer/full-res split dominated, but packer switch accounting matters. Do not treat the first `dec3_partial` as an intrinsically slow conv; it is absorbing package-switch state cost. The next useful optimization is reducing the number of full-resolution partial conv launches for `dec2`, `id_dec1`, `gd_dec1`, `cf_dec1`, and the heads, or replacing high-split exact decomposition with retrained compiler-aligned larger subgraphs.

## Optimization Priority

1. Prefer inference-only profiling as the primary optimization signal after resident packer is enabled.
2. CompletionFormer: target full-resolution head subgraphs first.
3. CSPN: keep 2-load exact packer; next target `dec1_input`, `dec1_conv0/1`, refine/depth/guidance full-res convs.
4. NLSPN: keep the 2-pack offset-safe partition; report load and first-run-after-switch separately; target dec2/dec1/head split launch count first.
5. Persistent board service helps wall latency and startup jitter, but not the core tracked inference sums above.
