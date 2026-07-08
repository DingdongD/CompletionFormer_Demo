# CompletionFormer HW128 ckpt00059 Error Attribution

## Inputs

- Checkpoint: `/root/demo/CompletionFormer/ref_model_hw/model_00059.pt`
- Board output: `/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/board_decoder_system_128x128_ckpt00059_train32_clearwrdone_outputs.npz`
- Feature/reference data: `/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/nyu_sample0_real_features_float_ckpt00059.npz`
- Corrected visualization: `/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/nyu_sample0_ref_vs_board_pred_ckpt00059_train32_clearwrdone_loadednlspn.png`

## Runtime Issue Already Fixed

The previous first-frame pollution was caused by stale `wr_done`.

- Stale state: `PL[0x20] bit0`
- Clear strobe: write `PL[0x00] = 0x100`
- Runner option: `--clear-wr-done-before-run`

With this enabled, the runner no longer needs `--use-second-run`, and the previous double-run latency is removed.

## Evaluation Fix

`tools/visualize_completionformer_board_real_pred.py` now loads `prop_layer.*` from the checkpoint into `NLSPNHW`.

Without this fix, `board_pred` was not a valid final NLSPN result because the Host NLSPN propagation weights were zero-initialized.

## Layerwise Error With Clear-wr-done

| tensor | abs mean | abs p95 | abs max | rmse |
| --- | ---: | ---: | ---: | ---: |
| fd6_f | 0.014208 | 0.080535 | 0.391482 | 0.039753 |
| fd5_f | 0.044467 | 0.255813 | 0.916024 | 0.107445 |
| fd4_f | 0.053885 | 0.208874 | 0.963151 | 0.095051 |
| fd3_f | 0.047248 | 0.204418 | 1.393213 | 0.095500 |
| fd2_f | 0.037438 | 0.159761 | 1.224261 | 0.075217 |
| dep_fd1_f | 0.024786 | 0.101835 | 2.708528 | 0.053227 |
| init_depth_f / pred_init | 0.066315 | 0.222511 | 0.679834 | 0.100928 |
| gd_fd1_f | 0.073793 | 0.282209 | 2.816748 | 0.137113 |
| guide_f | 0.500372 | 1.735570 | 12.936598 | 0.875035 |
| cf_fd1_f | 0.047031 | 0.223242 | 2.451305 | 0.110761 |
| confidence_f | 0.615178 | 0.977549 | 0.999999 | 0.646309 |

The decoder/depth path is reasonably aligned: `pred_init` mean error is `0.066315`.

## Hybrid NLSPN Attribution

All rows use the corrected checkpoint-loaded Host NLSPN.

| variant | final pred abs mean | p95 | max | rmse |
| --- | ---: | ---: | ---: | ---: |
| board init + board guide + board confidence | 0.218045 | 0.843321 | 8.052783 | 0.425115 |
| board init + board guide + Host sigmoid confidence | 0.021220 | 0.085317 | 0.757388 | 0.040173 |
| board init + board guide + ref confidence | 0.018499 | 0.069041 | 0.570070 | 0.034294 |
| board init + ref guide + board confidence | 0.213446 | 0.828376 | 6.190525 | 0.413045 |
| board init + ref guide + Host sigmoid confidence | 0.019191 | 0.074853 | 0.254633 | 0.034512 |
| board init + ref guide + ref confidence | 0.016149 | 0.054580 | 0.210998 | 0.028408 |
| ref init + board guide + board confidence | 0.209895 | 0.800402 | 9.072971 | 0.426599 |
| ref init + ref guide + ref confidence | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

The dominant residual source is `confidence_f`, not `init_depth` or NLSPN.

## Confidence Root Cause

The model file is:

`/root/demo/models/completionformer_test/head_tiny_cf_dec0_conv_sigmoid_128x128_ckpt.py`

Its PyTorch forward is `Conv2d + torch.sigmoid`.

The patched ONNX used for compilation is:

`/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/onnx_models_patched_train32/completionformer_test.head_tiny_cf_dec0_conv_sigmoid_128x128_ckpt.onnx`

Its nodes are:

```text
/conv/Conv Conv
/Sigmoid HardSwish
```

So the compiled board path is not actually running Sigmoid. This explains the invalid confidence range:

- board confidence: `-0.460938 .. 0.992188`, mean `0.078561`
- ref confidence: `0.002305 .. 1.000000`, mean `0.693739`

Using the same board `cf_fd1_f` input but computing `Conv + Sigmoid` on Host gives:

- Host sigmoid confidence vs ref: abs mean `0.023936`, rmse `0.041909`
- Final pred with Host sigmoid confidence: abs mean `0.021220`

## Current Conclusion

The RHB decoder/depth path is usable after `wr_done` clearing. The remaining major error is the confidence branch activation: Sigmoid is being compiled as HardSwish and is not semantically equivalent.

Recommended scheduling for the current exact-eval path:

```text
RHB: decoder, dep head, gd head, cf_dec1
Host: cf_dec0 Conv + Sigmoid, NLSPN
```

For a pure RHB path, the model should be retrained with the same activation that the compiler actually supports, or the compiler/runtime needs a real Sigmoid implementation.

## RHB Conv-only cf_dec0 + Host Sigmoid Test

A new submodel was compiled and packed:

`completionformer_test.head_tiny_cf_dec0_conv_only_128x128_ckpt`

This keeps `cf_dec0.conv` on RHB and applies true sigmoid on Host. The logits submodel uses:

- input scale: `16.0`
- output scale: `8.0`

New packer:

`/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/packer_decoder_system_128x128_ckpt00059_traincalib32_convonlycf_rram_false`

Board run:

`/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/board_decoder_system_128x128_ckpt00059_train32_convonlycf_hostsigmoid_outputs.npz`

Visualization:

`/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/nyu_sample0_ref_vs_board_pred_ckpt00059_train32_convonlycf_hostsigmoid.png`

Metrics:

| variant | abs mean | p95 | max | rmse |
| --- | ---: | ---: | ---: | ---: |
| conv-only RHB + Host sigmoid final pred vs ref | 0.021273 | 0.085987 | 0.758650 | 0.040104 |
| conv-only RHB + Host sigmoid pred_init vs ref | 0.066315 | 0.222511 | 0.679834 | 0.100928 |
| old HardSwish confidence final pred vs ref | 0.218045 | 0.843321 | 8.052783 | 0.425115 |

Latency from the board log:

| stage | latency |
| --- | ---: |
| cf_dec0 RHB conv-only | 104.095 ms |
| Host sigmoid | 1.954 ms |
| full tracked decoder system | 1485.161 ms |

Conclusion: this split is feasible and fixes the confidence semantic error without materially increasing latency.

## Multi-sample Board Visualization

Ran the `RHB cf_dec0 conv-only + Host sigmoid` pipeline on the first 4 NYU 128x128 samples.

Output directory:

`/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/multisample_convonlycf_hostsigmoid`

Grid visualization:

`/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/multisample_convonlycf_hostsigmoid/nyu_first4_ref_vs_board_convonlycf_hostsigmoid_grid.png`

Metrics CSV:

`/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/multisample_convonlycf_hostsigmoid/multisample_metrics.csv`

| sample | pred abs mean | p95 | max | rmse | pred_init abs mean | confidence abs mean | latency ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.021273 | 0.085987 | 0.758650 | 0.040104 | 0.066315 | 0.025458 | 1483.460 |
| 1 | 0.019641 | 0.067906 | 0.599941 | 0.039431 | 0.059845 | 0.044302 | 1484.747 |
| 2 | 0.012860 | 0.042504 | 0.196222 | 0.020190 | 0.043924 | 0.046782 | 1487.266 |
| 3 | 0.006328 | 0.018717 | 0.072284 | 0.008973 | 0.033588 | 0.032954 | 1480.737 |

Mean over 4 samples:

- final pred abs mean: `0.015026`
- tracked latency: `1484.053 ms`
- cf_dec0 RHB conv-only latency: about `104-105 ms`
- Host sigmoid latency: about `2 ms`

## 32-sample Val Board Run

Ran all 32 samples from:

`/root/demo/artifacts/nyu_val_representative_32_128x128`

with the current system split:

```text
RHB: decoder, dep head, gd head, cf_dec1, cf_dec0 conv-only
Host/CPU: cf_dec0 sigmoid, NLSPN
Runtime: clear wr_done before each submodel run
```

Source NPZ:

`/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/nyu_val32_source_128x128.npz`

Output directory:

`/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/val32_convonlycf_hostsigmoid`

Metrics CSV:

`/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/val32_convonlycf_hostsigmoid/val32_metrics.csv`

Visualizations:

`/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/val32_convonlycf_hostsigmoid/nyu_val32_ref_vs_board_error_grid.png`

`/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/val32_convonlycf_hostsigmoid/nyu_val32_ref_board_contact_sheet.png`

Aggregate metrics over 32 samples:

| metric | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: |
| final pred abs mean | 0.022418 | 0.015482 | 0.005087 | 0.087885 |
| final pred p95 | 0.096996 | 0.055036 | 0.012703 | 0.519716 |
| final pred max | 0.644724 | 0.490847 | 0.092803 | 1.847738 |
| final pred rmse | 0.052963 | 0.035372 | 0.008722 | 0.253732 |
| pred_init abs mean | 0.062948 | 0.050664 | 0.033843 | 0.174320 |
| confidence abs mean | 0.042193 | 0.042271 | 0.022044 | 0.061296 |
| guide abs mean | 0.613787 | 0.591789 | 0.427839 | 0.807017 |

Worst 5 samples by final pred abs mean:

| sample | pred abs mean | p95 | max | latency ms |
| ---: | ---: | ---: | ---: | ---: |
| 23 | 0.087885 | 0.432953 | 1.047497 | 1493.597 |
| 16 | 0.084896 | 0.519716 | 1.782694 | 1487.397 |
| 29 | 0.055707 | 0.404137 | 1.656262 | 1485.529 |
| 8 | 0.042170 | 0.200017 | 1.323596 | 1486.021 |
| 21 | 0.029482 | 0.100311 | 1.847738 | 1489.200 |

Latency:

| set | tracked latency mean | median | max | cf_dec0 RHB mean | cf_dec0 RHB median | cf_dec0 RHB max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all 32 | 1552.956 ms | 1486.700 ms | 3542.147 ms | 123.314 ms | 104.104 ms | 716.550 ms |
| excluding val31 outlier | 1488.787 ms | 1486.608 ms | 1523.074 ms | 104.206 ms | 104.097 ms | 104.888 ms |

No runtime error was reported for val31; it completed and produced output, but `gd_dec0`, `cf_dec1`, and `cf_dec0` latencies spiked in that run. Treat it as a board/runtime latency outlier unless it reproduces.

## Fused dep/gd/cf dec1 Head Probe

Tested a compiler-aligned fused dec1 head:

```text
input:  head_in [1,96,128,128]
RHB:    Conv2d 96 -> 80, ReLU
Host:   split output into dep_fd1[32], gd_fd1[32], cf_fd1[16]
```

Model file:

`/root/demo/models/completionformer_test/head_tiny_dec1_fused_dep_gd_cf_relu_128x128_ckpt.py`

Runner option:

`--fused-dec1-head`

Packers tested:

`/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/packer_decoder_system_128x128_ckpt00059_traincalib32_fuseddec1_convonlycf_rram_false`

`/root/demo/artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/packer_decoder_system_128x128_ckpt00059_traincalib32_fuseddec1s4_convonlycf_rram_false`

Both compile and cmodel complete. Both also execute on board. The scale=8 run reduced sample0 tracked latency from about `1488 ms` to `1340 ms`, but accuracy degraded substantially.

Sample0 metrics:

| variant | final pred abs mean | p95 | max | rmse |
| --- | ---: | ---: | ---: | ---: |
| baseline separate dec1 heads | 0.021070 | 0.076533 | 0.450950 | 0.038133 |
| fused dec1 output_scale=8 | 0.119938 | 0.292090 | 1.922341 | 0.164580 |
| fused dec1 output_scale=4 | 0.126806 | 0.314145 | 2.378506 | 0.181589 |

Layer error shows the degradation starts directly at fused dec1 output:

| tensor | baseline mean | fused s8 mean | fused s4 mean |
| --- | ---: | ---: | ---: |
| dep_fd1 | 0.024370 | 0.085380 | 0.092663 |
| gd_fd1 | 0.076720 | 0.321784 | 0.302040 |
| cf_fd1 | 0.049728 | 0.501928 | 0.498481 |

Conclusion: full fusion into one `96 -> 80` 128x128 RHB Conv is executable and faster, but not acceptable for the current numeric path. Changing the unified output scale from `8` to `4` does not fix it. The likely issue is the large full-resolution high-output-channel fused Conv pattern, plus the loss of per-head output scaling. Keep the separate dec1 heads for the accuracy path, or explore smaller partial fusion such as `dep+gd` and `cf` separately.
