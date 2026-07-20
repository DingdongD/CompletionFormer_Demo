#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/demo}"
PORT="$ROOT/artifacts/successful_pipelines/completionformer_hw128_ckpt00059_rhb_convonlycf_20260702/portable_runtime"
OUT="${OUT:-$ROOT/artifacts/output_sdformer_probe_20260717}"
PKG="${PKG:-$OUT/packers_sdformer_suite}"
BOARD="${BOARD:-root@192.168.115.122}"
BOARD_PASS="${BOARD_PASS:-root}"

mkdir -p "$OUT" "$PKG"

models=(
  sdformer_test.stem_rgb_relu_128
  sdformer_test.stem_dep_relu_128
  sdformer_test.down_conv_l1_128
  sdformer_test.up_conv_l4_16
  sdformer_test.attn_qkv_only_l1_128
  sdformer_test.attn_project_l1_128
  sdformer_test.ffn_project_out_l1_128
  sdformer_test.final_head_chunk16_padded8_128
  sdformer_test.attn_dw_only_72_128
  sdformer_test.ffn_project_in_only_l1_128
  sdformer_test.ffn_project_in_padded144_l1_128
  sdformer_test.ffn_dw_only_138_128
)

summary="$OUT/sdformer_probe_suite_summary.tsv"
printf 'model\tcompile_cmodel\tpack\tboard\tall_same\n' > "$summary"

cd "$ROOT"
for m in "${models[@]}"; do
  echo "===== compile/cmodel $m ====="
  make output_root="$OUT" model="$m" cv_model
  make output_root="$OUT" model="$m" compile layout="input0=BCHW"
  make output_root="$OUT" model="$m" seed=1 cmodel

  pack_dir="$PKG/packer_${m}"
  rm -rf "$pack_dir"
  echo "===== pack $m ====="
  python "$PORT/agentflow_rhb/rhb_auto_config/cli.py" pack \
    --workspace "$ROOT" \
    --compile-output-dir "$OUT/$m" \
    --packer-output-dir "$pack_dir" \
    --model-packer-dir "$ROOT/Model-Packer" \
    --timeout-sec 120

  echo "===== board $m ====="
  python "$PORT/agentflow_rhb/rhb_auto_config/cli.py" board-run \
    --packer-dir "$pack_dir" \
    --board "$BOARD" \
    --password "$BOARD_PASS" \
    --log-path "$OUT/$m/board_suite.log" \
    --timeout-sec 240

  all_same=$(grep -E 'All same:' "$OUT/$m/board_suite.log" | tail -n1 | sed -E 's/.*All same: *//;s/[[:space:]].*//')
  printf '%s\tok\tok\tok\t%s\n' "$m" "${all_same:-NA}" | tee -a "$summary"
done

echo "Wrote $summary"
