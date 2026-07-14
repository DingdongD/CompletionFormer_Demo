#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BOARD="${BOARD:-root@192.168.115.122}"
BOARD_PASS="${BOARD_PASS:-root}"
BOARD_WORK="${BOARD_WORK:-/home/root/workspace/demo_vp_xj/packers/rhb_auto_nlspn_val32_strict_20260714}"
LIMIT="${LIMIT:-32}"
START="${START:-0}"

WORK="$ROOT/artifacts/rhb_auto_config_framework/work"
REPORTS="$ROOT/artifacts/rhb_auto_config_framework/reports"
FEATURE_DIR="$WORK/nlspn_calib32_features"
OUT_DIR="$WORK/nlspn_val32_strict_outputs"
LOG_DIR="$REPORTS/nlspn_val32_strict_logs"
SUMMARY="$REPORTS/nlspn_val32_strict_summary.csv"
APPEND_SUMMARY="${APPEND_SUMMARY:-0}"

PACKER_DEC54="$WORK/packer_nlspn_final_dec5_dec4_20260714"
PACKER_REST="$WORK/packer_nlspn_final_rest"
RUNNER="$ROOT/artifacts/rhb_auto_config_framework/scripts/nlspn_hw_board_pipeline_runner.py"
SCALES="$WORK/nlspn_val32_effective_scales_predinit_guidance_fixed.csv"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$REPORTS"
test -d "$PACKER_DEC54"
test -d "$PACKER_REST"
test -f "$RUNNER"
test -f "$SCALES"

for packer in "$PACKER_DEC54" "$PACKER_REST"; do
  cp "$SCALES" "$packer/activation_scales.csv"
done

ssh_opts=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

sshpass -p "$BOARD_PASS" ssh "${ssh_opts[@]}" "$BOARD" "rm -rf '$BOARD_WORK' && mkdir -p '$BOARD_WORK'"
sshpass -p "$BOARD_PASS" scp "${ssh_opts[@]}" -r "$PACKER_DEC54" "$BOARD:$BOARD_WORK/packer_nlspn_final_dec5_dec4_20260714"
sshpass -p "$BOARD_PASS" scp "${ssh_opts[@]}" -r "$PACKER_REST" "$BOARD:$BOARD_WORK/packer_nlspn_final_rest"
sshpass -p "$BOARD_PASS" scp "${ssh_opts[@]}" "$RUNNER" "$BOARD:$BOARD_WORK/"

if [[ "$APPEND_SUMMARY" != "1" || ! -f "$SUMMARY" ]]; then
  printf 'sample,total_ms,pred_l1,pred_rmse,pred_max_abs,pred_init_l1,guidance_l1\n' > "$SUMMARY"
fi

end=$((START + LIMIT))
for ((idx=START; idx<end; idx++)); do
  sample="$(printf 'sample%02d' "$idx")"
  feature="$FEATURE_DIR/${sample}_features.npz"
  board_out="${sample}_strict.npz"
  log="$LOG_DIR/${sample}_strict.log"
  test -f "$feature"

  echo "RUN $sample"
  sshpass -p "$BOARD_PASS" scp "${ssh_opts[@]}" "$feature" "$BOARD:$BOARD_WORK/${sample}_features.npz" >/dev/null
  sshpass -p "$BOARD_PASS" ssh "${ssh_opts[@]}" "$BOARD" \
    "cd '$BOARD_WORK' && python3 nlspn_hw_board_pipeline_runner.py packer_nlspn_final_dec5_dec4_20260714 \
      --bundle dec3=packer_nlspn_final_rest \
      --bundle dec2=packer_nlspn_final_rest \
      --bundle id_dec1=packer_nlspn_final_rest \
      --bundle gd_dec1=packer_nlspn_final_rest \
      --bundle cf_dec1=packer_nlspn_final_rest \
      --bundle pred_init=packer_nlspn_final_rest \
      --bundle guidance=packer_nlspn_final_rest \
      --bundle confidence=packer_nlspn_final_rest \
      --input-npz ${sample}_features.npz \
      --save $board_out" | tee "$log"

  sshpass -p "$BOARD_PASS" scp "${ssh_opts[@]}" "$BOARD:$BOARD_WORK/$board_out" "$OUT_DIR/$board_out" >/dev/null

  python3 - "$sample" "$log" "$SUMMARY" <<'PY'
import csv
import re
import sys
from pathlib import Path

sample, log_path, summary_path = sys.argv[1:4]
text = Path(log_path).read_text(errors="replace")

def find_float(pattern: str, default: str = "") -> str:
    m = re.search(pattern, text)
    return f"{float(m.group(1)):.6f}" if m else default

def compare_metric(name: str, metric: str) -> str:
    m = re.search(rf"COMPARE {re.escape(name)}: ([^\n]+)", text)
    if not m:
        return ""
    mm = re.search(rf"{metric}=([0-9.eE+-]+)", m.group(1))
    return f"{float(mm.group(1)):.6f}" if mm else ""

row = {
    "sample": sample,
    "total_ms": find_float(r"LATENCY_TOTAL_MS\s+([0-9.]+)"),
    "pred_l1": compare_metric("pred", "mean_abs"),
    "pred_rmse": compare_metric("pred", "rmse"),
    "pred_max_abs": compare_metric("pred", "max_abs"),
    "pred_init_l1": compare_metric("pred_init", "mean_abs"),
    "guidance_l1": compare_metric("guidance", "mean_abs"),
}
with Path(summary_path).open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    writer.writerow(row)
print(row)
PY
done

echo "DONE: $SUMMARY"
echo "DONE: $OUT_DIR"
