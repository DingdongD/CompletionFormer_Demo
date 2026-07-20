#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/demo}"
PORT="$ROOT/artifacts/successful_pipelines/completionformer_hw128_ckpt00059_rhb_convonlycf_20260702/portable_runtime"
OUT="${OUT:-$ROOT/artifacts/output_sdformer_approx_attention_matrix_20260720}"
PKG="${PKG:-$OUT/packers}"
RUN_BOARD="${RUN_BOARD:-0}"
BOARD="${BOARD:-root@192.168.115.122}"
BOARD_PASS="${BOARD_PASS:-root}"
BOARD_TIMEOUT_SEC="${BOARD_TIMEOUT_SEC:-240}"

mkdir -p "$OUT" "$PKG"
summary="$OUT/summary.tsv"
printf 'model\tcompile_ok\tcmodel_ok\tpack_ok\tboard_run\tboard_status\tall_same\tmarker\n' > "$summary"

cd "$ROOT"
python "$PORT/agentflow_rhb/scripts/generate_sdformer_approx_attention_modules.py"

mapfile -t models < <(
  find models/sdformer_test/approx_generated -maxdepth 1 -type f -name '*.py' \
    ! -name '__init__.py' -printf '%f\n' \
    | sed 's/\.py$//' \
    | sort \
    | sed 's#^#sdformer_test.approx_generated.#'
)

for m in "${models[@]}"; do
  echo "===== compile/cmodel $m ====="
  make output_root="$OUT" model="$m" cv_model >/tmp/${m//./_}_cv.stdout 2>&1 || true
  make output_root="$OUT" model="$m" compile layout="input0=BCHW" >/tmp/${m//./_}_compile.stdout 2>&1 || true
  make output_root="$OUT" model="$m" seed=1 cmodel >/tmp/${m//./_}_cmodel.stdout 2>&1 || true

  compile_ok=0
  cmodel_ok=0
  pack_ok=0
  board_status="not_run"
  all_same="NA"

  if [[ -f "$OUT/$m/${m}.tmp" || -f "$OUT/$m/${m}_op_insts_ccode.bin" || -f "$OUT/$m/${m}.op_insts.bin" ]]; then
    compile_ok=1
  fi
  if grep -q 'ThreadID 0 completed' "$OUT/$m/cmodel.log" 2>/dev/null; then
    cmodel_ok=1
  fi

  marker=$(grep -nE 'ThreadID 0 completed|Failed to parse|Unsupported tensor layout|UIO interrupt polling timeout|TimeoutExpired|All same:|Error:|Assertion|layout attr|Traceback' \
    "$OUT/$m/compile.log" "$OUT/$m/cmodel.log" "$OUT/$m/model.log" 2>/dev/null \
    | tail -n 12 | tr '\n' ';' | sed 's/\t/ /g' || true)

  if [[ "$cmodel_ok" == "1" ]]; then
    pack_dir="$PKG/packer_${m}"
    rm -rf "$pack_dir"
    python "$PORT/agentflow_rhb/rhb_auto_config/cli.py" pack \
      --workspace "$ROOT" \
      --compile-output-dir "$OUT/$m" \
      --packer-output-dir "$pack_dir" \
      --model-packer-dir "$ROOT/Model-Packer" \
      --timeout-sec 120 >/tmp/${m//./_}_pack.stdout 2>&1 || true
    if [[ -f "$pack_dir/config.yaml" ]]; then
      pack_ok=1
    fi

    if [[ "$RUN_BOARD" == "1" && "$pack_ok" == "1" ]]; then
      python "$PORT/agentflow_rhb/rhb_auto_config/cli.py" board-run \
        --packer-dir "$pack_dir" \
        --board "$BOARD" \
        --password "$BOARD_PASS" \
        --log-path "$OUT/$m/board_approx_attention.log" \
        --timeout-sec "$BOARD_TIMEOUT_SEC" >/tmp/${m//./_}_board.stdout 2>&1 || true
      board_status=$(grep -E 'board_status:' /tmp/${m//./_}_board.stdout | tail -n1 | sed 's/.*board_status: *//')
      all_same=$(grep -E 'all_same:' /tmp/${m//./_}_board.stdout | tail -n1 | sed 's/.*all_same: *//')
      board_marker=$(grep -nE 'All same:|Unsupported tensor layout|UIO interrupt polling timeout|TimeoutExpired|Error:|Traceback' \
        "$OUT/$m/board_approx_attention.log" /tmp/${m//./_}_board.stdout 2>/dev/null \
        | tail -n 12 | tr '\n' ';' | sed 's/\t/ /g' || true)
      marker="${marker}${board_marker}"
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$m" "$compile_ok" "$cmodel_ok" "$pack_ok" "$RUN_BOARD" "$board_status" "$all_same" "$marker" | tee -a "$summary"
done

echo "Wrote $summary"
