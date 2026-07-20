#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/demo}"
OUT="${OUT:-$ROOT/artifacts/output_sdformer_tokenized_attention_matrix_20260720}"
LIMIT="${LIMIT:-0}"

mkdir -p "$OUT"
summary="$OUT/compile_cmodel_summary.tsv"
printf 'model\tcv_rc\tcompile_rc\tcmodel_rc\tcompile_ok\tcmodel_ok\tlayout\tmarker\n' > "$summary"

cd "$ROOT"

mapfile -t models < <(
  find models/sdformer_test/tokenized_generated -maxdepth 1 -type f -name '*.py' \
    ! -name '__init__.py' -printf '%f\n' \
    | sed 's/\.py$//' \
    | sort \
    | sed 's#^#sdformer_test.tokenized_generated.#'
)

count=0
for m in "${models[@]}"; do
  count=$((count + 1))
  if [[ "$LIMIT" != "0" && "$count" -gt "$LIMIT" ]]; then
    break
  fi

  echo "===== $m cv_model ====="
  make output_root="$OUT" model="$m" cv_model
  cv_rc=$?

  layout="input0=WC,input1=WC"
  if [[ "$m" == *".tokenized_window_softmax_"* ]]; then
    layout="input0=WC"
  fi

  echo "===== $m compile ====="
  make output_root="$OUT" model="$m" compile layout="$layout"
  comp_rc=$?

  echo "===== $m cmodel ====="
  make output_root="$OUT" model="$m" seed=1 cmodel
  cm_rc=$?

  compile_ok=0
  cmodel_ok=0
  if [[ -f "$OUT/$m/${m}_insts.bin" || -f "$OUT/$m/${m}_op_insts_ccode.bin" || -f "$OUT/$m/${m}.tmp" ]]; then
    compile_ok=1
  fi
  if grep -q 'ThreadID 0 completed' "$OUT/$m/cmodel.log" 2>/dev/null; then
    cmodel_ok=1
  fi

  marker=$(grep -nE 'ThreadID 0 completed|Failed to parse|Error:|assertion|Assertion|Traceback|failed|timeout|not supported|Rewrite failed|layout setting' \
    "$OUT/$m/cmodel.log" "$OUT/$m/compile.log" "$OUT/$m/model.log" 2>/dev/null \
    | tail -n 10 | tr '\n' ';' | sed 's/\t/ /g' || true)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$m" "$cv_rc" "$comp_rc" "$cm_rc" "$compile_ok" "$cmodel_ok" "$layout" "$marker" | tee -a "$summary"
done

echo "Wrote $summary"
