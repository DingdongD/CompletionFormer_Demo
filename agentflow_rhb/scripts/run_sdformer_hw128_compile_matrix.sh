#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/demo}"
OUT="${OUT:-$ROOT/artifacts/output_sdformer_hw128_generated_compile_20260720}"
LIST_FILE="${LIST_FILE:-}"
LIMIT="${LIMIT:-0}"

mkdir -p "$OUT"
summary="$OUT/compile_cmodel_summary.tsv"
printf 'model\tcv\tcompile\tcmodel\tmarker\n' > "$summary"

cd "$ROOT"

if [[ -n "$LIST_FILE" ]]; then
  mapfile -t models < "$LIST_FILE"
else
  mapfile -t models < <(
    find models/sdformer_test/generated -maxdepth 1 -type f -name '*.py' \
      ! -name '__init__.py' -printf '%f\n' \
      | sed 's/\.py$//' \
      | sort \
      | sed 's#^#sdformer_test.generated.#'
  )
fi

count=0
for m in "${models[@]}"; do
  [[ -z "$m" ]] && continue
  count=$((count + 1))
  if [[ "$LIMIT" != "0" && "$count" -gt "$LIMIT" ]]; then
    break
  fi
  echo "===== $m cv_model ====="
  make output_root="$OUT" model="$m" cv_model
  cv_rc=$?
  echo "===== $m compile ====="
  make output_root="$OUT" model="$m" compile layout="input0=BCHW"
  comp_rc=$?
  echo "===== $m cmodel ====="
  make output_root="$OUT" model="$m" seed=1 cmodel
  cm_rc=$?
  marker=$(grep -nE 'ThreadID 0 completed|Failed to parse|Error:|assertion|Traceback|failed|timeout|not supported|Rewrite failed' \
    "$OUT/$m/cmodel.log" "$OUT/$m/compile.log" "$OUT/$m/model.log" 2>/dev/null \
    | tail -n 10 | tr '\n' ';' | sed 's/\t/ /g' || true)
  printf '%s\t%s\t%s\t%s\t%s\n' "$m" "$cv_rc" "$comp_rc" "$cm_rc" "$marker" | tee -a "$summary"
done

echo "Wrote $summary"
