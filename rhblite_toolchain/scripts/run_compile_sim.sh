#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORKSPACE="${WORKSPACE:-/root/demo}"
MODEL="${MODEL:?set MODEL, for example completionformer_test.decoder_tiny_dec6_resize_conv_basicblock_nocbam_4x4_to_8x8_ckpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output_rhblite_toolchain}"
ARCH_PATH="${ARCH_PATH:-arch_16.yaml,arch_256.yaml}"
LAYOUT="${LAYOUT:-}"
SEED="${SEED:-1}"
RUN_ACTSIM="${RUN_ACTSIM:-0}"
ACTSIM_ARCH="${ACTSIM_ARCH:-arch/rhb_arch.yaml}"

cd "${WORKSPACE}"
export PYTHONPATH="${ROOT}:${WORKSPACE}:${PYTHONPATH:-}"
mkdir -p onnx_models "${OUTPUT_ROOT}/${MODEL}"

echo "[1/5] Export ONNX and quantize: ${MODEL}"
/opt/conda/bin/python "${ROOT}/scripts/cv_onnx.py" "${MODEL}" > "${OUTPUT_ROOT}/${MODEL}/model.log" 2>&1
mv "${MODEL}_simp_sc.onnx" "onnx_models/${MODEL}.onnx"
mkdir -p "onnx_models/${MODEL}"
mv "${MODEL}"_*.onnx "onnx_models/${MODEL}/" 2>/dev/null || true

echo "[2/5] Compile with ACompiler: ${MODEL}"
compile_args=(
  --model="onnx_models/${MODEL}.onnx"
  --model_py="models.${MODEL}"
  --opt_level=1
  --sim=1
  --codegen=3
  --arch_path="${ARCH_PATH}"
  --split=32
  --output_path="${OUTPUT_ROOT}/${MODEL}/${MODEL}"
  --max_concat_in_cnt=100
  --log_path="${OUTPUT_ROOT}/${MODEL}"
)
if [[ -n "${LAYOUT}" ]]; then
  compile_args+=(--layouts="${LAYOUT}")
else
  compile_args+=(--layouts="")
fi
/opt/conda/bin/python "${ROOT}/scripts/compile.py" "${compile_args[@]}" > "${OUTPUT_ROOT}/${MODEL}/compile.log" 2>&1

echo "[3/5] Run ACSim/cmodel: seed=${SEED}"
/opt/conda/bin/acsim \
  --case_path="${OUTPUT_ROOT}/${MODEL}/${MODEL}" \
  --arch_path="${ARCH_PATH}" \
  --seed="${SEED}" \
  > "${OUTPUT_ROOT}/${MODEL}/cmodel.log" 2>&1

if grep -Eiq 'Segmentation fault|dumped core|Assertion|Traceback|Check failed|Aborted|Failed' "${OUTPUT_ROOT}/${MODEL}/cmodel.log"; then
  tail -n 80 "${OUTPUT_ROOT}/${MODEL}/cmodel.log" >&2 || true
  echo "cmodel failed: ${OUTPUT_ROOT}/${MODEL}/cmodel.log" >&2
  exit 1
fi

echo "[4/5] cmodel passed"

if [[ "${RUN_ACTSIM}" == "1" ]]; then
  echo "[5/5] Run ACTSim timing simulation"
  timeout "${ACTSIM_TIMEOUT:-120s}" ACTSim "${OUTPUT_ROOT}/${MODEL}/${MODEL}/" "${ACTSIM_ARCH}" > "${OUTPUT_ROOT}/${MODEL}/actsim.log" 2>&1
  if ! grep -q "ACTSim Ends" "${OUTPUT_ROOT}/${MODEL}/actsim.log"; then
    tail -n 80 "${OUTPUT_ROOT}/${MODEL}/actsim.log" >&2 || true
    echo "ACTSim did not close: ${OUTPUT_ROOT}/${MODEL}/actsim.log" >&2
    exit 1
  fi
else
  echo "[5/5] ACTSim skipped. Set RUN_ACTSIM=1 to enable."
fi

echo "OK: ${OUTPUT_ROOT}/${MODEL}"
