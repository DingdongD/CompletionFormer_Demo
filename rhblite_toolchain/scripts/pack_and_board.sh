#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE="${WORKSPACE:-/root/demo}"
MODEL="${MODEL:?set MODEL}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output_rhblite_toolchain}"
PACKER_DIR="${PACKER_DIR:-Model-Packer}"
PACKER_OUTPUT="${PACKER_OUTPUT:-${OUTPUT_ROOT}/packer_${MODEL}}"
BOARD_HOST="${BOARD_HOST:-192.168.115.122}"
BOARD_USER="${BOARD_USER:-root}"
BOARD_PASS="${BOARD_PASS:-root}"
BOARD_WORKDIR="${BOARD_WORKDIR:-/home/root/workspace/demo_vp_xj}"
BOARD_PACKERS_DIR="${BOARD_PACKERS_DIR:-${BOARD_WORKDIR}/packers}"
BOARD_PACKER_NAME="${BOARD_PACKER_NAME:-packer_${MODEL}}"

cd "${WORKSPACE}"

compiled="${OUTPUT_ROOT}/${MODEL}"
cmodel_log="${compiled}/cmodel.log"
if [[ ! -f "${cmodel_log}" ]]; then
  echo "missing cmodel log: ${cmodel_log}" >&2
  exit 1
fi
if grep -Eiq 'Segmentation fault|dumped core|Assertion|Traceback|Check failed|Aborted|Failed' "${cmodel_log}"; then
  tail -n 80 "${cmodel_log}" >&2 || true
  echo "cmodel log contains failure markers: ${cmodel_log}" >&2
  exit 1
fi

echo "[1/4] Pack compiled model"
python "${PACKER_DIR}/main_packer.py" "${PACKER_OUTPUT}" "${compiled}" --force

if [[ -n "${BOARD_PASS}" ]]; then
  ssh_cmd=(sshpass -p "${BOARD_PASS}" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "${BOARD_USER}@${BOARD_HOST}")
  scp_cmd=(sshpass -p "${BOARD_PASS}" scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
else
  ssh_cmd=(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "${BOARD_USER}@${BOARD_HOST}")
  scp_cmd=(scp -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
fi

echo "[2/4] Upload packer"
"${ssh_cmd[@]}" "mkdir -p '${BOARD_PACKERS_DIR}' && rm -rf '${BOARD_PACKERS_DIR}/${BOARD_PACKER_NAME}'"
"${scp_cmd[@]}" -r "${PACKER_OUTPUT}/" "${BOARD_USER}@${BOARD_HOST}:${BOARD_PACKERS_DIR}/${BOARD_PACKER_NAME}"

echo "[3/4] Run board deploy"
board_log="${PACKER_OUTPUT}/board_test.log"
"${ssh_cmd[@]}" "cd '${BOARD_PACKERS_DIR}' && python3 '${BOARD_WORKDIR}/deploy.py' '${BOARD_PACKER_NAME}/' '${MODEL}'" 2>&1 | tee "${board_log}"

echo "[4/4] Check board log"
if grep -Eiq 'All same: False|UIO interrupt polling timeout|Problem for wait interrupt|Error:|Traceback|Exception|failed|mismatch' "${board_log}"; then
  echo "board log contains failure markers: ${board_log}" >&2
  exit 1
fi

echo "OK: ${board_log}"
