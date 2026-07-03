#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

test -f "$BUNDLE_DIR/CompletionFormer/ref_model_hw/model_00059.pt"
test -f "$BUNDLE_DIR/data/nyu_val32_source_128x128.npz"
test -f "$BUNDLE_DIR/packer/packer_decoder_system_128x128_ckpt00059_traincalib32_convonlycf_rram_false/config.yaml"
test -f "$BUNDLE_DIR/scripts/rhb_completionformer_decoder_system_runner_128x128_ckpt.py"

python - <<'PY'
import importlib.util
import sys

for name in ["numpy", "torch", "matplotlib"]:
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"missing python package: {name}")
print("PYTHON_DEPS_OK")
PY

command -v ssh >/dev/null
command -v scp >/dev/null
command -v sshpass >/dev/null

echo "PORTABLE_ENV_OK: $BUNDLE_DIR"
