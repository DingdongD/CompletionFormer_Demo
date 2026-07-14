#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_INDEX="${1:-0}"
BOARD="${BOARD:-root@192.168.115.122}"
BOARD_PASS="${BOARD_PASS:-root}"
BOARD_WORK="${BOARD_WORK:-/home/root/workspace/demo_vp_xj/packers/completionformer_hw128_ckpt00059_portable}"

PACKER="$BUNDLE_DIR/packer/packer_decoder_system_128x128_ckpt00059_traincalib32_convonlycf_rram_false"
RUNNER="$BUNDLE_DIR/scripts/rhb_completionformer_decoder_system_runner_128x128_ckpt.py"
EXPORT_SCRIPT="$BUNDLE_DIR/scripts/export_completionformer_hw128_real_features.py"
VIS_SCRIPT="$BUNDLE_DIR/scripts/visualize_completionformer_board_real_pred.py"
CKPT="$BUNDLE_DIR/CompletionFormer/ref_model_hw/model_00059.pt"
SOURCE_NPZ="$BUNDLE_DIR/data/nyu_val32_source_128x128.npz"
OUT="$BUNDLE_DIR/outputs/sample${SAMPLE_INDEX}"

mkdir -p "$OUT"

FEATURE_NPZ="$OUT/nyu_val${SAMPLE_INDEX}_features_ckpt00059.npz"
BOARD_NPZ="$OUT/board_val${SAMPLE_INDEX}_convonlycf_hostsigmoid_outputs.npz"
BOARD_LOG="$OUT/board_val${SAMPLE_INDEX}_convonlycf_hostsigmoid.log"

python "$EXPORT_SCRIPT" \
  --ckpt "$CKPT" \
  --source-npz "$SOURCE_NPZ" \
  --sample-index "$SAMPLE_INDEX" \
  --out "$FEATURE_NPZ"

sshpass -p "$BOARD_PASS" ssh -o StrictHostKeyChecking=no "$BOARD" "rm -rf '$BOARD_WORK' && mkdir -p '$BOARD_WORK'"
sshpass -p "$BOARD_PASS" scp -o StrictHostKeyChecking=no -r "$PACKER" "$BOARD:$BOARD_WORK/packer"
sshpass -p "$BOARD_PASS" scp -o StrictHostKeyChecking=no "$RUNNER" "$FEATURE_NPZ" "$BOARD:$BOARD_WORK/"

sshpass -p "$BOARD_PASS" ssh -o StrictHostKeyChecking=no "$BOARD" \
  "cd '$BOARD_WORK' && python3 rhb_completionformer_decoder_system_runner_128x128_ckpt.py '$BOARD_WORK/packer' \
    --feature-npz '$BOARD_WORK/$(basename "$FEATURE_NPZ")' \
    --save-npz '$BOARD_WORK/$(basename "$BOARD_NPZ")' \
    --cf-dec0-host-sigmoid \
    --clear-wr-done-before-run \
    --lock-cpu-max-freq \
    --mlockall \
    --pretouch-inputs \
    --disable-gc" | tee "$BOARD_LOG"

sshpass -p "$BOARD_PASS" scp -o StrictHostKeyChecking=no "$BOARD:$BOARD_WORK/$(basename "$BOARD_NPZ")" "$BOARD_NPZ"

python "$VIS_SCRIPT" \
  --feature-npz "$FEATURE_NPZ" \
  --board-npz "$BOARD_NPZ" \
  --ckpt "$CKPT" \
  --out-png "$OUT/nyu_val${SAMPLE_INDEX}_ref_vs_board_convonlycf_hostsigmoid.png" \
  --out-npz "$OUT/nyu_val${SAMPLE_INDEX}_ref_vs_board_convonlycf_hostsigmoid.npz"

echo "DONE: $OUT"
