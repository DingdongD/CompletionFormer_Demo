#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for idx in $(seq 0 31); do
  "$BUNDLE_DIR/run_board_single_sample.sh" "$idx"
done
