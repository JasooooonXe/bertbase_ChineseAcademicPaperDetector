#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-/root/autodl-fs}"
AUTO_SHUTDOWN_ON_SUCCESS="${AUTO_SHUTDOWN_ON_SUCCESS:-0}"
SHUTDOWN_DELAY_MINUTES="${SHUTDOWN_DELAY_MINUTES:-1}"

python "$REPO_ROOT/train.py" \
  --config "$REPO_ROOT/configs/train_base.yaml" \
  --output-root "$PERSIST_ROOT/ai-detector"

if [[ "$AUTO_SHUTDOWN_ON_SUCCESS" == "1" ]]; then
  echo "Training finished successfully. Scheduling shutdown in ${SHUTDOWN_DELAY_MINUTES} minute(s)."
  shutdown -h +"$SHUTDOWN_DELAY_MINUTES" "Auto shutdown after successful training" \
    || poweroff
fi
