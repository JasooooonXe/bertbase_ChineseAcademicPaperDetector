#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-/root/autodl-fs}"

python "$REPO_ROOT/train.py" \
  --config "$REPO_ROOT/configs/train_base.yaml" \
  --output-root "$PERSIST_ROOT/ai-detector"
