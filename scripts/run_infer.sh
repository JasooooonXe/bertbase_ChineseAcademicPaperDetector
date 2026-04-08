#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-/root/autodl-fs}"
CHECKPOINT="${CHECKPOINT:-}"
INPUT_FILE="${INPUT_FILE:-$PERSIST_ROOT/ai-detector/data/processed/documents.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-$PERSIST_ROOT/ai-detector/runs/infer/predictions.jsonl}"

if [[ -z "$CHECKPOINT" ]]; then
  echo "Set CHECKPOINT to a checkpoint directory or file path." >&2
  exit 1
fi

python "$REPO_ROOT/infer.py" \
  --checkpoint "$CHECKPOINT" \
  --input-file "$INPUT_FILE" \
  --output-file "$OUTPUT_FILE"
