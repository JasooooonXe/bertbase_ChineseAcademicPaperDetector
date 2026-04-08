#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-/root/autodl-fs}"
INPUT_DIR="${INPUT_DIR:-$PERSIST_ROOT/ai-detector/data/raw}"
OUTPUT_DIR="${OUTPUT_DIR:-$PERSIST_ROOT/ai-detector/data/processed}"
TOKENIZER_NAME="${TOKENIZER_NAME:-hfl/chinese-roberta-wwm-ext}"

python "$REPO_ROOT/prepare_data.py" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --tokenizer-name "$TOKENIZER_NAME"
