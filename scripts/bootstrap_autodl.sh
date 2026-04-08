#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/root/projects/ai-detector}"
PERSIST_ROOT="${PERSIST_ROOT:-/root/autodl-fs}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$PROJECT_ROOT"
mkdir -p "$PERSIST_ROOT/ai-detector/data/raw"
mkdir -p "$PERSIST_ROOT/ai-detector/data/processed"
mkdir -p "$PERSIST_ROOT/ai-detector/checkpoints"
mkdir -p "$PERSIST_ROOT/ai-detector/runs"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$REPO_ROOT/requirements.txt"

cat <<EOF
Bootstrap complete.
Repo root: $REPO_ROOT
Project root: $PROJECT_ROOT
Persist root: $PERSIST_ROOT
Virtualenv: $VENV_DIR

Next:
  source "$VENV_DIR/bin/activate"
  bash "$REPO_ROOT/scripts/run_prepare.sh"
EOF
