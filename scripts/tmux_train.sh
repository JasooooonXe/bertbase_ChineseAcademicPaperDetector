#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${1:-ai-detector-train}"
shift || true

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed." >&2
  exit 1
fi

if [[ "$#" -eq 0 ]]; then
  COMMAND="bash $REPO_ROOT/scripts/run_train.sh"
else
  COMMAND="$*"
fi

tmux new-session -d -s "$SESSION_NAME" "$COMMAND"
echo "Started session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
