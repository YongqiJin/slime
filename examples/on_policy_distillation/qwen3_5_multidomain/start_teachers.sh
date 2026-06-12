#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${TEACHERS_CONFIG:-$SCRIPT_DIR/teachers.yaml}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  CONFIG="$1"
  shift
fi

RUN_DIR="${OPD_TEACHER_RUN_DIR:-$SCRIPT_DIR/runs/$(date +%Y%m%d_%H%M%S)}"
python3 "$SCRIPT_DIR/teacher_pool.py" start --config "$CONFIG" --run-dir "$RUN_DIR" "$@"
mkdir -p "$SCRIPT_DIR/runs"
ln -sfn "$RUN_DIR" "$SCRIPT_DIR/runs/latest"
