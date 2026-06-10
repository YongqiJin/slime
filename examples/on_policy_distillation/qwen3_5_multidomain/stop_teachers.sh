#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${OPD_TEACHER_RUN_DIR:-$SCRIPT_DIR/runs/latest}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  RUN_DIR="$1"
  shift
fi

python3 "$SCRIPT_DIR/teacher_pool.py" stop --run-dir "$RUN_DIR" "$@"
