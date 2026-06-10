#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec python3 "$ROOT/examples/on_policy_distillation/qwen3_5_multidomain/launch.py" production "$@"
