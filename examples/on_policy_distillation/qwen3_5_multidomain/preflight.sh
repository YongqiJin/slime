#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODE="${1:-smoke}"
if [[ "$MODE" != "smoke" && "$MODE" != "production" ]]; then
  echo "usage: $0 [smoke|production]" >&2
  exit 2
fi

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "missing required env: $name" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    echo "missing $label directory: $path" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing $label file: $path" >&2
    exit 1
  fi
}

check_slime_checkpoint_target() {
  local path="$1"
  if [[ -f "$path/latest_checkpointed_iteration.txt" ]]; then
    echo "slime_checkpoint=existing:$path"
  else
    echo "slime_checkpoint=will_initialize_from_ref_load_and_save:$path"
  fi
}

check_hostfile() {
  local hostfile="$1"
  local require_worker="${2:-0}"
  require_file "$hostfile" HOSTFILE
  if ! awk 'NF > 0 {found=1} END {exit found ? 0 : 1}' "$hostfile"; then
    echo "HOSTFILE must contain at least one non-empty line: $hostfile" >&2
    exit 1
  fi
  if [[ "$require_worker" == "1" ]] && ! awk -v master="$MASTER_ADDR" 'NF > 0 && $1 != master {found=1} END {exit found ? 0 : 1}' "$hostfile"; then
    echo "two-node smoke HOSTFILE must contain at least one worker IP different from MASTER_ADDR=$MASTER_ADDR: $hostfile" >&2
    exit 1
  fi
}

check_teacher_runtime() {
  local runtime_config="${TEACHER_RUNTIME_CONFIG:-}"
  if [[ -n "$runtime_config" ]]; then
    require_file "$runtime_config" TEACHER_RUNTIME_CONFIG
    python - "$runtime_config" <<'PY'
import sys
from slime.rollout.on_policy_distillation import load_teacher_runtime_config

config = load_teacher_runtime_config(sys.argv[1])
print("teacher_runtime_default=", config.default_teacher)
print("teacher_runtime_teachers=", ",".join(sorted(config.teachers_by_name)))
PY
    return
  fi

  if [[ -n "${OPD_TEACHER_RUN_DIR:-}" && -f "$OPD_TEACHER_RUN_DIR/teacher_runtime.json" ]]; then
    runtime_config="$OPD_TEACHER_RUN_DIR/teacher_runtime.json"
    python - "$runtime_config" <<'PY'
import sys
from slime.rollout.on_policy_distillation import load_teacher_runtime_config

config = load_teacher_runtime_config(sys.argv[1])
print("teacher_runtime_default=", config.default_teacher)
print("teacher_runtime_teachers=", ",".join(sorted(config.teachers_by_name)))
PY
    return
  fi

  if [[ -z "${TEACHER_RM_URL:-}" ]]; then
    echo "teacher_runtime_config=not yet generated; source teacher config will be checked" >&2
    return
  fi
  echo "teacher_rm_url=$TEACHER_RM_URL"
}

check_teacher_source_config() {
  if [[ -n "${TEACHER_RM_URL:-}" && -z "${TEACHERS_CONFIG:-}" && -z "${OPD_TEACHER_RUN_DIR:-}" && -z "${TEACHER_RUNTIME_CONFIG:-}" ]]; then
    echo "teacher_source_config=skipped_for_single_teacher_rm_url"
    return
  fi

  local config="${TEACHERS_CONFIG:-$SCRIPT_DIR/teachers.yaml}"
  if [[ ! -f "$config" ]]; then
    if [[ -n "${TEACHER_RM_URL:-}" ]]; then
      echo "teacher_source_config=skipped_missing_for_single_teacher_rm_url"
      return
    fi
    echo "missing TEACHERS_CONFIG file: $config" >&2
    exit 1
  fi

  local output
  output="$(mktemp)"
  BASE_FOLDER="$BASE_FOLDER" python "$SCRIPT_DIR/teacher_pool.py" render-runtime \
    --config "$config" \
    --run-dir "${OPD_TEACHER_RUN_DIR:-/tmp/slime-opd-teachers-preflight}" \
    --output "$output"
  rm -f "$output"
}

require_env BASE_FOLDER
require_env MASTER_ADDR

if [[ "$MODE" == "smoke" ]]; then
  SMOKE_MODEL_SIZE="${SMOKE_MODEL_SIZE:-9B}"
  MODEL_SCRIPT="$ROOT/scripts/models/qwen3.5-${SMOKE_MODEL_SIZE}.sh"
  require_file "$MODEL_SCRIPT" "Qwen3.5 model script"
  require_env SMOKE_DATA_FILE
  require_env HOSTFILE
  require_file "$SMOKE_DATA_FILE" SMOKE_DATA_FILE
  check_hostfile "$HOSTFILE" 1
  require_dir "$BASE_FOLDER/Qwen3.5-${SMOKE_MODEL_SIZE}" "hf checkpoint"
  require_dir "$BASE_FOLDER/Qwen3.5-${SMOKE_MODEL_SIZE}_torch_dist" "torch_dist checkpoint"
  check_slime_checkpoint_target "$BASE_FOLDER/Qwen3.5-${SMOKE_MODEL_SIZE}_slime"
else
  MODEL_SCRIPT="$ROOT/scripts/models/qwen3.5-27B.sh"
  require_file "$MODEL_SCRIPT" "Qwen3.5 model script"
  require_env DATA_FILE
  require_file "$DATA_FILE" DATA_FILE
  if [[ -n "${HOSTFILE:-}" ]]; then
    check_hostfile "$HOSTFILE"
  fi
  require_dir "$BASE_FOLDER/Qwen3.5-27B" "hf checkpoint"
  require_dir "$BASE_FOLDER/Qwen3.5-27B_torch_dist" "torch_dist checkpoint"
  check_slime_checkpoint_target "$BASE_FOLDER/Qwen3.5-27B_slime"
fi

check_teacher_source_config
check_teacher_runtime

echo "preflight_ok mode=$MODE master_addr=$MASTER_ADDR base_folder=$BASE_FOLDER"
