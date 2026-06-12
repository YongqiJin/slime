# Qwen3.5 Multi-Domain OPD Launchers

These short entry scripts cover the migrated Uni-OPD-style workflow where a Qwen3.5 student learns from a script-managed pool of SGLang teachers.

The Qwen3.5 model arguments and loss masking used here come from slime's existing Qwen3.5 support. This example does not depend on, or reintroduce, Uni-OPD `qwen3-next` compatibility edits.

## Files

- `teachers.yaml`: example SGLang teacher pool config.
- `teacher_pool.py`: local/SSH process manager used by the shell entry points.
- `launch.py`: Python launcher that owns Ray startup and training argument assembly.
- `preflight.sh`: validates env vars, checkpoint layout, hostfile, and teacher runtime config without starting Ray or teachers.
- `start_teachers.sh`: starts teachers, waits for `/health_generate`, and writes `teacher_runtime.json`.
- `stop_teachers.sh`: stops teacher processes recorded in the run directory.
- `train_27b.sh`: short production entry point for a Qwen3.5-27B student.
- `smoke_2node.sh`: short two-node smoke entry point, defaulting to Qwen3.5-9B; set `SMOKE_MODEL_SIZE=4B` for Qwen3.5-4B.

## Expected Checkpoints And Data

`BASE_FOLDER` is the checkpoint root. The production launcher requires:

- `$BASE_FOLDER/Qwen3.5-27B`
- `$BASE_FOLDER/Qwen3.5-27B_torch_dist`

The smoke launcher requires the same pattern for `SMOKE_MODEL_SIZE`:

- `$BASE_FOLDER/Qwen3.5-9B` or `$BASE_FOLDER/Qwen3.5-4B`
- `$BASE_FOLDER/Qwen3.5-9B_torch_dist` or `$BASE_FOLDER/Qwen3.5-4B_torch_dist`

`$BASE_FOLDER/Qwen3.5-27B_slime` and `$BASE_FOLDER/Qwen3.5-${SMOKE_MODEL_SIZE}_slime` are save/load targets. If they already contain `latest_checkpointed_iteration.txt`, training resumes from them; otherwise slime initializes from `--ref-load` and writes new checkpoints there.

Training data must be readable by slime's existing `--prompt-data` loader. The default input fields are `prompt` and `label`; override them with `INPUT_KEY` and `LABEL_KEY` when the dataset uses different names. Domain routing uses `sample.metadata["teacher_model_name"]` first, then `sample.metadata["domain"]`, then the runtime config default teacher.
Set `LABEL_KEY=` for prompt-only datasets that do not contain a top-level label column.

## Required Environment

- `BASE_FOLDER`: checkpoint root described above.
- `DATA_FILE`: production training file used by `train_27b.sh`.
- `SMOKE_DATA_FILE`: smoke training file used by `smoke_2node.sh`.
- `MASTER_ADDR`: Ray head node IP.
- `HOSTFILE`: optional for production, required for two-node smoke; first column contains worker IPs.
- `OPD_TEACHER_RUN_DIR`: teacher run directory for pids, logs, and generated `teacher_runtime.json`.
- `TEACHER_RUNTIME_CONFIG`: optional explicit runtime config path; defaults to `$OPD_TEACHER_RUN_DIR/teacher_runtime.json` when `OPD_TEACHER_RUN_DIR` is set.
- `TEACHER_RM_URL`: optional single-teacher fallback endpoint when no runtime config is set.

Useful optional overrides:

- `GPUS_PER_NODE`, `ACTOR_NUM_NODES`, `ROLLOUT_NUM_GPUS`
- `NUM_ROLLOUT`, `ROLLOUT_BATCH_SIZE`, `N_SAMPLES_PER_PROMPT`, `MAX_RESPONSE_LEN`
- `GLOBAL_BATCH_SIZE`, `LR`, `OPD_KL_COEF`
- `TP_SIZE`, `PP_SIZE`, `CP_SIZE`, `MAX_TOKENS_PER_GPU`
- `RAY_DASHBOARD_PORT`, `MEGATRON_PATH`
- `SLIME_RAY_WORKERS_PRESTARTED=1`: skip launcher SSH worker startup when the platform command already starts Ray workers, such as DLC/Kubernetes jobs.

## Teacher Lifecycle

Teacher servers are script-managed. slime reads `teacher_runtime.json`; it does not start or stop SGLang teacher processes.

Edit `teachers.yaml` before launching. At minimum, set teacher `model_path`, `host`, `gpus`, `port`, and `tp`. Values like `${BASE_FOLDER}` are expanded by `teacher_pool.py` at runtime.

Run a local preflight before starting services:

```bash
export BASE_FOLDER=/path/to/checkpoints
export SMOKE_DATA_FILE=/path/to/smoke.parquet
export MASTER_ADDR=10.0.0.1
export HOSTFILE=/path/to/two-node-hostfile
export OPD_TEACHER_RUN_DIR=/tmp/slime-opd-teachers

bash examples/on_policy_distillation/qwen3_5_multidomain/preflight.sh smoke
```

For production, set `DATA_FILE` instead of `SMOKE_DATA_FILE` and run `preflight.sh production`.

Start teachers:

```bash
export BASE_FOLDER=/path/to/checkpoints
export OPD_TEACHER_RUN_DIR=/tmp/slime-opd-teachers

bash examples/on_policy_distillation/qwen3_5_multidomain/start_teachers.sh \
  examples/on_policy_distillation/qwen3_5_multidomain/teachers.yaml
```

`start_teachers.sh` writes `$OPD_TEACHER_RUN_DIR/teacher_runtime.json` after every configured instance passes `/health_generate`.

Stop teachers:

```bash
bash examples/on_policy_distillation/qwen3_5_multidomain/stop_teachers.sh "$OPD_TEACHER_RUN_DIR"
```

For a one-off single teacher, skip the lifecycle scripts and set `TEACHER_RM_URL=http://host:port/generate`; both launchers keep that compatibility path.

## Production Qwen3.5-27B

Use the generated runtime config from the teacher lifecycle:

```bash
export BASE_FOLDER=/path/to/checkpoints
export OPD_TEACHER_RUN_DIR=/tmp/slime-opd-teachers
export TEACHER_RUNTIME_CONFIG=$OPD_TEACHER_RUN_DIR/teacher_runtime.json
export DATA_FILE=/path/to/train.parquet
export MASTER_ADDR=10.0.0.1
export HOSTFILE=/path/to/hostfile

bash examples/on_policy_distillation/qwen3_5_multidomain/train_27b.sh
```

`HOSTFILE` is optional for single-node production. When present, workers are joined to the Ray head before the Ray job is submitted.

## Two-Node Smoke

The smoke path uses the same teacher lifecycle as production: start teachers first, pass the generated runtime config to slime, then stop teachers after the job.

```bash
export BASE_FOLDER=/path/to/checkpoints
export OPD_TEACHER_RUN_DIR=/tmp/slime-opd-teachers
export TEACHER_RUNTIME_CONFIG=$OPD_TEACHER_RUN_DIR/teacher_runtime.json
export SMOKE_DATA_FILE=/path/to/smoke.parquet
export MASTER_ADDR=10.0.0.1
export HOSTFILE=/path/to/two-node-hostfile

bash examples/on_policy_distillation/qwen3_5_multidomain/smoke_2node.sh
```

Use `SMOKE_MODEL_SIZE=4B` for a smaller Qwen3.5-4B smoke. The default is `9B`.

## Optional OPD Features

The launchers keep optional migration features disabled by default. Enable them explicitly when the dataset carries verified correctness labels.

Correctness balance:

```bash
export ENABLE_OPD_CORRECTNESS_BALANCE=1
export OPD_CORRECTNESS_BALANCE_MODE=global
export OPD_CORRECTNESS_BALANCE_RATIO=1.0
```

Margin shift:

```bash
export ENABLE_OPD_MARGIN_SHIFT=1
export OPD_MARGIN_SCOPE=local
export OPD_MARGIN_MODE=mean
export OPD_MARGIN_DELTA=0.0
export OPD_MARGIN_DIRECTION=correct_up
```

Both features only treat strict boolean `response_correct` metadata as known. Missing or non-boolean correctness is preserved as unknown.

## Generated Runtime Config

The generated JSON matches slime's `--opd-teacher-config` schema:

```json
{
  "default_teacher": "general",
  "teachers": [
    {
      "name": "math",
      "domains": ["math", "gsm8k"],
      "urls": ["http://127.0.0.1:31001/generate"],
      "metadata": {"model": "Qwen3.5-27B"}
    }
  ]
}
```

Validate a config without starting servers:

```bash
BASE_FOLDER=/path/to/checkpoints \
python examples/on_policy_distillation/qwen3_5_multidomain/teacher_pool.py render-runtime \
  --config examples/on_policy_distillation/qwen3_5_multidomain/teachers.yaml \
  --run-dir /tmp/slime-opd-teachers \
  --output /tmp/slime_teacher_runtime.json
```

Print launch commands without starting servers:

```bash
BASE_FOLDER=/path/to/checkpoints \
python examples/on_policy_distillation/qwen3_5_multidomain/teacher_pool.py start \
  --config examples/on_policy_distillation/qwen3_5_multidomain/teachers.yaml \
  --run-dir /tmp/slime-opd-teachers-dry-run \
  --dry-run
```
