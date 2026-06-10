# Qwen3.5 OPD GPU Smoke Runbook

This runbook is the external validation path for the Uni-OPD migration milestone. It is not required for CPU CI. Use it before closing the migration PR or marking the milestone as fully done.

## Scope

Validate the migrated OPD path on a real two-node Qwen3.5 smoke run:

- script-managed SGLang teacher pool starts and writes `teacher_runtime.json`
- slime reads `--opd-teacher-config` or falls back to `--rm-url`
- OPD reverse-KL metrics are emitted during training
- teacher failures are observable instead of silently corrupting loss
- optional correctness balance and margin shift stay disabled unless explicitly enabled

Issue #8 greedy mask is intentionally deferred and is not part of this smoke.

## Prerequisites

- Two GPU nodes with SSH access between the Ray head and workers.
- `HOSTFILE` containing worker node IPs in the first column.
- `MASTER_ADDR` set to the Ray head node IP.
- Qwen3.5 checkpoints under `BASE_FOLDER`.
- A small smoke dataset readable by slime's `--prompt-data` loader.
- `teachers.yaml` edited for the target teacher hosts, ports, GPUs, tensor parallel size, and model paths.
- SGLang importable in the environment used by `teacher_pool.py`.
- Megatron path available through `MEGATRON_PATH` or `/root/Megatron-LM/`.

Expected checkpoint layout for the default smoke:

```bash
$BASE_FOLDER/Qwen3.5-9B
$BASE_FOLDER/Qwen3.5-9B_torch_dist
```

`$BASE_FOLDER/Qwen3.5-9B_slime` is a save/load target. It may already contain a slime checkpoint for resume, but it is not required for a fresh smoke run; slime initializes from `Qwen3.5-9B_torch_dist` when the target has no `latest_checkpointed_iteration.txt`.

Set `SMOKE_MODEL_SIZE=4B` to use the Qwen3.5-4B smoke layout. Teacher models are controlled by `examples/on_policy_distillation/qwen3_5_multidomain/teachers.yaml`; the example defaults to Qwen3.5-27B teachers.

The smoke dataset should include `prompt` and `label` columns by default. Set `INPUT_KEY` and `LABEL_KEY` if the dataset uses different names. Domain routing is optional and uses `sample.metadata["teacher_model_name"]`, then `sample.metadata["domain"]`, then the runtime config default teacher.

## Preflight

Run these checks from the repository root:

```bash
bash -n examples/on_policy_distillation/qwen3_5_multidomain/start_teachers.sh \
  examples/on_policy_distillation/qwen3_5_multidomain/stop_teachers.sh \
  examples/on_policy_distillation/qwen3_5_multidomain/preflight.sh \
  examples/on_policy_distillation/qwen3_5_multidomain/train_27b.sh \
  examples/on_policy_distillation/qwen3_5_multidomain/smoke_2node.sh

BASE_FOLDER=/path/to/checkpoints \
python examples/on_policy_distillation/qwen3_5_multidomain/teacher_pool.py render-runtime \
  --config examples/on_policy_distillation/qwen3_5_multidomain/teachers.yaml \
  --run-dir /tmp/slime-opd-teachers-preflight \
  --output /tmp/slime_teacher_runtime_preflight.json

python - <<'PY'
from slime.rollout.on_policy_distillation import load_teacher_runtime_config
runtime = load_teacher_runtime_config("/tmp/slime_teacher_runtime_preflight.json")
print("default_teacher=", runtime.default_teacher)
print("teachers=", ",".join(sorted(runtime.teachers_by_name)))
PY
```

After model/data paths and teacher config are set, run the local preflight. This checks paths and config only; it does not start Ray or teacher servers:

```bash
export BASE_FOLDER=/path/to/checkpoints
export OPD_TEACHER_RUN_DIR=/tmp/slime-opd-teachers
export TEACHER_RUNTIME_CONFIG=$OPD_TEACHER_RUN_DIR/teacher_runtime.json
export SMOKE_DATA_FILE=/path/to/smoke.parquet
export MASTER_ADDR=<ray-head-ip>
export HOSTFILE=/path/to/two-node-hostfile

bash examples/on_policy_distillation/qwen3_5_multidomain/preflight.sh smoke
```

If the cluster has stale Ray processes, clean them before the run:

```bash
ray stop --force || true
for WORKER_IP in $(awk '{print $1}' "$HOSTFILE"); do
  ssh root@"$WORKER_IP" "ray stop --force" || true
done
```

## Run

Start teachers and generate the runtime config:

```bash
export BASE_FOLDER=/path/to/checkpoints
export OPD_TEACHER_RUN_DIR=/tmp/slime-opd-teachers

bash examples/on_policy_distillation/qwen3_5_multidomain/start_teachers.sh \
  examples/on_policy_distillation/qwen3_5_multidomain/teachers.yaml
```

Run the two-node smoke:

```bash
export TEACHER_RUNTIME_CONFIG=$OPD_TEACHER_RUN_DIR/teacher_runtime.json
export SMOKE_DATA_FILE=/path/to/smoke.parquet
export MASTER_ADDR=<ray-head-ip>
export HOSTFILE=/path/to/two-node-hostfile

bash examples/on_policy_distillation/qwen3_5_multidomain/smoke_2node.sh
```

Stop teachers after the Ray job exits:

```bash
bash examples/on_policy_distillation/qwen3_5_multidomain/stop_teachers.sh "$OPD_TEACHER_RUN_DIR"
```

For a single teacher endpoint smoke, skip the teacher pool and set:

```bash
export TEACHER_RM_URL=http://host:port/generate
unset TEACHER_RUNTIME_CONFIG
```

## Optional Feature Toggles

Keep these disabled for the baseline smoke. Enable only when the dataset carries verified strict-boolean `response_correct` metadata.

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

## Success Evidence

Paste these items into the PR or issue update:

- branch and commit SHA
- model size, node count, GPU count, and key env vars with private paths redacted if needed
- `start_teachers.sh` output showing health checks passed and `teacher_runtime.json` path
- runtime config summary: `default_teacher`, teacher names, and URL count per teacher
- Ray job id, status, and exit code
- training log snippets showing OPD metrics such as `opd_reverse_kl`
- teacher failure signal: no sustained `opd_teacher_logprob_failed` or log line `OPD teacher logprob failures` spike
- `opd_teacher/known_ratio` and `opd_teacher/name_*` metrics when using `--opd-teacher-config`
- `response_correct/known_ratio` and `response_correct/accuracy` when the dataset includes strict-boolean correctness
- teacher log directory and Ray log path

## Failure Triage

| Symptom | First checks |
| --- | --- |
| Teacher health timeout | Check `teachers.yaml` host, port, GPUs, `tp`, model path, and SGLang logs under `$OPD_TEACHER_RUN_DIR/logs`. |
| Runtime config rejected | Run the preflight `load_teacher_runtime_config` check; multi-teacher configs require `default_teacher`. |
| Smoke exits before training | Confirm either `TEACHER_RUNTIME_CONFIG` points to an existing file or `TEACHER_RM_URL` is set. |
| Ray worker does not join | Check `MASTER_ADDR`, `HOSTFILE`, SSH user, and firewall access to Ray ports. |
| CUDA OOM in teacher | Lower teacher `tp` pressure, reduce `mem_fraction_static`, or move teachers to different GPUs/nodes. |
| CUDA OOM in rollout/training | Reduce `ROLLOUT_BATCH_SIZE`, `N_SAMPLES_PER_PROMPT`, `MAX_RESPONSE_LEN`, `ROLLOUT_NUM_GPUS`, or `MAX_TOKENS_PER_GPU`. |
| Dataset key error | Set `INPUT_KEY` and `LABEL_KEY` to match the smoke file schema. |
| Teacher failure metric spikes | Check teacher `/generate` route, network reachability, request logs, and whether selected teacher names/domains exist in runtime config. |
| No teacher distribution metrics | Confirm `--opd-teacher-config` is used; single `--rm-url` mode has no per-teacher config distribution. |

## Result Template

```markdown
## Qwen3.5 OPD GPU Smoke Result

- Date:
- Branch / commit:
- Model size:
- Nodes / GPUs:
- Teacher config:
- Smoke data:
- Commands:
- Result:
- Ray job id / status:
- Key metrics:
  - opd_reverse_kl:
  - opd_teacher/known_ratio:
  - opd_teacher_logprob_failed or failure log:
  - response_correct/known_ratio:
- Logs:
- Follow-ups:
```
