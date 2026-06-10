# Qwen3.5 Multi-Domain OPD Teacher Lifecycle

This example keeps teacher process management outside slime. The scripts start an
SGLang teacher pool, health-check each configured endpoint, and write a
`teacher_runtime.json` file that slime can consume with `--opd-teacher-config`.

## Start Teachers

Edit `teachers.yaml` first. Set each teacher's `model_path`, `host`, `gpus`,
`port`, and `tp`. Environment variables such as `${BASE_FOLDER}` are expanded by
`teacher_pool.py`.

```bash
export BASE_FOLDER=/path/to/checkpoints
export OPD_TEACHER_RUN_DIR=/tmp/slime-opd-teachers

bash examples/on_policy_distillation/qwen3_5_multidomain/start_teachers.sh \
  examples/on_policy_distillation/qwen3_5_multidomain/teachers.yaml
```

After every configured teacher passes `/health_generate`, the script writes:

```bash
$OPD_TEACHER_RUN_DIR/teacher_runtime.json
```

## Train With The Runtime Config

Pass the generated runtime config to any OPD training entry point:

```bash
python train.py \
  --rollout-function-path slime.rollout.on_policy_distillation.generate_rollout \
  --loss-type on_policy_distillation \
  --opd-teacher-config "$OPD_TEACHER_RUN_DIR/teacher_runtime.json"
```

For a one-off single teacher, skip the lifecycle scripts and set the existing
`--rm-url http://host:port/generate` path instead.

## Stop Teachers

Stop all processes recorded in the run directory:

```bash
bash examples/on_policy_distillation/qwen3_5_multidomain/stop_teachers.sh \
  "$OPD_TEACHER_RUN_DIR"
```

Teacher logs and process files stay under `OPD_TEACHER_RUN_DIR` for inspection.
