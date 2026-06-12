#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _require_env(env: dict[str, str], name: str) -> str:
    value = env.get(name)
    if not value:
        raise SystemExit(f"missing required env: {name}")
    return value


def _env(env: dict[str, str], name: str, default: str) -> str:
    return env.get(name) or default


def _runtime_teacher_args(env: dict[str, str]) -> list[str]:
    runtime_config = env.get("TEACHER_RUNTIME_CONFIG")
    if not runtime_config and env.get("OPD_TEACHER_RUN_DIR"):
        runtime_config = f"{env['OPD_TEACHER_RUN_DIR']}/teacher_runtime.json"

    args = [
        "--custom-rm-path",
        "slime.rollout.on_policy_distillation.reward_func",
        "--custom-reward-post-process-path",
        "slime.rollout.on_policy_distillation.post_process_rewards",
    ]
    if runtime_config:
        return args + ["--opd-teacher-config", runtime_config]

    rm_url = _require_env(env, "TEACHER_RM_URL")
    return args + ["--rm-url", rm_url]


def _optional_opd_args(env: dict[str, str]) -> list[str]:
    args = []
    if env.get("ENABLE_OPD_CORRECTNESS_BALANCE", "0") == "1":
        args += [
            "--rollout-sample-filter-path",
            "slime.rollout.filter_hub.correctness_balance.rollout_sample_filter",
            "--opd-correctness-balance-mode",
            _env(env, "OPD_CORRECTNESS_BALANCE_MODE", "global"),
            "--opd-correctness-balance-ratio",
            _env(env, "OPD_CORRECTNESS_BALANCE_RATIO", "1.0"),
        ]
    if env.get("ENABLE_OPD_MARGIN_SHIFT", "0") == "1":
        args += [
            "--use-opd-margin-shift",
            "--opd-margin-scope",
            _env(env, "OPD_MARGIN_SCOPE", "local"),
            "--opd-margin-mode",
            _env(env, "OPD_MARGIN_MODE", "mean"),
            "--opd-margin-delta",
            _env(env, "OPD_MARGIN_DELTA", "0.0"),
            "--opd-margin-direction",
            _env(env, "OPD_MARGIN_DIRECTION", "correct_up"),
        ]
    return args


def _optional_rollout_args(env: dict[str, str]) -> list[str]:
    args = []
    if env.get("SGLANG_ROUTER_PORT"):
        args += ["--sglang-router-port", env["SGLANG_ROUTER_PORT"]]
    return args


def _is_enabled(env: dict[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _optional_wandb_args(env: dict[str, str], mode: str) -> list[str]:
    if not _is_enabled(env, "ENABLE_WANDB", default=mode == "smoke"):
        return []

    args = [
        "--use-wandb",
        "--wandb-project",
        _env(env, "WANDB_PROJECT", "slime-opd-smoke" if mode == "smoke" else "slime-opd"),
        "--wandb-group",
        _env(env, "WANDB_GROUP", f"qwen3_5_multidomain-{mode}"),
    ]
    if env.get("WANDB_TEAM"):
        args += ["--wandb-team", env["WANDB_TEAM"]]
    if env.get("WANDB_HOST"):
        args += ["--wandb-host", env["WANDB_HOST"]]
    if env.get("WANDB_DIR"):
        args += ["--wandb-dir", env["WANDB_DIR"]]
    if env.get("WANDB_RUN_ID"):
        args += ["--wandb-run-id", env["WANDB_RUN_ID"]]
    if env.get("WANDB_MODE"):
        args += ["--wandb-mode", env["WANDB_MODE"]]

    if not (env.get("WANDB_KEY") or env.get("WANDB_API_KEY") or env.get("WANDB_MODE")):
        args += ["--wandb-mode", "offline"]

    if _is_enabled(env, "DISABLE_WANDB_RANDOM_SUFFIX", default=False):
        args += ["--disable-wandb-random-suffix"]
    if _is_enabled(env, "WANDB_ALWAYS_USE_TRAIN_STEP", default=False):
        args += ["--wandb-always-use-train-step"]
    return args


def _smoke_num_rollout(env: dict[str, str]) -> str:
    value = _env(env, "NUM_ROLLOUT", "4")
    try:
        num_rollout = int(value)
    except ValueError as exc:
        raise SystemExit(f"NUM_ROLLOUT must be an integer for smoke, got {value!r}") from exc
    if num_rollout < 3:
        raise SystemExit(f"smoke requires NUM_ROLLOUT >= 3, got {num_rollout}")
    return value


def _dataset_key_args(env: dict[str, str]) -> list[str]:
    args = ["--input-key", _env(env, "INPUT_KEY", "prompt")]
    label_key = env["LABEL_KEY"] if "LABEL_KEY" in env else "label"
    if label_key:
        args += ["--label-key", label_key]
    return args


def _model_args(root: Path, model_size: str) -> list[str]:
    model_script = root / "scripts" / "models" / f"qwen3.5-{model_size}.sh"
    if not model_script.exists():
        raise SystemExit(f"missing Qwen3.5 model script: {model_script}")

    begin = "__SLIME_MODEL_ARGS_BEGIN__"
    end = "__SLIME_MODEL_ARGS_END__"
    command = (
        f"printf '%s\\0' {shlex.quote(begin)}; "
        f"source {shlex.quote(str(model_script))}; "
        f"printf '%s\\0' \"${{MODEL_ARGS[@]}}\"; "
        f"printf '%s\\0' {shlex.quote(end)}"
    )
    output = subprocess.check_output(["bash", "--noprofile", "--norc", "-c", command])
    parts = [item.decode() for item in output.split(b"\0") if item]
    try:
        begin_index = parts.index(begin)
        end_index = parts.index(end)
    except ValueError as exc:
        raise RuntimeError(f"Could not parse MODEL_ARGS from {model_script}") from exc
    return parts[begin_index + 1 : end_index]


def _runtime_env_json(env: dict[str, str]) -> str:
    env_vars = {
        "PYTHONPATH": _env(env, "MEGATRON_PATH", "/root/Megatron-LM/"),
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "MASTER_ADDR": _require_env(env, "MASTER_ADDR"),
    }
    for name in [
        "WANDB_API_KEY",
        "WANDB_BASE_URL",
        "WANDB_ENTITY",
        "WANDB_MODE",
    ]:
        if env.get(name):
            env_vars[name] = env[name]
    if env.get("WANDB_KEY") and not env.get("WANDB_API_KEY"):
        env_vars["WANDB_API_KEY"] = env["WANDB_KEY"]

    return json.dumps(
        {"env_vars": env_vars},
        separators=(",", ":"),
    )


def _redact_command(command: list[str]) -> list[str]:
    def redact_runtime_env_json(value: str) -> str:
        try:
            runtime_env = json.loads(value)
        except json.JSONDecodeError:
            return "<redacted>"
        env_vars = runtime_env.get("env_vars")
        if isinstance(env_vars, dict):
            for name in list(env_vars):
                if any(token in name.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                    env_vars[name] = "<redacted>"
        return json.dumps(runtime_env, separators=(",", ":"))

    redacted = []
    redact_next = False
    for item in command:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if item.startswith("--runtime-env-json="):
            prefix, value = item.split("=", 1)
            redacted.append(f"{prefix}={redact_runtime_env_json(value)}")
            continue
        redacted.append(item)
        if item in {"--wandb-key"}:
            redact_next = True
    return redacted


def _smoke_log_path(root: Path, env: dict[str, str]) -> Path:
    default_log_root = root / "examples" / "on_policy_distillation" / "qwen3_5_multidomain" / "runs"
    log_root = Path(_env(env, "OPD_SMOKE_LOG_DIR", str(default_log_root)))
    run_id = env.get("OPD_SMOKE_RUN_ID")
    if not run_id:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        job = env.get("DLC_JOB_ID") or "local"
        pod = env.get("POD_NAME") or "master"
        run_id = f"{timestamp}-{job}-{pod}"

    log_dir = log_root / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    latest = log_root / "latest"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(log_dir, target_is_directory=True)
    except OSError:
        pass
    return log_dir / "smoke.log"


def _run(command: list[str], dry_run: bool, log_path: Path | None = None) -> None:
    display_command = _redact_command(command)
    print("+", shlex.join(display_command))
    if dry_run:
        if log_path is not None:
            print(f"Would log command output to {log_path}")
        return

    if log_path is None:
        subprocess.run(command, check=True)
        return

    print(f"Logging command output to {log_path}")
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("+ " + shlex.join(display_command) + "\n")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args)


def _start_ray(root: Path, env: dict[str, str], require_hostfile: bool, dry_run: bool) -> None:
    master_addr = _require_env(env, "MASTER_ADDR")
    gpus_per_node = _env(env, "GPUS_PER_NODE", "8")
    dashboard_port = _env(env, "RAY_DASHBOARD_PORT", "8265")
    _run(
        [
            "ray",
            "start",
            "--head",
            "--node-ip-address",
            master_addr,
            "--num-gpus",
            gpus_per_node,
            "--disable-usage-stats",
            "--dashboard-host=0.0.0.0",
            f"--dashboard-port={dashboard_port}",
        ],
        dry_run,
    )

    hostfile = env.get("HOSTFILE")
    if require_hostfile:
        hostfile = _require_env(env, "HOSTFILE")
    if not hostfile:
        return

    workers = [
        line.split()[0]
        for line in Path(hostfile).read_text().splitlines()
        if line.split() and line.split()[0] != master_addr
    ]
    if env.get("SLIME_RAY_WORKERS_PRESTARTED", "0") == "1":
        print(f"Skipping Ray worker SSH startup for prestarted workers: {', '.join(workers)}")
        return

    commands = [
        [
            "ssh",
            f"root@{worker_ip}",
            (
                "ray stop --force; "
                f"ray start --address={master_addr}:6379 --num-gpus {gpus_per_node} "
                f"--node-ip-address {worker_ip} --disable-usage-stats"
            ),
        ]
        for worker_ip in workers
    ]
    if dry_run:
        for command in commands:
            print("+", shlex.join(command))
        return

    processes = [subprocess.Popen(command) for command in commands]
    for process in processes:
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args)


def _wait_for_ray_nodes(env: dict[str, str], dry_run: bool) -> None:
    expected = env.get("SLIME_RAY_EXPECTED_NODES")
    if not expected:
        return

    expected_nodes = int(expected)
    timeout = int(_env(env, "SLIME_RAY_NODE_WAIT_SECONDS", "600"))
    interval = int(_env(env, "SLIME_RAY_NODE_WAIT_INTERVAL_SECONDS", "5"))
    if dry_run:
        print(f"Waiting for Ray nodes: expected={expected_nodes} timeout={timeout}s")
        return

    import ray

    ray.init(address="auto", ignore_reinit_error=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        alive = sum(1 for node in ray.nodes() if node.get("Alive"))
        if alive >= expected_nodes:
            print(f"Ray cluster ready: {alive}/{expected_nodes} alive nodes")
            return
        print(f"Waiting for Ray nodes: {alive}/{expected_nodes}")
        time.sleep(interval)

    alive = sum(1 for node in ray.nodes() if node.get("Alive"))
    raise TimeoutError(f"Timed out waiting for Ray nodes: {alive}/{expected_nodes} alive nodes")


def _build_train_cmd(root: Path, mode: str, env: dict[str, str]) -> list[str]:
    base_folder = _require_env(env, "BASE_FOLDER")
    gpus_per_node = _env(env, "GPUS_PER_NODE", "8")

    if mode == "production":
        model_size = "27B"
        data_file = _require_env(env, "DATA_FILE")
        actor_num_nodes = _env(env, "ACTOR_NUM_NODES", "4")
        rollout_num_gpus = _env(env, "ROLLOUT_NUM_GPUS", "8")
        launcher_args = [
            "--save-interval",
            _env(env, "SAVE_INTERVAL", "20"),
            "--prompt-data",
            data_file,
            "--rollout-shuffle",
            "--num-rollout",
            _env(env, "NUM_ROLLOUT", "3000"),
            "--rollout-batch-size",
            _env(env, "ROLLOUT_BATCH_SIZE", "8"),
            "--n-samples-per-prompt",
            _env(env, "N_SAMPLES_PER_PROMPT", "8"),
            "--rollout-max-response-len",
            _env(env, "MAX_RESPONSE_LEN", "32768"),
            "--rollout-temperature",
            _env(env, "TEMPERATURE", "1.0"),
            "--global-batch-size",
            _env(env, "GLOBAL_BATCH_SIZE", "64"),
            "--balance-data",
            "--tensor-model-parallel-size",
            _env(env, "TP_SIZE", "4"),
            "--pipeline-model-parallel-size",
            _env(env, "PP_SIZE", "2"),
            "--decoder-last-pipeline-num-layers",
            _env(env, "DECODER_LAST_PIPELINE_NUM_LAYERS", "30"),
            "--context-parallel-size",
            _env(env, "CP_SIZE", "4"),
            "--calculate-per-token-loss",
            "--max-tokens-per-gpu",
            _env(env, "MAX_TOKENS_PER_GPU", "8192"),
            "--rollout-num-gpus",
            rollout_num_gpus,
            "--rollout-num-gpus-per-engine",
            _env(env, "ROLLOUT_GPUS_PER_ENGINE", "2"),
            "--sglang-mem-fraction-static",
            _env(env, "SGLANG_MEM_FRACTION_STATIC", "0.75"),
        ]
    else:
        model_size = _env(env, "SMOKE_MODEL_SIZE", "9B")
        data_file = _require_env(env, "SMOKE_DATA_FILE")
        actor_num_nodes = _env(env, "ACTOR_NUM_NODES", "2")
        launcher_args = [
            "--save-interval",
            "1",
            "--prompt-data",
            data_file,
            "--num-rollout",
            _smoke_num_rollout(env),
            "--rollout-batch-size",
            _env(env, "ROLLOUT_BATCH_SIZE", "2"),
            "--n-samples-per-prompt",
            _env(env, "N_SAMPLES_PER_PROMPT", "2"),
            "--rollout-max-response-len",
            _env(env, "MAX_RESPONSE_LEN", "2048"),
            "--rollout-temperature",
            "1.0",
            "--global-batch-size",
            _env(env, "GLOBAL_BATCH_SIZE", "4"),
            "--tensor-model-parallel-size",
            _env(env, "TP_SIZE", "2"),
            "--pipeline-model-parallel-size",
            _env(env, "PP_SIZE", "1"),
            "--context-parallel-size",
            _env(env, "CP_SIZE", "1"),
            "--max-tokens-per-gpu",
            _env(env, "MAX_TOKENS_PER_GPU", "4096"),
            "--rollout-num-gpus",
            _env(env, "ROLLOUT_NUM_GPUS", "4"),
            "--rollout-num-gpus-per-engine",
            _env(env, "ROLLOUT_GPUS_PER_ENGINE", "1"),
            "--sglang-mem-fraction-static",
            _env(env, "SGLANG_MEM_FRACTION_STATIC", "0.6"),
            "--ci-test",
            "--ci-disable-kl-checker",
            "--start-rollout-id",
            "0",
            "--no-load-optim",
            "--no-load-rng",
            "--finetune",
            "--no-save-optim",
        ]

    return [
        "python3",
        str(root / "train.py"),
        "--actor-num-nodes",
        actor_num_nodes,
        "--actor-num-gpus-per-node",
        gpus_per_node,
        "--num-gpus-per-node",
        gpus_per_node,
        "--colocate",
        *_model_args(root, model_size),
        "--hf-checkpoint",
        f"{base_folder}/Qwen3.5-{model_size}",
        "--ref-load",
        f"{base_folder}/Qwen3.5-{model_size}_torch_dist",
        "--load",
        f"{base_folder}/Qwen3.5-{model_size}_slime",
        "--save",
        f"{base_folder}/Qwen3.5-{model_size}_slime",
        *_dataset_key_args(env),
        "--apply-chat-template",
        *launcher_args,
        *_optional_rollout_args(env),
        *_runtime_teacher_args(env),
        *_optional_wandb_args(env, mode),
        "--advantage-estimator",
        "grpo",
        "--use-opd",
        "--opd-type",
        "sglang",
        "--opd-kl-coef",
        _env(env, "OPD_KL_COEF", "1.0"),
        *_optional_opd_args(env),
        "--kl-loss-coef",
        "0.00",
        "--kl-loss-type",
        "low_var_kl",
        "--entropy-coef",
        "0.00",
        "--optimizer",
        "adam",
        "--lr",
        _env(env, "LR", "1e-6"),
        "--lr-decay-style",
        "constant",
        "--weight-decay",
        "0.1",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.98",
        "--sequence-parallel",
        "--use-dynamic-batch-size",
        "--attention-backend",
        "flash",
        "--loss-mask-type",
        "qwen3_5",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch Qwen3.5 OPD production or smoke jobs.")
    parser.add_argument("mode", choices=["production", "smoke"])
    parser.add_argument("--dry-run", action="store_true", help="Print Ray and training commands without running them.")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    require_hostfile = args.mode == "smoke"
    _start_ray(root, env, require_hostfile=require_hostfile, dry_run=args.dry_run)
    _wait_for_ray_nodes(env, dry_run=args.dry_run)
    train_cmd = _build_train_cmd(root, args.mode, env)
    log_path = _smoke_log_path(root, env) if args.mode == "smoke" else None
    _run(
        [
            "ray",
            "job",
            "submit",
            f"--address=http://127.0.0.1:{_env(env, 'RAY_DASHBOARD_PORT', '8265')}",
            f"--runtime-env-json={_runtime_env_json(env)}",
            "--",
            *train_cmd,
        ],
        args.dry_run,
        log_path=log_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
