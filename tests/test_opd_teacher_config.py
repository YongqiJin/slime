from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.rollout.on_policy_distillation import (
    load_teacher_runtime_config,
    reward_func,
    select_teacher_for_sample,
    select_teacher_url,
)
from slime.ray.rollout import compute_metrics_from_samples
from slime.utils.types import Sample


NUM_GPUS = 0


class _SuccessfulPostContext:
    def __init__(self, url: str):
        self.url = url

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        pass

    async def json(self):
        return {
            "meta_info": {
                "input_token_logprobs": [
                    (None,),
                    (-0.2,),
                ]
            }
        }


class _CapturingClientSession:
    requested_urls: list[str] = []

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json):
        self.requested_urls.append(url)
        return _SuccessfulPostContext(url)


def _write_config(path: Path, config: dict) -> str:
    path.write_text(json.dumps(config))
    return str(path)


def _valid_config() -> dict:
    return {
        "default_teacher": "general",
        "teachers": [
            {
                "name": "math",
                "domains": ["math", "gsm8k"],
                "urls": ["http://math-0/generate", "http://math-1/generate"],
                "metadata": {"model": "qwen3.5-27b"},
            },
            {
                "name": "general",
                "domains": ["general"],
                "urls": ["http://general/generate"],
            },
        ],
    }


def _args(**overrides):
    values = {
        "rm_url": "http://single-teacher/generate",
        "opd_teacher_config": None,
        "reward_key": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _metric_args():
    return types.SimpleNamespace(log_reward_category=None, advantage_estimator="ppo")


def _load_teacher_pool_module():
    module_path = (
        REPO_ROOT
        / "examples"
        / "on_policy_distillation"
        / "qwen3_5_multidomain"
        / "teacher_pool.py"
    )
    module_name = "test_qwen35_teacher_pool"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_qwen35_launch_module():
    module_path = (
        REPO_ROOT
        / "examples"
        / "on_policy_distillation"
        / "qwen3_5_multidomain"
        / "launch.py"
    )
    module_name = "test_qwen35_launch"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_load_teacher_runtime_config_parses_schema(tmp_path):
    config_path = _write_config(tmp_path / "teacher_runtime.json", _valid_config())

    config = load_teacher_runtime_config(config_path)

    assert config.default_teacher == "general"
    assert [teacher.name for teacher in config.teachers] == ["math", "general"]
    assert config.teachers[0].domains == ("math", "gsm8k")
    assert config.teachers[0].urls == ("http://math-0/generate", "http://math-1/generate")
    assert config.teachers[0].metadata == {"model": "qwen3.5-27b"}


@pytest.mark.unit
def test_select_teacher_precedence_teacher_name_then_domain_then_default(tmp_path):
    config = load_teacher_runtime_config(_write_config(tmp_path / "teacher_runtime.json", _valid_config()))

    by_name = select_teacher_for_sample(
        config,
        Sample(metadata={"teacher_model_name": "general", "domain": "math"}),
    )
    by_domain = select_teacher_for_sample(config, Sample(metadata={"domain": "math"}))
    by_default = select_teacher_for_sample(config, Sample(metadata={"domain": "unknown"}))

    assert by_name.name == "general"
    assert by_domain.name == "math"
    assert by_default.name == "general"


@pytest.mark.unit
def test_select_teacher_url_uses_config_and_round_robins_by_sample_index(tmp_path):
    config_path = _write_config(tmp_path / "teacher_runtime.json", _valid_config())

    url, teacher_name = select_teacher_url(_args(opd_teacher_config=config_path), Sample(index=3, metadata={"domain": "math"}))

    assert teacher_name == "math"
    assert url == "http://math-1/generate"


@pytest.mark.unit
def test_select_teacher_url_uses_first_url_when_sample_index_is_missing(tmp_path):
    config_path = _write_config(tmp_path / "teacher_runtime.json", _valid_config())

    url, teacher_name = select_teacher_url(_args(opd_teacher_config=config_path), Sample(index=None, metadata={"domain": "math"}))

    assert teacher_name == "math"
    assert url == "http://math-0/generate"


@pytest.mark.unit
def test_select_teacher_url_keeps_single_rm_url_when_config_is_not_set():
    url, teacher_name = select_teacher_url(_args(), Sample(metadata={"domain": "math"}))

    assert teacher_name is None
    assert url == "http://single-teacher/generate"


@pytest.mark.unit
def test_load_teacher_runtime_config_reloads_when_file_changes(tmp_path):
    config_path = Path(_write_config(tmp_path / "teacher_runtime.json", _valid_config()))
    first = load_teacher_runtime_config(str(config_path))

    updated = _valid_config()
    updated["teachers"][0]["urls"] = ["http://math-updated/generate"]
    config_path.write_text(json.dumps(updated))
    stat = config_path.stat()
    os.utime(config_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    second = load_teacher_runtime_config(str(config_path))

    assert first.teachers_by_name["math"].urls == ("http://math-0/generate", "http://math-1/generate")
    assert second.teachers_by_name["math"].urls == ("http://math-updated/generate",)


@pytest.mark.unit
def test_reward_func_posts_to_selected_teacher_config_url(monkeypatch, tmp_path):
    import slime.rollout.on_policy_distillation as opd

    _CapturingClientSession.requested_urls = []
    monkeypatch.setattr(opd.aiohttp, "ClientSession", _CapturingClientSession)
    config_path = _write_config(tmp_path / "teacher_runtime.json", _valid_config())
    sample = Sample(index=0, tokens=[1, 2], response_length=1, metadata={"domain": "math"})

    reward = asyncio.run(reward_func(_args(opd_teacher_config=config_path), sample))

    assert _CapturingClientSession.requested_urls == ["http://math-0/generate"]
    assert reward["meta_info"]["input_token_logprobs"] == [(None,), (-0.2,)]
    assert sample.metadata["opd_teacher_name"] == "math"
    assert sample.metadata["opd_teacher_url"] == "http://math-0/generate"


@pytest.mark.unit
def test_reward_func_replaces_non_dict_metadata_before_recording_selected_teacher(monkeypatch, tmp_path):
    import slime.rollout.on_policy_distillation as opd

    _CapturingClientSession.requested_urls = []
    monkeypatch.setattr(opd.aiohttp, "ClientSession", _CapturingClientSession)
    config_path = _write_config(tmp_path / "teacher_runtime.json", _valid_config())
    sample = Sample(index=0, tokens=[1, 2], response_length=1, metadata=None)

    asyncio.run(reward_func(_args(opd_teacher_config=config_path), sample))

    assert _CapturingClientSession.requested_urls == ["http://general/generate"]
    assert sample.metadata == {
        "opd_teacher_name": "general",
        "opd_teacher_url": "http://general/generate",
    }


@pytest.mark.unit
def test_opd_teacher_distribution_metrics_use_selected_teacher_metadata():
    samples = [
        Sample(index=0, response="ok", response_length=1, metadata={"opd_teacher_name": "math"}),
        Sample(index=1, response="ok", response_length=1, metadata={"opd_teacher_name": "math"}),
        Sample(index=2, response="ok", response_length=1, metadata={"opd_teacher_name": "code/domain"}),
        Sample(index=3, response="ok", response_length=1, metadata={}),
    ]

    metrics = compute_metrics_from_samples(_metric_args(), samples)

    assert metrics["opd_teacher/known_ratio"] == 0.75
    assert metrics["opd_teacher/name_math"] == pytest.approx(2 / 3)
    assert metrics["opd_teacher/name_code_domain"] == pytest.approx(1 / 3)


@pytest.mark.unit
def test_teacher_pool_generated_runtime_matches_loader_schema(tmp_path):
    teacher_pool = _load_teacher_pool_module()
    _, runtime = teacher_pool._teacher_entries(
        {
            "default_teacher": "general",
            "teachers": [
                {
                    "name": "math",
                    "domains": ["math"],
                    "model_path": "/tmp/math",
                    "instances": [{"host": "127.0.0.1", "port": 31001}],
                },
                {
                    "name": "general",
                    "domains": ["general"],
                    "model_path": "/tmp/general",
                    "instances": [{"host": "127.0.0.1", "port": 31002}],
                },
            ],
        },
        tmp_path,
    )
    config_path = tmp_path / "teacher_runtime.json"
    config_path.write_text(json.dumps(runtime))

    config = load_teacher_runtime_config(str(config_path))

    assert config.default_teacher == "general"
    assert config.teachers_by_name["math"].urls == ("http://127.0.0.1:31001/generate",)


@pytest.mark.unit
def test_teacher_pool_supports_local_and_multi_host_runtime(tmp_path):
    teacher_pool = _load_teacher_pool_module()
    entries, runtime = teacher_pool._teacher_entries(
        {
            "default_teacher": "general",
            "teachers": [
                {
                    "name": "general",
                    "domains": ["general"],
                    "model_path": "/tmp/general",
                    "instances": [{"host": "127.0.0.1", "port": 31001}],
                },
                {
                    "name": "code",
                    "domains": ["code"],
                    "model_path": "/tmp/code",
                    "instances": [
                        {"host": "10.0.0.12", "public_host": "10.0.0.12", "port": 31002, "tp": 2},
                        {"host": "10.0.0.13", "public_host": "teacher-code-b", "port": 31003, "tp": 2},
                    ],
                },
            ],
        },
        tmp_path,
    )
    config_path = tmp_path / "teacher_runtime.json"
    config_path.write_text(json.dumps(runtime))

    config = load_teacher_runtime_config(str(config_path))

    assert [entry["host"] for entry in entries] == ["127.0.0.1", "10.0.0.12", "10.0.0.13"]
    assert [entry["tp"] for entry in entries] == [1, 2, 2]
    assert config.teachers_by_name["code"].urls == (
        "http://10.0.0.12:31002/generate",
        "http://teacher-code-b:31003/generate",
    )


@pytest.mark.unit
def test_teacher_pool_dispatches_local_and_remote_commands(monkeypatch):
    teacher_pool = _load_teacher_pool_module()
    calls = []

    monkeypatch.setattr(teacher_pool.subprocess, "run", lambda command, check: calls.append((command, check)))

    teacher_pool._run_on_host("127.0.0.1", "echo local", ssh_user="root", dry_run=False)
    teacher_pool._run_on_host("10.0.0.12", "echo remote", ssh_user="root", dry_run=False)

    assert calls == [
        (["bash", "-lc", "echo local"], True),
        (["ssh", "root@10.0.0.12", "echo remote"], True),
    ]


@pytest.mark.unit
def test_teacher_pool_launch_command_detaches_from_calling_session(tmp_path):
    teacher_pool = _load_teacher_pool_module()
    entries, _ = teacher_pool._teacher_entries(
        {
            "teachers": [
                {
                    "name": "math",
                    "model_path": "/tmp/math",
                    "instances": [{"host": "127.0.0.1", "port": 31001, "gpus": "0,1", "tp": 2}],
                }
            ],
        },
        tmp_path,
    )

    command = teacher_pool._launch_command(entries[0], {"python": "python3", "sglang_module": "sglang.launch_server"})

    assert "CUDA_VISIBLE_DEVICES=0,1 nohup setsid python3 -m sglang.launch_server" in command
    assert "> " in command
    assert "2>&1 < /dev/null & echo $!" in command


@pytest.mark.unit
def test_teacher_pool_refuses_to_write_runtime_when_post_health_check_fails(monkeypatch, tmp_path):
    teacher_pool = _load_teacher_pool_module()
    run_dir = tmp_path / "run"
    config_path = tmp_path / "teachers.yaml"
    config_path.write_text(
        json.dumps(
            {
                "run": {"post_health_stability_seconds": 1},
                "teachers": [
                    {
                        "name": "math",
                        "model_path": "/tmp/math",
                        "instances": [{"host": "127.0.0.1", "port": 31001}],
                    }
                ],
            }
        )
    )

    monkeypatch.setattr(teacher_pool, "_run_on_host", lambda *args, **kwargs: None)
    monkeypatch.setattr(teacher_pool, "_wait_for_health", lambda *args, **kwargs: None)

    def fail_stability(*args, **kwargs):
        raise RuntimeError("dead teacher")

    monkeypatch.setattr(teacher_pool, "_wait_for_post_health_stability", fail_stability)

    with pytest.raises(RuntimeError, match="dead teacher"):
        teacher_pool.start(types.SimpleNamespace(config=str(config_path), run_dir=str(run_dir), dry_run=False))

    assert not (run_dir / teacher_pool.RUNTIME_FILE).exists()
    assert not (run_dir / teacher_pool.PROCESS_FILE).exists()


@pytest.mark.unit
def test_teacher_pool_stop_reuses_start_ssh_user_from_process_file(monkeypatch, tmp_path):
    teacher_pool = _load_teacher_pool_module()
    calls = []
    run_dir = tmp_path / "teachers"
    run_dir.mkdir()
    pid_path = run_dir / "remote.pid"
    process_path = run_dir / teacher_pool.PROCESS_FILE
    process_path.write_text(
        json.dumps(
            {
                "ssh_user": "root",
                "processes": [
                    {
                        "host": "10.0.0.12",
                        "pid_path": str(pid_path),
                    }
                ],
            }
        )
    )

    monkeypatch.setattr(teacher_pool.subprocess, "run", lambda command, check: calls.append((command, check)))

    teacher_pool.stop(types.SimpleNamespace(run_dir=str(run_dir), ssh_user=None, dry_run=False))

    assert calls == [
        (
            [
                "ssh",
                "root@10.0.0.12",
                f"if [ -f {pid_path} ]; then kill $(cat {pid_path}) 2>/dev/null || true; rm -f {pid_path}; fi",
            ],
            True,
        )
    ]


@pytest.mark.unit
def test_stop_teachers_wrapper_accepts_dry_run_without_explicit_run_dir(tmp_path):
    run_dir = tmp_path / "teachers"
    run_dir.mkdir()
    (run_dir / "teacher_processes.json").write_text('{"processes": []}\n')
    script = REPO_ROOT / "examples" / "on_policy_distillation" / "qwen3_5_multidomain" / "stop_teachers.sh"

    env = {**os.environ, "OPD_TEACHER_RUN_DIR": str(run_dir)}
    subprocess.run(["bash", str(script), "--dry-run"], check=True, env=env)
    subprocess.run(["bash", str(script), str(run_dir), "--dry-run"], check=True)


def _make_qwen35_production_preflight_files(tmp_path: Path):
    base = tmp_path / "base"
    for name in ["Qwen3.5-27B", "Qwen3.5-27B_torch_dist"]:
        (base / name).mkdir(parents=True)
    data_file = tmp_path / "train.parquet"
    data_file.write_text("")
    return base, data_file


def _make_qwen35_smoke_preflight_files(tmp_path: Path):
    base = tmp_path / "base"
    for name in ["Qwen3.5-9B", "Qwen3.5-9B_torch_dist"]:
        (base / name).mkdir(parents=True)
    data_file = tmp_path / "smoke.parquet"
    data_file.write_text("")
    return base, data_file


@pytest.mark.unit
def test_preflight_allows_teacher_pool_before_runtime_config_exists(tmp_path):
    base, data_file = _make_qwen35_production_preflight_files(tmp_path)
    run_dir = tmp_path / "teacher-run"
    script = REPO_ROOT / "examples" / "on_policy_distillation" / "qwen3_5_multidomain" / "preflight.sh"

    env = {
        **os.environ,
        "BASE_FOLDER": str(base),
        "DATA_FILE": str(data_file),
        "MASTER_ADDR": "127.0.0.1",
        "OPD_TEACHER_RUN_DIR": str(run_dir),
    }
    result = subprocess.run(["bash", str(script), "production"], check=True, env=env, text=True, capture_output=True)

    assert "teacher_runtime_config=not yet generated" in result.stderr
    assert "slime_checkpoint=will_initialize_from_ref_load_and_save" in result.stdout
    assert "preflight_ok mode=production" in result.stdout


@pytest.mark.unit
def test_preflight_reports_existing_slime_checkpoint(tmp_path):
    base, data_file = _make_qwen35_production_preflight_files(tmp_path)
    slime_dir = base / "Qwen3.5-27B_slime"
    slime_dir.mkdir()
    (slime_dir / "latest_checkpointed_iteration.txt").write_text("release")
    script = REPO_ROOT / "examples" / "on_policy_distillation" / "qwen3_5_multidomain" / "preflight.sh"

    env = {
        **os.environ,
        "BASE_FOLDER": str(base),
        "DATA_FILE": str(data_file),
        "MASTER_ADDR": "127.0.0.1",
        "TEACHER_RM_URL": "http://127.0.0.1:31001/generate",
    }
    result = subprocess.run(["bash", str(script), "production"], check=True, env=env, text=True, capture_output=True)

    assert f"slime_checkpoint=existing:{slime_dir}" in result.stdout
    assert "preflight_ok mode=production" in result.stdout


@pytest.mark.unit
def test_preflight_single_teacher_rm_url_does_not_require_teacher_source_config(tmp_path):
    base, data_file = _make_qwen35_production_preflight_files(tmp_path)
    script = REPO_ROOT / "examples" / "on_policy_distillation" / "qwen3_5_multidomain" / "preflight.sh"

    env = {
        **os.environ,
        "BASE_FOLDER": str(base),
        "DATA_FILE": str(data_file),
        "MASTER_ADDR": "127.0.0.1",
        "TEACHER_RM_URL": "http://127.0.0.1:31001/generate",
        "TEACHERS_CONFIG": str(tmp_path / "missing-teachers.yaml"),
    }
    result = subprocess.run(["bash", str(script), "production"], check=True, env=env, text=True, capture_output=True)

    assert "teacher_source_config=skipped_missing_for_single_teacher_rm_url" in result.stdout
    assert "teacher_rm_url=http://127.0.0.1:31001/generate" in result.stdout
    assert "preflight_ok mode=production" in result.stdout


@pytest.mark.unit
def test_smoke_preflight_requires_hostfile_worker_distinct_from_master(tmp_path):
    base, data_file = _make_qwen35_smoke_preflight_files(tmp_path)
    hostfile = tmp_path / "hostfile"
    hostfile.write_text("127.0.0.1\n")
    script = REPO_ROOT / "examples" / "on_policy_distillation" / "qwen3_5_multidomain" / "preflight.sh"

    env = {
        **os.environ,
        "BASE_FOLDER": str(base),
        "SMOKE_DATA_FILE": str(data_file),
        "MASTER_ADDR": "127.0.0.1",
        "HOSTFILE": str(hostfile),
        "TEACHER_RM_URL": "http://127.0.0.1:31001/generate",
    }
    result = subprocess.run(["bash", str(script), "smoke"], env=env, text=True, capture_output=True)

    assert result.returncode != 0
    assert "worker IP different from MASTER_ADDR" in result.stderr


@pytest.mark.unit
def test_smoke_preflight_accepts_worker_only_hostfile(tmp_path):
    base, data_file = _make_qwen35_smoke_preflight_files(tmp_path)
    hostfile = tmp_path / "hostfile"
    hostfile.write_text("10.0.0.2 slots=8\n")
    script = REPO_ROOT / "examples" / "on_policy_distillation" / "qwen3_5_multidomain" / "preflight.sh"

    env = {
        **os.environ,
        "BASE_FOLDER": str(base),
        "SMOKE_DATA_FILE": str(data_file),
        "MASTER_ADDR": "10.0.0.1",
        "HOSTFILE": str(hostfile),
        "TEACHER_RM_URL": "http://127.0.0.1:31001/generate",
    }
    result = subprocess.run(["bash", str(script), "smoke"], check=True, env=env, text=True, capture_output=True)

    assert "slime_checkpoint=will_initialize_from_ref_load_and_save" in result.stdout
    assert "preflight_ok mode=smoke" in result.stdout


@pytest.mark.unit
def test_qwen35_train_entry_scripts_stay_short():
    example_dir = REPO_ROOT / "examples" / "on_policy_distillation" / "qwen3_5_multidomain"

    for name in ["train_27b.sh", "smoke_2node.sh"]:
        lines = (example_dir / name).read_text().splitlines()
        assert len(lines) <= 10
        assert "launch.py" in lines[-1]


@pytest.mark.unit
def test_qwen35_launcher_builds_production_and_smoke_commands(tmp_path):
    launch = _load_qwen35_launch_module()
    base_env = {
        "BASE_FOLDER": "/models",
        "MASTER_ADDR": "10.0.0.1",
        "TEACHER_RM_URL": "http://teacher/generate",
    }

    production_cmd = launch._build_train_cmd(
        REPO_ROOT,
        "production",
        {
            **base_env,
            "DATA_FILE": "/data/train.parquet",
        },
    )
    smoke_cmd = launch._build_train_cmd(
        REPO_ROOT,
        "smoke",
        {
            **base_env,
            "SMOKE_DATA_FILE": "/data/smoke.parquet",
            "SMOKE_MODEL_SIZE": "9B",
            "SGLANG_ROUTER_PORT": "39817",
        },
    )

    assert "--ci-test" not in production_cmd
    assert "--ci-test" in smoke_cmd
    assert "--ci-disable-kl-checker" not in production_cmd
    assert "--ci-disable-kl-checker" in smoke_cmd
    assert "--no-load-optim" not in production_cmd
    assert "--no-load-optim" in smoke_cmd
    assert "--no-save-optim" in smoke_cmd
    assert int(smoke_cmd[smoke_cmd.index("--num-rollout") + 1]) >= 3
    assert smoke_cmd[smoke_cmd.index("--sglang-router-port") + 1] == "39817"
    assert "/models/Qwen3.5-27B" in production_cmd
    assert "/models/Qwen3.5-9B" in smoke_cmd
    assert "--spec" in production_cmd
    for cmd in [production_cmd, smoke_cmd]:
        assert cmd[cmd.index("--actor-num-gpus-per-node") + 1] == "8"
        assert cmd[cmd.index("--num-gpus-per-node") + 1] == "8"
    assert all("Welcome to PAI DSW" not in arg for arg in production_cmd + smoke_cmd)
    assert all("____" not in arg for arg in production_cmd + smoke_cmd)
    assert "http://teacher/generate" in production_cmd
    assert "http://teacher/generate" in smoke_cmd


@pytest.mark.unit
def test_qwen35_launcher_can_omit_label_key_for_unlabeled_prompt_data(tmp_path):
    launch = _load_qwen35_launch_module()

    smoke_cmd = launch._build_train_cmd(
        REPO_ROOT,
        "smoke",
        {
            "BASE_FOLDER": "/models",
            "MASTER_ADDR": "10.0.0.1",
            "TEACHER_RM_URL": "http://teacher/generate",
            "SMOKE_DATA_FILE": "/data/smoke.parquet",
            "LABEL_KEY": "",
        },
    )

    assert "--input-key" in smoke_cmd
    assert "--label-key" not in smoke_cmd


@pytest.mark.unit
def test_qwen35_smoke_rejects_less_than_three_rollouts(tmp_path):
    launch = _load_qwen35_launch_module()

    with pytest.raises(SystemExit, match="NUM_ROLLOUT >= 3"):
        launch._build_train_cmd(
            REPO_ROOT,
            "smoke",
            {
                "BASE_FOLDER": "/models",
                "MASTER_ADDR": "10.0.0.1",
                "TEACHER_RM_URL": "http://teacher/generate",
                "SMOKE_DATA_FILE": "/data/smoke.parquet",
                "NUM_ROLLOUT": "2",
            },
        )


@pytest.mark.unit
def test_qwen35_shell_entrypoints_support_dry_run(tmp_path):
    hostfile = tmp_path / "hostfile"
    hostfile.write_text("10.0.0.2 slots=8\n")
    example_dir = REPO_ROOT / "examples" / "on_policy_distillation" / "qwen3_5_multidomain"
    env = {
        **os.environ,
        "BASE_FOLDER": "/models",
        "DATA_FILE": "/data/train.parquet",
        "SMOKE_DATA_FILE": "/data/smoke.parquet",
        "MASTER_ADDR": "10.0.0.1",
        "HOSTFILE": str(hostfile),
        "TEACHER_RM_URL": "http://teacher/generate",
    }

    production = subprocess.run(
        ["bash", str(example_dir / "train_27b.sh"), "--dry-run"],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    smoke = subprocess.run(
        ["bash", str(example_dir / "smoke_2node.sh"), "--dry-run"],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    assert "ray job submit" in production.stdout
    assert "Qwen3.5-27B" in production.stdout
    assert "ray job submit" in smoke.stdout
    assert "Qwen3.5-9B" in smoke.stdout
    assert "--ci-test" in smoke.stdout
    assert "--ci-disable-kl-checker" in smoke.stdout


@pytest.mark.unit
def test_qwen35_launcher_can_skip_worker_ssh_for_prestarted_ray(tmp_path, capsys):
    launch = _load_qwen35_launch_module()
    hostfile = tmp_path / "hostfile"
    hostfile.write_text("10.0.0.2 slots=8\n")

    launch._start_ray(
        REPO_ROOT,
        {
            "MASTER_ADDR": "10.0.0.1",
            "HOSTFILE": str(hostfile),
            "SLIME_RAY_WORKERS_PRESTARTED": "1",
        },
        require_hostfile=True,
        dry_run=True,
    )

    output = capsys.readouterr().out
    assert "ray start --head" in output
    assert "Skipping Ray worker SSH startup" in output
    assert "ssh root@10.0.0.2" not in output


@pytest.mark.unit
def test_qwen35_launcher_waits_for_expected_ray_nodes(monkeypatch, capsys):
    launch = _load_qwen35_launch_module()
    fake_ray = types.ModuleType("ray")
    fake_ray.init = lambda **kwargs: None
    fake_ray.nodes = lambda: [{"Alive": True}, {"Alive": True}]
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    launch._wait_for_ray_nodes({"SLIME_RAY_EXPECTED_NODES": "2"}, dry_run=False)

    assert "Ray cluster ready: 2/2 alive nodes" in capsys.readouterr().out


@pytest.mark.unit
def test_qwen35_launcher_ray_node_wait_is_visible_in_dry_run(capsys):
    launch = _load_qwen35_launch_module()

    launch._wait_for_ray_nodes({"SLIME_RAY_EXPECTED_NODES": "2"}, dry_run=True)

    assert "Waiting for Ray nodes: expected=2" in capsys.readouterr().out


@pytest.mark.unit
def test_teacher_pool_rejects_multiteacher_runtime_without_default(tmp_path):
    teacher_pool = _load_teacher_pool_module()

    with pytest.raises(ValueError, match="multiple teachers must define default_teacher"):
        teacher_pool._teacher_entries(
            {
                "teachers": [
                    {
                        "name": "math",
                        "model_path": "/tmp/math",
                        "instances": [{"host": "127.0.0.1", "port": 31001}],
                    },
                    {
                        "name": "general",
                        "model_path": "/tmp/general",
                        "instances": [{"host": "127.0.0.1", "port": 31002}],
                    },
                ],
            },
            tmp_path,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "config,match",
    [
        ({}, "non-empty 'teachers' list"),
        ({"teachers": [{"name": "math", "urls": []}]}, "non-empty string 'urls' list"),
        (
            {"teachers": [{"name": "math", "urls": ["http://math"]}, {"name": "math", "urls": ["http://math-2"]}]},
            "Duplicate OPD teacher name",
        ),
        (
            {"default_teacher": "missing", "teachers": [{"name": "math", "urls": ["http://math"]}]},
            "default_teacher",
        ),
        (
            {
                "teachers": [
                    {"name": "math", "urls": ["http://math"]},
                    {"name": "general", "urls": ["http://general"]},
                ]
            },
            "multiple teachers must define 'default_teacher'",
        ),
    ],
)
def test_invalid_teacher_runtime_config_fails_with_clear_error(tmp_path, config, match):
    config_path = _write_config(tmp_path / "teacher_runtime.json", config)

    with pytest.raises(ValueError, match=match):
        load_teacher_runtime_config(config_path)


@pytest.mark.unit
def test_unknown_explicit_teacher_name_fails(tmp_path):
    config = load_teacher_runtime_config(_write_config(tmp_path / "teacher_runtime.json", _valid_config()))

    with pytest.raises(ValueError, match="unknown OPD teacher_model_name"):
        select_teacher_for_sample(config, Sample(metadata={"teacher_model_name": "missing"}))


def _load_slime_arguments_module(monkeypatch):
    router_pkg_mod = types.ModuleType("sglang_router")
    router_launch_mod = types.ModuleType("sglang_router.launch_router")
    sglang_arguments_mod = types.ModuleType("slime.backends.sglang_utils.arguments")
    sglang_external_mod = types.ModuleType("slime.backends.sglang_utils.external")
    logging_utils_mod = types.ModuleType("slime.utils.logging_utils")

    router_launch_mod.RouterArgs = object
    sglang_arguments_mod.sglang_parse_args = lambda *args, **kwargs: None
    sglang_arguments_mod.validate_args = lambda args: args
    sglang_external_mod.apply_external_engine_info_to_args = lambda *args, **kwargs: None
    logging_utils_mod.configure_logger = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "sglang_router", router_pkg_mod)
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", router_launch_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.arguments", sglang_arguments_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.external", sglang_external_mod)
    monkeypatch.setitem(sys.modules, "slime.utils.logging_utils", logging_utils_mod)

    import importlib.util

    module_path = REPO_ROOT / "slime" / "utils" / "arguments.py"
    module_name = "test_opd_teacher_config_arguments_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _minimal_validate_args(**overrides):
    values = dict(
        eval_config=None,
        eval_prompt_data=None,
        use_slime_router=False,
        kl_coef=0,
        use_kl_loss=False,
        ref_load="/tmp/ref",
        use_opd=True,
        opd_type="sglang",
        opd_teacher_load=None,
        opd_teacher_config=None,
        rm_url="http://single-teacher/generate",
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


@pytest.mark.unit
def test_opd_teacher_config_rejected_without_use_opd(monkeypatch):
    module = _load_slime_arguments_module(monkeypatch)
    args = _minimal_validate_args(use_opd=False, opd_teacher_config="/tmp/teachers.json")

    with pytest.raises(ValueError, match="opd-teacher-config.*use-opd"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_opd_teacher_config_rejected_for_megatron_opd(monkeypatch):
    module = _load_slime_arguments_module(monkeypatch)
    args = _minimal_validate_args(opd_type="megatron", opd_teacher_config="/tmp/teachers.json")

    with pytest.raises(ValueError, match="only supported when --opd-type=sglang"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_missing_opd_teacher_config_path_fails_for_sglang_opd(monkeypatch):
    module = _load_slime_arguments_module(monkeypatch)
    args = _minimal_validate_args(opd_teacher_config="/tmp/missing-teachers.json")

    with pytest.raises(FileNotFoundError, match="opd_teacher_config"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_sglang_opd_requires_teacher_config_or_rm_url(monkeypatch):
    module = _load_slime_arguments_module(monkeypatch)
    args = _minimal_validate_args(opd_teacher_config=None, rm_url=None)

    with pytest.raises(ValueError, match="opd-teacher-config.*rm-url"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_sglang_opd_teacher_config_does_not_require_rm_url(monkeypatch, tmp_path):
    module = _load_slime_arguments_module(monkeypatch)
    config_path = tmp_path / "teachers.json"
    config_path.write_text('{"default_teacher":"default","teachers":[{"name":"default","urls":["http://teacher"]}]}\n')
    args = _minimal_validate_args(opd_teacher_config=str(config_path), rm_url=None)

    module._validate_opd_args(args)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
