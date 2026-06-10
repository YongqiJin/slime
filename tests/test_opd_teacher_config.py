from __future__ import annotations

import asyncio
import json
import os
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
        return {"meta_info": {"input_token_logprobs": [(None,), (-0.2,)]}}


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

    by_name = select_teacher_for_sample(config, Sample(metadata={"teacher_model_name": "general", "domain": "math"}))
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
    config_path.write_text(
        '{"default_teacher":"default","teachers":[{"name":"default","urls":["http://teacher"]}]}\n'
    )
    args = _minimal_validate_args(opd_teacher_config=str(config_path), rm_url=None)

    module._validate_opd_args(args)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
