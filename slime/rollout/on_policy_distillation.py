import json
import logging
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import torch

from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample


logger = logging.getLogger(__name__)

OPD_TEACHER_LOGPROB_FAILURE_SENTINEL = -1e9


@dataclass(frozen=True)
class OpdTeacher:
    name: str
    urls: tuple[str, ...]
    domains: tuple[str, ...] = ()
    metadata: dict | None = None


@dataclass(frozen=True)
class OpdTeacherRuntimeConfig:
    teachers: tuple[OpdTeacher, ...]
    default_teacher: str | None = None

    @property
    def teachers_by_name(self) -> dict[str, OpdTeacher]:
        return {teacher.name: teacher for teacher in self.teachers}


_TEACHER_CONFIG_CACHE: dict[str, tuple[int, OpdTeacherRuntimeConfig]] = {}


def _failed_teacher_logprob_payload(response_length: int, error: str | None = None):
    payload = {
        "meta_info": {
            "input_token_logprobs": [(None,)]
            + [(OPD_TEACHER_LOGPROB_FAILURE_SENTINEL,) for _ in range(response_length)]
        },
        "opd_teacher_logprob_failed": True,
    }
    if error is not None:
        payload["opd_teacher_logprob_error"] = error
    return payload


def _normalize_urls(raw_teacher: dict) -> tuple[str, ...]:
    urls = raw_teacher.get("urls")
    if urls is None and raw_teacher.get("url") is not None:
        urls = [raw_teacher["url"]]
    if not isinstance(urls, list) or not urls or not all(isinstance(url, str) and url for url in urls):
        raise ValueError(f"OPD teacher {raw_teacher.get('name')!r} must define a non-empty string 'urls' list.")
    return tuple(urls)


def load_teacher_runtime_config(path: str) -> OpdTeacherRuntimeConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"OPD teacher config does not exist: {path}")

    cache_key = str(config_path.resolve())
    mtime = config_path.stat().st_mtime_ns
    cached = _TEACHER_CONFIG_CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    with config_path.open() as f:
        raw_config = json.load(f)

    raw_teachers = raw_config.get("teachers") if isinstance(raw_config, dict) else None
    if not isinstance(raw_teachers, list) or not raw_teachers:
        raise ValueError("OPD teacher config must contain a non-empty 'teachers' list.")

    teachers = []
    seen_names = set()
    for raw_teacher in raw_teachers:
        if not isinstance(raw_teacher, dict):
            raise ValueError("Each OPD teacher entry must be an object.")
        name = raw_teacher.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Each OPD teacher entry must define a non-empty string 'name'.")
        if name in seen_names:
            raise ValueError(f"Duplicate OPD teacher name: {name}")
        seen_names.add(name)

        raw_domains = raw_teacher.get("domains", [])
        if not isinstance(raw_domains, list) or not all(isinstance(domain, str) for domain in raw_domains):
            raise ValueError(f"OPD teacher {name!r} field 'domains' must be a list of strings.")

        metadata = raw_teacher.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(f"OPD teacher {name!r} field 'metadata' must be an object when set.")

        teachers.append(
            OpdTeacher(
                name=name,
                urls=_normalize_urls(raw_teacher),
                domains=tuple(raw_domains),
                metadata=metadata,
            )
        )

    default_teacher = raw_config.get("default_teacher")
    if default_teacher is not None and default_teacher not in seen_names:
        raise ValueError(f"OPD default_teacher {default_teacher!r} is not defined in teachers.")
    if default_teacher is None and len(teachers) > 1:
        raise ValueError("OPD teacher config with multiple teachers must define 'default_teacher'.")

    config = OpdTeacherRuntimeConfig(teachers=tuple(teachers), default_teacher=default_teacher)
    _TEACHER_CONFIG_CACHE[cache_key] = (mtime, config)
    logger.info(
        "Loaded OPD teacher config %s with teachers=%s",
        path,
        {teacher.name: len(teacher.urls) for teacher in teachers},
    )
    return config


def select_teacher_for_sample(config: OpdTeacherRuntimeConfig, sample: Sample) -> OpdTeacher:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    teachers_by_name = config.teachers_by_name

    teacher_model_name = metadata.get("teacher_model_name")
    if teacher_model_name is not None:
        if teacher_model_name not in teachers_by_name:
            raise ValueError(f"Sample requested unknown OPD teacher_model_name {teacher_model_name!r}.")
        return teachers_by_name[teacher_model_name]

    domain = metadata.get("domain")
    if domain is not None:
        for teacher in config.teachers:
            if domain in teacher.domains:
                return teacher

    if config.default_teacher is not None:
        return teachers_by_name[config.default_teacher]

    return config.teachers[0]


def select_teacher_url(args, sample: Sample) -> tuple[str, str | None]:
    config_path = getattr(args, "opd_teacher_config", None)
    if config_path is None:
        return args.rm_url, None

    config = load_teacher_runtime_config(config_path)
    teacher = select_teacher_for_sample(config, sample)
    url_index = sample.index % len(teacher.urls) if sample.index is not None else 0
    return teacher.urls[url_index], teacher.name


async def reward_func(args, sample, **kwargs):
    teacher_url, teacher_name = select_teacher_url(args, sample)
    if teacher_name is not None:
        if not isinstance(sample.metadata, dict):
            sample.metadata = {}
        sample.metadata["opd_teacher_name"] = teacher_name
        sample.metadata["opd_teacher_url"] = teacher_url

    payload = {
        # "text": sample.prompt + sample.response,
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    session_kwargs = {}
    try:
        async with aiohttp.ClientSession(**session_kwargs) as session:
            async with session.post(teacher_url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()
    except Exception as exc:
        logger.warning("OPD teacher logprob request failed; filling sentinel logprobs: %s", exc)
        return _failed_teacher_logprob_payload(sample.response_length, error=repr(exc))


def _extract_teacher_log_probs(reward, response_length: int) -> tuple[torch.Tensor, bool]:
    try:
        input_token_logprobs = reward["meta_info"]["input_token_logprobs"]
        logprobs = [item[0] for item in input_token_logprobs[1:]]
        if len(logprobs) < response_length:
            raise ValueError(
                f"teacher logprobs length {len(logprobs)} is shorter than response_length {response_length}"
            )
        if response_length == 0:
            return torch.empty((0,), dtype=torch.float32), bool(reward.get("opd_teacher_logprob_failed", False))
        teacher_log_probs = torch.tensor(logprobs, dtype=torch.float32)[-response_length:]
        return teacher_log_probs, bool(reward.get("opd_teacher_logprob_failed", False))
    except Exception as exc:
        logger.warning("Failed to parse OPD teacher logprobs; filling sentinel logprobs: %s", exc)
        return (
            torch.full((response_length,), OPD_TEACHER_LOGPROB_FAILURE_SENTINEL, dtype=torch.float32),
            True,
        )


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    Note: The reward_func calls the teacher server which returns token-level log-probs.
    For pure on-policy distillation without task rewards, we return 0.0 for each sample.
    The actual learning signal comes from the OPD KL penalty applied in compute_advantages_and_returns.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    teacher_log_probs = []
    teacher_failures = []
    failed_samples = 0
    failed_tokens = 0
    for reward, response_length in zip(raw_rewards, response_lengths, strict=False):
        t_log_probs, failed = _extract_teacher_log_probs(reward, response_length)
        teacher_log_probs.append(t_log_probs)
        teacher_failures.append(failed)
        if failed:
            failed_samples += 1
            failed_tokens += response_length

    for sample, t_log_probs, failed in zip(samples, teacher_log_probs, teacher_failures, strict=False):
        sample.teacher_log_probs = t_log_probs
        if failed or torch.eq(t_log_probs, OPD_TEACHER_LOGPROB_FAILURE_SENTINEL).any().item():
            sample.metadata["opd_teacher_logprob_failed"] = True

    if failed_samples:
        logger.warning("OPD teacher logprob failures: samples=%s tokens=%s", failed_samples, failed_tokens)

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards
