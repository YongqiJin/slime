import logging

import aiohttp
import torch

from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample


logger = logging.getLogger(__name__)

OPD_TEACHER_LOGPROB_FAILURE_SENTINEL = -1e9


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


async def reward_func(args, sample, **kwargs):
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
            async with session.post(args.rm_url, json=payload) as resp:
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
