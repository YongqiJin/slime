from __future__ import annotations

import asyncio
import sys
from argparse import Namespace
from pathlib import Path

import _cp_dist_helpers  # noqa: F401
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.backends.megatron_utils.loss import apply_opd_kl_to_advantages
from slime.rollout.on_policy_distillation import (
    OPD_TEACHER_LOGPROB_FAILURE_SENTINEL,
    post_process_rewards,
    reward_func,
)
from slime.utils.types import Sample


NUM_GPUS = 0


class _FailingPostContext:
    async def __aenter__(self):
        raise TimeoutError("teacher timed out")

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FailingClientSession:
    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json):
        return _FailingPostContext()


def _make_args(**overrides):
    values = {"reward_key": None, "rm_url": "http://teacher.invalid/generate"}
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.unit
def test_reward_func_returns_sentinel_payload_when_teacher_request_fails(monkeypatch):
    import slime.rollout.on_policy_distillation as opd

    monkeypatch.setattr(opd.aiohttp, "ClientSession", _FailingClientSession)
    sample = Sample(tokens=[1, 2, 3], response_length=2)

    reward = asyncio.run(reward_func(_make_args(), sample))

    assert reward["opd_teacher_logprob_failed"] is True
    assert reward["meta_info"]["input_token_logprobs"] == [
        (None,),
        (OPD_TEACHER_LOGPROB_FAILURE_SENTINEL,),
        (OPD_TEACHER_LOGPROB_FAILURE_SENTINEL,),
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_reward",
    [
        {},
        {"meta_info": {}},
        {"meta_info": {"input_token_logprobs": []}},
        {"meta_info": {"input_token_logprobs": [(None,), (-0.1,)]}},
    ],
)
def test_post_process_rewards_fills_sentinel_for_malformed_or_short_teacher_payloads(bad_reward):
    sample = Sample(response_length=3, reward=bad_reward)

    rewards, scores = post_process_rewards(_make_args(), [sample])

    expected = torch.full((3,), OPD_TEACHER_LOGPROB_FAILURE_SENTINEL, dtype=torch.float32)
    assert rewards == [0.0]
    assert scores == [0.0]
    assert torch.equal(sample.teacher_log_probs, expected)
    assert sample.metadata["opd_teacher_logprob_failed"] is True


@pytest.mark.unit
def test_post_process_rewards_marks_zero_length_teacher_failure(caplog):
    sample = Sample(
        response_length=0,
        reward={
            "meta_info": {"input_token_logprobs": [(None,)]},
            "opd_teacher_logprob_failed": True,
        },
    )

    rewards, scores = post_process_rewards(_make_args(), [sample])

    assert rewards == [0.0]
    assert scores == [0.0]
    assert sample.teacher_log_probs.shape == (0,)
    assert sample.metadata["opd_teacher_logprob_failed"] is True
    assert "samples=1 tokens=0" in caplog.text


@pytest.mark.unit
def test_post_process_rewards_keeps_successful_teacher_payload_behavior():
    sample = Sample(
        response_length=2,
        reward={
            "meta_info": {
                "input_token_logprobs": [
                    (None,),
                    (-9.0,),
                    (-8.0,),
                    (-0.3,),
                    (-0.2,),
                ]
            }
        },
    )

    post_process_rewards(_make_args(), [sample])

    assert torch.equal(sample.teacher_log_probs, torch.tensor([-0.3, -0.2], dtype=torch.float32))
    assert "opd_teacher_logprob_failed" not in sample.metadata


@pytest.mark.unit
def test_apply_opd_kl_masks_teacher_failure_sentinel_tokens():
    args = Namespace(opd_type="sglang", opd_kl_coef=0.5)
    rollout_data = {
        "teacher_log_probs": [
            torch.tensor(
                [-2.0, OPD_TEACHER_LOGPROB_FAILURE_SENTINEL, -4.0],
                dtype=torch.float32,
            )
        ]
    }
    advantages = [torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)]
    student_log_probs = [torch.tensor([-1.0, -123.0, -3.0], dtype=torch.float32)]

    apply_opd_kl_to_advantages(args, rollout_data, advantages, student_log_probs)

    assert torch.equal(rollout_data["opd_reverse_kl"][0], torch.tensor([1.0, 0.0, 1.0]))
    assert torch.equal(advantages[0], torch.tensor([0.5, 1.0, 0.5]))


@pytest.mark.unit
def test_apply_opd_kl_rejects_mismatched_sample_counts():
    args = Namespace(opd_type="sglang", opd_kl_coef=0.5)
    rollout_data = {"teacher_log_probs": [torch.tensor([-1.0])]}
    advantages = [torch.tensor([0.0]), torch.tensor([0.0])]
    student_log_probs = [torch.tensor([-1.0]), torch.tensor([-1.0])]

    with pytest.raises(ValueError, match="same number of samples"):
        apply_opd_kl_to_advantages(args, rollout_data, advantages, student_log_probs)


@pytest.mark.unit
def test_apply_opd_kl_rejects_mismatched_tensor_shapes():
    args = Namespace(opd_type="sglang", opd_kl_coef=0.5)
    rollout_data = {"teacher_log_probs": [torch.tensor([-1.0])]}
    advantages = [torch.tensor([0.0, 0.0])]
    student_log_probs = [torch.tensor([-1.0, -2.0])]

    with pytest.raises(ValueError, match="matching tensor shapes.*sample 0"):
        apply_opd_kl_to_advantages(args, rollout_data, advantages, student_log_probs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
