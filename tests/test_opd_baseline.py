from __future__ import annotations

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
from slime.ray.rollout import compute_metrics_from_samples
from slime.rollout.on_policy_distillation import post_process_rewards
from slime.utils.types import Sample


NUM_GPUS = 0


def _make_sglang_reward(logprobs: list[float]) -> dict:
    return {
        "meta_info": {
            "input_token_logprobs": [(None,)] + [(logprob,) for logprob in logprobs],
        }
    }


@pytest.mark.unit
def test_post_process_rewards_trims_teacher_logprobs_to_response_span():
    args = Namespace(reward_key=None)
    samples = [
        Sample(response_length=2, reward=_make_sglang_reward([-9.0, -8.0, -0.3, -0.2])),
        Sample(response_length=3, reward=_make_sglang_reward([-5.0, -0.7, -0.6, -0.5])),
    ]

    rewards, scores = post_process_rewards(args, samples)

    assert rewards == [0.0, 0.0]
    assert scores == [0.0, 0.0]
    assert torch.equal(samples[0].teacher_log_probs, torch.tensor([-0.3, -0.2], dtype=torch.float32))
    assert torch.equal(samples[1].teacher_log_probs, torch.tensor([-0.7, -0.6, -0.5], dtype=torch.float32))
    assert samples[0].teacher_log_probs.dtype is torch.float32
    assert samples[1].teacher_log_probs.dtype is torch.float32


@pytest.mark.unit
def test_post_process_rewards_handles_zero_length_response():
    args = Namespace(reward_key=None)
    sample = Sample(response_length=0, reward=_make_sglang_reward([-9.0, -8.0]))

    rewards, scores = post_process_rewards(args, [sample])

    assert rewards == [0.0]
    assert scores == [0.0]
    assert sample.teacher_log_probs.shape == (0,)
    assert sample.teacher_log_probs.dtype is torch.float32
    assert "opd_teacher_logprob_failed" not in sample.metadata


@pytest.mark.unit
def test_apply_opd_kl_updates_advantages_and_records_reverse_kl():
    args = Namespace(opd_type="sglang", opd_kl_coef=0.5)
    rollout_data = {
        "teacher_log_probs": [
            torch.tensor([-2.0, -4.0], dtype=torch.float32),
            torch.tensor([-1.0], dtype=torch.float32),
        ]
    }
    advantages = [
        torch.tensor([1.0, 1.0], dtype=torch.float32),
        torch.tensor([2.0], dtype=torch.float32),
    ]
    student_log_probs = [
        torch.tensor([-1.0, -3.0], dtype=torch.float32),
        torch.tensor([-1.5], dtype=torch.float32),
    ]

    apply_opd_kl_to_advantages(args, rollout_data, advantages, student_log_probs)

    assert torch.equal(rollout_data["opd_reverse_kl"][0], torch.tensor([1.0, 1.0]))
    assert torch.equal(rollout_data["opd_reverse_kl"][1], torch.tensor([-0.5]))
    assert torch.equal(advantages[0], torch.tensor([0.5, 0.5]))
    assert torch.equal(advantages[1], torch.tensor([2.25]))


@pytest.mark.unit
def test_apply_opd_kl_is_noop_without_student_logprobs():
    args = Namespace(opd_type="sglang", opd_kl_coef=1.0)
    rollout_data = {"teacher_log_probs": [torch.tensor([-2.0], dtype=torch.float32)]}
    advantages = [torch.tensor([3.0], dtype=torch.float32)]

    apply_opd_kl_to_advantages(args, rollout_data, advantages, student_log_probs=None)

    assert torch.equal(advantages[0], torch.tensor([3.0]))
    assert "opd_reverse_kl" not in rollout_data


@pytest.mark.unit
def test_apply_opd_kl_requires_teacher_logprobs_when_student_logprobs_exist():
    args = Namespace(opd_type="sglang", opd_kl_coef=1.0)

    with pytest.raises(ValueError, match="requires teacher_log_probs"):
        apply_opd_kl_to_advantages(
            args,
            rollout_data={},
            advantages=[torch.tensor([1.0], dtype=torch.float32)],
            student_log_probs=[torch.tensor([-1.0], dtype=torch.float32)],
        )


@pytest.mark.unit
def test_rollout_metrics_skip_zero_std_for_opd_dict_rewards_before_post_process():
    args = Namespace(log_reward_category=None, advantage_estimator="grpo", reward_key=None)
    samples = [
        Sample(group_index=0, response_length=1, response="a", reward=_make_sglang_reward([-0.1])),
        Sample(group_index=0, response_length=1, response="b", reward=_make_sglang_reward([-0.2])),
    ]

    metrics = compute_metrics_from_samples(args, samples)

    assert not any(key.startswith("zero_std/") for key in metrics)
    assert "response_len/mean" in metrics


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
