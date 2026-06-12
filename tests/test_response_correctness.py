from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.ray.rollout import RolloutManager, compute_metrics_from_samples
from slime.utils.types import Sample


NUM_GPUS = 0
RawRolloutManager = RolloutManager.__ray_metadata__.modified_class


class _FakeRolloutManager:
    custom_convert_samples_to_train_data_func = None

    def _post_process_rewards(self, samples):
        rewards = [sample.reward for sample in samples]
        return rewards, rewards


def _make_sample(index: int, response_correct=...):
    metadata = {}
    if response_correct is not ...:
        metadata["response_correct"] = response_correct
    return Sample(
        index=index,
        rollout_id=index,
        tokens=[index, index + 1, index + 2],
        response="answer",
        response_length=2,
        reward=float(index),
        loss_mask=[1, 1],
        metadata=metadata,
        status=Sample.Status.COMPLETED,
    )


def _metric_args():
    return SimpleNamespace(log_reward_category=None, advantage_estimator="ppo")


def _split_args():
    return SimpleNamespace(
        global_batch_size=4,
        micro_batch_size=1,
        use_dynamic_batch_size=False,
        max_tokens_per_gpu=None,
        balance_data=False,
        balance_by_flops=False,
    )


@pytest.mark.unit
def test_response_correctness_propagates_through_train_data_true_false_unknown():
    samples = [
        _make_sample(0, True),
        _make_sample(1, False),
        _make_sample(2),
    ]

    train_data = RawRolloutManager._convert_samples_to_train_data(_FakeRolloutManager(), samples)

    assert train_data["response_correct"] == [True, False, None]


@pytest.mark.unit
def test_response_correctness_is_omitted_when_all_samples_are_unknown():
    samples = [_make_sample(0), _make_sample(1, None), _make_sample(2, "yes")]

    train_data = RawRolloutManager._convert_samples_to_train_data(_FakeRolloutManager(), samples)

    assert "response_correct" not in train_data


@pytest.mark.unit
def test_response_correctness_is_split_into_rank_rollout_data(monkeypatch):
    import slime.ray.rollout as rollout_module

    monkeypatch.setattr(rollout_module.ray, "put", lambda value: value)
    samples = [
        _make_sample(0, True),
        _make_sample(1, False),
        _make_sample(2),
        _make_sample(3, True),
    ]
    train_data = RawRolloutManager._convert_samples_to_train_data(_FakeRolloutManager(), samples)
    manager = _FakeRolloutManager()
    manager.train_parallel_config = {
        "dp_size": 2,
        "cp_size": 1,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
    }
    manager.args = _split_args()

    refs = RawRolloutManager._split_train_data_by_dp(manager, train_data)
    rank0 = refs[0].inner
    rank1 = refs[1].inner

    pairs = sorted(
        zip(
            rank0["sample_indices"] + rank1["sample_indices"],
            rank0["response_correct"] + rank1["response_correct"],
            strict=True,
        )
    )
    assert pairs == [(0, True), (1, False), (2, None), (3, True)]


@pytest.mark.unit
def test_response_correctness_metrics_use_known_samples_only():
    samples = [
        _make_sample(0, True),
        _make_sample(1, False),
        _make_sample(2),
        _make_sample(3, True),
    ]

    metrics = compute_metrics_from_samples(_metric_args(), samples)

    assert metrics["response_correct/known_ratio"] == 0.75
    assert metrics["response_correct/accuracy"] == pytest.approx(2 / 3)


@pytest.mark.unit
def test_response_correctness_metrics_omitted_when_unknown():
    samples = [_make_sample(0), _make_sample(1, None), _make_sample(2, "yes")]

    metrics = compute_metrics_from_samples(_metric_args(), samples)

    assert "response_correct/known_ratio" not in metrics
    assert "response_correct/accuracy" not in metrics


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
