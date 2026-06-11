from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import _cp_dist_helpers  # noqa: F401
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import slime.backends.megatron_utils.loss as loss_module
from slime.backends.megatron_utils.loss import apply_opd_margin_shift_to_advantages, policy_loss_function
from slime.ray.rollout import RolloutManager
from slime.utils.types import Sample


NUM_GPUS = 0
RawRolloutManager = RolloutManager.__ray_metadata__.modified_class


class _FakeRolloutManager:
    custom_convert_samples_to_train_data_func = None

    def _post_process_rewards(self, samples):
        rewards = [sample.reward for sample in samples]
        return rewards, rewards


def _args(**overrides):
    values = dict(
        use_opd_margin_shift=True,
        opd_margin_scope="local",
        opd_margin_mode="mean",
        opd_margin_delta=0.5,
        opd_margin_direction="both",
    )
    values.update(overrides)
    return Namespace(**values)


def _sample(index: int, group_index: int | None = None):
    return Sample(
        index=index,
        rollout_id=index,
        group_index=group_index,
        tokens=[index, index + 1],
        response_length=1,
        reward=float(index),
        loss_mask=[1],
        metadata={},
        status=Sample.Status.COMPLETED,
    )


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
def test_opd_margin_shift_disabled_is_noop():
    advantages = [torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])]
    rollout_data = {"response_correct": [True, False]}

    apply_opd_margin_shift_to_advantages(
        _args(use_opd_margin_shift=False),
        rollout_data,
        advantages,
        loss_masks=[torch.ones(2), torch.ones(2)],
    )

    assert torch.equal(advantages[0], torch.tensor([0.0, 0.0]))
    assert torch.equal(advantages[1], torch.tensor([1.0, 1.0]))
    assert "opd_margin_shift" not in rollout_data


@pytest.mark.unit
def test_local_opd_margin_shift_skips_unknown_and_zero_mask_samples():
    advantages = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([10.0]),
        torch.tensor([3.0, 3.0]),
    ]
    rollout_data = {"response_correct": [True, False, None, True]}
    loss_masks = [torch.ones(2), torch.ones(2), torch.ones(1), torch.zeros(2)]

    apply_opd_margin_shift_to_advantages(_args(), rollout_data, advantages, loss_masks=loss_masks)

    assert torch.equal(advantages[0], torch.tensor([0.75, 0.75]))
    assert torch.equal(advantages[1], torch.tensor([0.25, 0.25]))
    assert torch.equal(advantages[2], torch.tensor([10.0]))
    assert torch.equal(advantages[3], torch.tensor([3.0, 3.0]))
    assert rollout_data["opd_margin_affected_samples"].item() == 2
    assert rollout_data["opd_margin_known_samples"].item() == 2
    assert torch.equal(rollout_data["opd_margin_shift"][0], torch.tensor([0.75, 0.75]))
    assert torch.equal(rollout_data["opd_margin_shift"][1], torch.tensor([-0.75, -0.75]))


@pytest.mark.unit
def test_group_opd_margin_shift_applies_per_group_only_when_gap_is_small():
    advantages = [
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.tensor([5.0]),
        torch.tensor([1.0]),
    ]
    rollout_data = {
        "response_correct": [True, False, True, False],
        "sample_group_index": [0, 0, 1, 1],
    }

    apply_opd_margin_shift_to_advantages(
        _args(opd_margin_scope="group", opd_margin_direction="correct_up"),
        rollout_data,
        advantages,
        loss_masks=[torch.ones(1) for _ in advantages],
    )

    assert torch.equal(advantages[0], torch.tensor([1.5]))
    assert torch.equal(advantages[1], torch.tensor([1.0]))
    assert torch.equal(advantages[2], torch.tensor([5.0]))
    assert torch.equal(advantages[3], torch.tensor([1.0]))
    assert rollout_data["opd_margin_affected_samples"].item() == 1


@pytest.mark.unit
def test_global_opd_margin_shift_falls_back_to_local_when_distributed_uninitialized():
    advantages = [
        torch.tensor([0.0]),
        torch.tensor([1.0]),
    ]
    rollout_data = {"response_correct": [True, False]}

    apply_opd_margin_shift_to_advantages(
        _args(opd_margin_scope="global", opd_margin_direction="correct_up"),
        rollout_data,
        advantages,
        loss_masks=[torch.ones(1), torch.ones(1)],
    )

    assert torch.equal(advantages[0], torch.tensor([1.5]))
    assert torch.equal(advantages[1], torch.tensor([1.0]))
    assert rollout_data["opd_margin_affected_samples"].item() == 1
    assert rollout_data["opd_margin_correct_samples"].item() == 1
    assert rollout_data["opd_margin_incorrect_samples"].item() == 1


@pytest.mark.unit
def test_global_opd_margin_shift_uses_data_parallel_group_for_mean(monkeypatch):
    dp_group = object()
    reduce_calls = []

    monkeypatch.setattr(loss_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(loss_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(loss_module.mpu, "get_data_parallel_group", lambda: dp_group, raising=False)

    def fake_all_reduce(tensor, op=None, group=None):
        reduce_calls.append((op, group, tensor.clone()))

    monkeypatch.setattr(loss_module.dist, "all_reduce", fake_all_reduce)

    advantages = [torch.tensor([0.0]), torch.tensor([1.0])]
    rollout_data = {"response_correct": [True, False]}

    apply_opd_margin_shift_to_advantages(
        _args(opd_margin_scope="global", opd_margin_direction="correct_up"),
        rollout_data,
        advantages,
        loss_masks=[torch.ones(1), torch.ones(1)],
    )

    assert torch.equal(advantages[0], torch.tensor([1.5]))
    assert len(reduce_calls) == 1
    op, group, stats = reduce_calls[0]
    assert op == loss_module.dist.ReduceOp.SUM
    assert group is dp_group
    assert torch.equal(stats, torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=torch.float64))


@pytest.mark.unit
def test_global_opd_margin_shift_uses_data_parallel_group_for_minmax(monkeypatch):
    dp_group = object()
    reduce_calls = []

    monkeypatch.setattr(loss_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(loss_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(loss_module.mpu, "get_data_parallel_group", lambda: dp_group, raising=False)

    def fake_all_reduce(tensor, op=None, group=None):
        reduce_calls.append((op, group, tensor.clone()))

    monkeypatch.setattr(loss_module.dist, "all_reduce", fake_all_reduce)

    advantages = [torch.tensor([0.0]), torch.tensor([2.0])]
    rollout_data = {"response_correct": [True, False]}

    apply_opd_margin_shift_to_advantages(
        _args(
            opd_margin_scope="global",
            opd_margin_mode="minmax",
            opd_margin_direction="incorrect_down",
        ),
        rollout_data,
        advantages,
        loss_masks=[torch.ones(1), torch.ones(1)],
    )

    assert torch.equal(advantages[1], torch.tensor([-0.5]))
    assert [(op, group) for op, group, _ in reduce_calls] == [
        (loss_module.dist.ReduceOp.MIN, dp_group),
        (loss_module.dist.ReduceOp.SUM, dp_group),
    ]
    assert torch.equal(reduce_calls[0][2], torch.tensor([0.0, -2.0], dtype=torch.float64))
    assert torch.equal(reduce_calls[1][2], torch.tensor([1.0, 1.0], dtype=torch.float64))


@pytest.mark.unit
def test_group_opd_margin_shift_skips_none_group_index():
    advantages = [
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.tensor([2.0]),
    ]
    rollout_data = {
        "response_correct": [True, False, True],
        "sample_group_index": [0, 0, None],
    }

    apply_opd_margin_shift_to_advantages(
        _args(opd_margin_scope="group", opd_margin_direction="correct_up"),
        rollout_data,
        advantages,
        loss_masks=[torch.ones(1) for _ in advantages],
    )

    assert torch.equal(advantages[0], torch.tensor([1.5]))
    assert torch.equal(advantages[1], torch.tensor([1.0]))
    assert torch.equal(advantages[2], torch.tensor([2.0]))


@pytest.mark.unit
def test_opd_margin_shift_rejects_mismatched_metadata_lengths():
    with pytest.raises(ValueError, match="response_correct length"):
        apply_opd_margin_shift_to_advantages(
            _args(),
            {"response_correct": [True]},
            [torch.tensor([0.0]), torch.tensor([1.0])],
            loss_masks=[torch.ones(1), torch.ones(1)],
        )

    with pytest.raises(ValueError, match="loss_masks length"):
        apply_opd_margin_shift_to_advantages(
            _args(),
            {"response_correct": [True, False]},
            [torch.tensor([0.0]), torch.tensor([1.0])],
            loss_masks=[torch.ones(1)],
        )

    with pytest.raises(ValueError, match="sample_group_index length"):
        apply_opd_margin_shift_to_advantages(
            _args(opd_margin_scope="group"),
            {"response_correct": [True, False], "sample_group_index": [0]},
            [torch.tensor([0.0]), torch.tensor([1.0])],
            loss_masks=[torch.ones(1), torch.ones(1)],
        )


@pytest.mark.unit
def test_minmax_opd_margin_shift_supports_incorrect_down_direction():
    advantages = [
        torch.tensor([0.0]),
        torch.tensor([2.0]),
        torch.tensor([3.0]),
        torch.tensor([1.0]),
    ]
    rollout_data = {"response_correct": [True, True, False, False]}

    apply_opd_margin_shift_to_advantages(
        _args(opd_margin_mode="minmax", opd_margin_direction="incorrect_down"),
        rollout_data,
        advantages,
        loss_masks=[torch.ones(1) for _ in advantages],
    )

    assert torch.equal(advantages[0], torch.tensor([0.0]))
    assert torch.equal(advantages[1], torch.tensor([2.0]))
    assert torch.equal(advantages[2], torch.tensor([-0.5]))
    assert torch.equal(advantages[3], torch.tensor([-2.5]))


@pytest.mark.unit
def test_policy_loss_ignores_absent_margin_shift_metrics(monkeypatch):
    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: (
            None,
            {
                "log_probs": [torch.tensor([0.1])],
                "entropy": [torch.tensor([0.0])],
            },
        ),
    )
    args = Namespace(
        use_rollout_logprobs=False,
        use_opsm=False,
        advantage_estimator="grpo",
        eps_clip=0.2,
        eps_clip_high=0.2,
        get_mismatch_metrics=False,
        use_tis=False,
        custom_pg_loss_reducer_function_path=None,
        calculate_per_token_loss=False,
        entropy_coef=0.0,
        use_kl_loss=False,
    )
    batch = {
        "advantages": [torch.tensor([1.0])],
        "log_probs": [torch.tensor([0.1])],
        "unconcat_tokens": [torch.tensor([1])],
        "response_lengths": [1],
        "total_lengths": [1],
        "loss_masks": [torch.ones(1)],
        "rollout_mask_sums": [1],
        "opd_margin_shift": None,
        "opd_margin_affected": None,
    }

    _, reported_loss = policy_loss_function(args, batch, torch.zeros(1, 1, 1), lambda tensor: tensor.mean())

    assert "opd_margin_shift" not in reported_loss
    assert "opd_margin_affected" not in reported_loss


@pytest.mark.unit
def test_sample_group_index_propagates_through_train_data_split(monkeypatch):
    import slime.ray.rollout as rollout_module

    monkeypatch.setattr(rollout_module.ray, "put", lambda value: value)
    samples = [_sample(0, 10), _sample(1, 10), _sample(2, 11), _sample(3, 11)]

    train_data = RawRolloutManager._convert_samples_to_train_data(_FakeRolloutManager(), samples)
    assert train_data["sample_group_index"] == [10, 10, 11, 11]

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
            rank0["sample_group_index"] + rank1["sample_group_index"],
            strict=True,
        )
    )
    assert pairs == [(0, 10), (1, 10), (2, 11), (3, 11)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
