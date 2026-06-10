from __future__ import annotations

import sys
from math import inf
from math import nan
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.rollout.filter_hub.correctness_balance import rollout_sample_filter
from slime.utils.types import Sample


NUM_GPUS = 0


def _args(mode: str = "global", ratio: float = 1.0):
    return SimpleNamespace(opd_correctness_balance_mode=mode, opd_correctness_balance_ratio=ratio)


def _sample(index: int, response_correct=...):
    metadata = {}
    if response_correct is not ...:
        metadata["response_correct"] = response_correct
    return Sample(index=index, metadata=metadata)


def _removed_indices(groups: list[list[Sample]]) -> list[int]:
    return [sample.index for group in groups for sample in group if sample.remove_sample]


def _known_counts(groups: list[list[Sample]]) -> tuple[int, int]:
    correct = incorrect = 0
    for sample in [sample for group in groups for sample in group if not sample.remove_sample]:
        value = sample.metadata.get("response_correct") if isinstance(sample.metadata, dict) else None
        if value is True:
            correct += 1
        elif value is False:
            incorrect += 1
    return correct, incorrect


@pytest.mark.unit
def test_global_correctness_balance_masks_only_known_surplus_samples():
    groups = [
        [_sample(0, True), _sample(1, True)],
        [_sample(2, True), _sample(3, False)],
        [_sample(4), _sample(5, "yes")],
    ]

    rollout_sample_filter(_args(mode="global", ratio=1.0), groups)

    assert [len(group) for group in groups] == [2, 2, 2]
    assert _removed_indices(groups) == [1, 2]
    assert _known_counts(groups) == (1, 1)
    assert groups[2][0].remove_sample is False
    assert groups[2][1].remove_sample is False


@pytest.mark.unit
def test_per_group_correctness_balance_applies_inside_each_group():
    groups = [
        [_sample(0, True), _sample(1, True), _sample(2, False)],
        [_sample(3, True), _sample(4, False), _sample(5, False)],
    ]

    rollout_sample_filter(_args(mode="per_group", ratio=1.0), groups)

    assert [len(group) for group in groups] == [3, 3]
    assert _removed_indices(groups) == [1, 5]
    assert _known_counts([groups[0]]) == (1, 1)
    assert _known_counts([groups[1]]) == (1, 1)


@pytest.mark.unit
def test_correctness_balance_supports_incorrect_only_target_ratio():
    groups = [[_sample(0, True), _sample(1, False), _sample(2, True), _sample(3, False)]]

    rollout_sample_filter(_args(ratio=0.0), groups)

    assert _removed_indices(groups) == [0, 2]
    assert _known_counts(groups) == (0, 2)


@pytest.mark.unit
def test_correctness_balance_supports_correct_only_target_ratio():
    groups = [[_sample(0, True), _sample(1, False), _sample(2, True), _sample(3, False)]]

    rollout_sample_filter(_args(ratio=inf), groups)

    assert _removed_indices(groups) == [1, 3]
    assert _known_counts(groups) == (2, 0)


@pytest.mark.unit
def test_correctness_balance_masks_fractional_correct_surplus():
    groups = [[_sample(0, True), _sample(1, True), _sample(2, True), _sample(3, False)]]

    rollout_sample_filter(_args(ratio=0.5), groups)

    assert _removed_indices(groups) == [0, 1, 2]
    assert _known_counts(groups) == (0, 1)


@pytest.mark.unit
def test_correctness_balance_masks_fractional_incorrect_surplus():
    groups = [[_sample(0, True), _sample(1, False), _sample(2, False), _sample(3, False)]]

    rollout_sample_filter(_args(ratio=2.0), groups)

    assert _removed_indices(groups) == [1, 2, 3]
    assert _known_counts(groups) == (1, 0)


@pytest.mark.unit
def test_correctness_balance_keeps_single_class_batches_for_finite_ratio():
    all_correct_groups = [[_sample(0, True), _sample(1, True)]]
    all_incorrect_groups = [[_sample(2, False), _sample(3, False)]]

    rollout_sample_filter(_args(ratio=1.0), all_correct_groups)
    rollout_sample_filter(_args(ratio=1.0), all_incorrect_groups)

    assert _removed_indices(all_correct_groups) == []
    assert _removed_indices(all_incorrect_groups) == []


@pytest.mark.unit
def test_correctness_balance_rejects_invalid_config():
    groups = [[_sample(0, True), _sample(1, False)]]

    with pytest.raises(ValueError, match="non-negative"):
        rollout_sample_filter(_args(ratio=-1.0), groups)

    with pytest.raises(ValueError, match="not NaN"):
        rollout_sample_filter(_args(ratio=nan), groups)

    with pytest.raises(ValueError, match="global.*per_group"):
        rollout_sample_filter(_args(mode="sample"), groups)


@pytest.mark.unit
def test_correctness_balance_zero_surplus_mask_is_noop():
    groups = [[_sample(0, True), _sample(1, False)]]

    rollout_sample_filter(_args(ratio=1.0), groups)

    assert _removed_indices(groups) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
