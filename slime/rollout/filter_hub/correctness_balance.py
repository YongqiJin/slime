import logging
import math

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

__all__ = ["rollout_sample_filter"]


def _get_response_correct(sample: Sample) -> bool | None:
    value = sample.metadata.get("response_correct") if isinstance(sample.metadata, dict) else None
    return value if isinstance(value, bool) else None


def _mask_tail(samples: list[Sample], count: int) -> None:
    if count <= 0:
        return
    for sample in samples[-count:]:
        sample.remove_sample = True


def _mask_excess_samples(correct_samples: list[Sample], incorrect_samples: list[Sample], ratio: float) -> tuple[int, int]:
    if math.isnan(ratio) or ratio < 0:
        raise ValueError("--opd-correctness-balance-ratio must be non-negative and not NaN.")

    correct_count = len(correct_samples)
    incorrect_count = len(incorrect_samples)

    if ratio == 0:
        _mask_tail(correct_samples, correct_count)
        return 0, incorrect_count

    if math.isinf(ratio):
        _mask_tail(incorrect_samples, incorrect_count)
        return correct_count, 0

    if correct_count == 0 or incorrect_count == 0:
        return correct_count, incorrect_count

    desired_correct = int(incorrect_count * ratio)
    desired_incorrect = int(correct_count / ratio) if ratio > 0 else correct_count

    if correct_count > desired_correct:
        _mask_tail(correct_samples, correct_count - desired_correct)
        return desired_correct, incorrect_count

    if incorrect_count > desired_incorrect:
        _mask_tail(incorrect_samples, incorrect_count - desired_incorrect)
        return correct_count, desired_incorrect

    return correct_count, incorrect_count


def _balance_scope(samples: list[Sample], ratio: float) -> tuple[int, int, int, int]:
    correct_samples = [sample for sample in samples if _get_response_correct(sample) is True]
    incorrect_samples = [sample for sample in samples if _get_response_correct(sample) is False]

    kept_correct, kept_incorrect = _mask_excess_samples(correct_samples, incorrect_samples, ratio)
    return len(correct_samples), len(incorrect_samples), kept_correct, kept_incorrect


def _balance_global(groups: list[list[Sample]], ratio: float) -> None:
    flat_samples = [sample for group in groups for sample in group]
    before_correct, before_incorrect, after_correct, after_incorrect = _balance_scope(flat_samples, ratio)
    logger.info(
        "OPD correctness balance global: correct=%s->%s incorrect=%s->%s ratio=%s",
        before_correct,
        after_correct,
        before_incorrect,
        after_incorrect,
        ratio,
    )


def _balance_per_group(groups: list[list[Sample]], ratio: float) -> None:
    before_correct = before_incorrect = after_correct = after_incorrect = 0
    for group in groups:
        group_before_correct, group_before_incorrect, group_after_correct, group_after_incorrect = _balance_scope(
            group, ratio
        )
        before_correct += group_before_correct
        before_incorrect += group_before_incorrect
        after_correct += group_after_correct
        after_incorrect += group_after_incorrect

    logger.info(
        "OPD correctness balance per_group: correct=%s->%s incorrect=%s->%s ratio=%s",
        before_correct,
        after_correct,
        before_incorrect,
        after_incorrect,
        ratio,
    )


def rollout_sample_filter(args, groups: list[list[Sample]]) -> None:
    mode = getattr(args, "opd_correctness_balance_mode", "global")
    ratio = getattr(args, "opd_correctness_balance_ratio", 1.0)

    if mode == "global":
        _balance_global(groups, ratio)
        return

    if mode == "per_group":
        _balance_per_group(groups, ratio)
        return

    raise ValueError("--opd-correctness-balance-mode must be 'global' or 'per_group'.")
