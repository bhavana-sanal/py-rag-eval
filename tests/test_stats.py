import pytest

from rag_eval.stats import bootstrap_ci, paired_comparison


def test_bootstrap_of_constant_sample_has_zero_width() -> None:
    interval = bootstrap_ci([0.5] * 20)
    assert (interval.mean, interval.low, interval.high) == (0.5, 0.5, 0.5)


def test_bootstrap_interval_brackets_mean_and_is_reproducible() -> None:
    values = [0.0, 1.0] * 25
    interval = bootstrap_ci(values, seed=7)
    assert interval.low < interval.mean == 0.5 < interval.high
    assert bootstrap_ci(values, seed=7) == interval


def test_bootstrap_rejects_empty_sample() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci([])


def test_paired_comparison_of_identical_systems() -> None:
    result = paired_comparison([0.2, 0.4, 0.9], [0.2, 0.4, 0.9])
    assert result.delta.mean == 0.0
    assert result.p_value == 1.0


def test_paired_comparison_detects_consistent_improvement() -> None:
    a = [0.1 * (i % 5) for i in range(40)]
    b = [x + 0.2 for x in a]
    result = paired_comparison(a, b, resamples=2000)
    assert result.delta.mean == pytest.approx(0.2)
    assert result.delta.low == pytest.approx(0.2)
    assert result.p_value < 0.01


def test_paired_comparison_requires_equal_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        paired_comparison([1.0], [1.0, 2.0])
