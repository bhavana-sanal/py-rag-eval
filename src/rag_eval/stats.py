"""Uncertainty estimates for metric means over a question set."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Sample = Sequence[float] | NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Interval:
    mean: float
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class PairedComparison:
    mean_a: float
    mean_b: float
    delta: Interval
    """Mean of ``b - a`` with its bootstrap confidence interval."""
    p_value: float


def bootstrap_ci(
    values: Sample, *, resamples: int = 2000, confidence: float = 0.95, seed: int = 0
) -> Interval:
    """Percentile bootstrap confidence interval for the mean."""
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(resamples, data.size))].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return Interval(float(data.mean()), float(low), float(high))


def paired_comparison(
    a: Sample,
    b: Sample,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    """Compare two systems scored on the same questions.

    The CI on the mean difference comes from a paired bootstrap. The two-sided p-value
    comes from a sign-flip permutation test: under the null hypothesis that both
    systems are exchangeable, each per-question difference is equally likely to have
    either sign.
    """
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("paired samples must have the same length")
    diff = y - x
    delta = bootstrap_ci(diff, resamples=resamples, confidence=confidence, seed=seed)
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(resamples, diff.size))
    null_means = (signs * diff).mean(axis=1)
    extreme = np.count_nonzero(np.abs(null_means) >= abs(delta.mean) - 1e-12)
    # The +1 counts the observed assignment itself, so p is never exactly zero.
    p_value = (extreme + 1) / (resamples + 1)
    return PairedComparison(float(x.mean()), float(y.mean()), delta, p_value)
