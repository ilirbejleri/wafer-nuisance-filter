"""Estimation error of a frequency (Hoeffding's inequality).

For X_1, ..., X_n i.i.d. Bernoulli(p) and p_hat their mean, Hoeffding's inequality gives
Pr(p_hat - p >= eps) <= exp(-2 n eps^2); applying it to 1 - X_i gives the other side, so by a
union bound over the two sides, with probability >= 1 - delta,

    |p_hat - p| <= eps(n, delta) = sqrt( ln(2/delta) / (2n) ).

Used to report the estimation error of every population quantity estimated from a finite pool
(e.g. a positive rate or an error rate measured on the test pool).
"""

from __future__ import annotations

import math


def hoeffding_halfwidth(n: int, delta: float = 0.05) -> float:
    """sqrt(ln(2/delta) / (2n)) (two-sided Hoeffding)."""
    if n <= 0 or not 0 < delta < 1:
        raise ValueError("need n >= 1 and 0 < delta < 1")
    return math.sqrt(math.log(2 / delta) / (2 * n))


def hoeffding_interval(p_hat: float, n: int, delta: float = 0.05) -> tuple[float, float]:
    """[p_hat - eps, p_hat + eps] clipped to [0, 1]: contains p w.p. >= 1 - delta."""
    eps = hoeffding_halfwidth(n, delta)
    return max(0.0, p_hat - eps), min(1.0, p_hat + eps)
