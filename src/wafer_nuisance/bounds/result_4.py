"""Result 4: certified die-level false-alarm rate (Hoeffding's inequality + a union bound).

p(t) = probability that a defect-free die is flagged at score threshold t (some candidate's
stage-2 score exceeds t); p_hat_m(t) = its frequency on m i.i.d. defect-free dies. For each fixed
t the flags are i.i.d. Bernoulli(p(t)), so Hoeffding's inequality applied to 1 - X_i gives
Pr(p(t) - p_hat_m(t) >= eps) <= exp(-2 m eps^2). For a grid t_1, ..., t_T fixed before the
certification dies are drawn, a union bound over the T thresholds gives, with probability
>= 1 - delta, simultaneously for all j:

    p(t_j) <= p_hat_m(t_j) + sqrt( ln(T / delta) / (2m) ).

Selection rule: the most sensitive (lowest) grid threshold whose right-hand side is <= alpha.
Because the inequality holds for all j at once, it also holds for the selected (data-dependent)
threshold.

Multiplicative-Chernoff certificate. Hoeffding's margin is the one for a Bernoulli variance of
1/4; flags at a rate near alpha have variance about alpha. The lower-tail Chernoff bound
P(X <= (1 - eta) mu) <= exp(-eta^2 mu / 2) for X ~ Binomial(m, p), mu = m p, uses that. Rule:
certify the lowest t_j with

    p_hat_m(t_j) <= (1 - eta) alpha,   eta = sqrt( 2 ln(T / delta) / (alpha m) ),

and nothing if eta >= 1. For a threshold with p(t_j) > alpha the binomial lower tail is largest
at p = alpha (Binomial(m, p) is stochastically increasing in p), where Chernoff gives
exp(-eta^2 alpha m / 2) = delta / T; a union bound over the T thresholds gives delta. The
matching simultaneous upper bound is U(p_hat) = ((sqrt(c) + sqrt(c + 4 p_hat)) / 2)^2 with
c = 2 ln(T / delta) / m (``chernoff_certified_far``): p(t_j) > U(p_hat_m(t_j)) iff
p_hat_m(t_j) < p(t_j) - sqrt(c p(t_j)) = (1 - eta(p)) p(t_j), of probability <= delta / T, and
U(p_hat) <= alpha iff p_hat <= (1 - eta) alpha.
"""

from __future__ import annotations

import math

import numpy as np


def certification_margin(m: int, T: int, delta: float = 0.05) -> float:
    """sqrt(ln(T/delta) / (2m))."""
    if m <= 0 or T <= 0 or not 0 < delta < 1:
        raise ValueError("need m, T >= 1 and 0 < delta < 1")
    return math.sqrt(math.log(T / delta) / (2 * m))


def certified_far(p_hat: np.ndarray, m: int, delta: float = 0.05) -> np.ndarray:
    """Upper confidence bounds p_hat(t_j) + margin for a grid of T = len(p_hat) thresholds."""
    p_hat = np.asarray(p_hat, dtype=np.float64)
    return p_hat + certification_margin(m, len(p_hat), delta)


def select_threshold(grid: np.ndarray, p_hat: np.ndarray, m: int, alpha: float = 0.05,
                     delta: float = 0.05) -> int | None:
    """Index of the lowest grid threshold whose certified FAR is <= alpha, or None ("no
    certifiable threshold at this m"). ``grid`` must be increasing."""
    grid = np.asarray(grid)
    if np.any(np.diff(grid) <= 0):
        raise ValueError("grid must be strictly increasing")
    ok = np.flatnonzero(certified_far(p_hat, m, delta) <= alpha)
    return int(ok[0]) if len(ok) else None


def violation_rate_bound(v: int, R: int, delta: float = 0.05) -> float:
    """Upper confidence bound on a failure rate q from v failures in R i.i.d. Bernoulli(q) trials
    (used for the violation rate of the certificate over independent repeats, and for L(h*)
    from zero errors on an independent pool):

        v = 0:  ln(1/delta) / R,
        v > 0:  v/R + sqrt( ln(1/delta) / (2R) )   (capped at 1).

    Derivation. v = 0: P(V = 0) = (1 - q)^R <= e^(-qR) (from 1 - q <= e^-q), which is < delta
    whenever q > ln(1/delta)/R. v > 0: one-sided Hoeffding,
    P(q - V/R >= eps) <= exp(-2R eps^2) = delta at eps = sqrt(ln(1/delta)/(2R)).
    The combined rule U(V) is still a valid (1 - delta) upper bound for every q. Let
    s = sqrt(ln(1/delta)/(2R)). If q <= s, U(V) >= s >= q whenever V > 0, so the only possible
    failure is V = 0 with q > ln(1/delta)/R, of probability < delta. If q > s, failing means
    V/R < q - s (for V = 0 too, since q - s > 0), of probability <= delta by Hoeffding.
    """
    if R <= 0 or not 0 <= v <= R or not 0 < delta < 1:
        raise ValueError("need R >= 1, 0 <= v <= R and 0 < delta < 1")
    if v == 0:
        return min(1.0, math.log(1 / delta) / R)
    return min(1.0, v / R + math.sqrt(math.log(1 / delta) / (2 * R)))


def chernoff_eta(m: int, T: int, alpha: float = 0.05, delta: float = 0.05) -> float:
    """eta = sqrt(2 ln(T/delta) / (alpha m)): the relative margin of the Chernoff rule."""
    if m <= 0 or T <= 0 or not 0 < delta < 1 or not 0 < alpha < 1:
        raise ValueError("need m, T >= 1 and 0 < alpha, delta < 1")
    return math.sqrt(2 * math.log(T / delta) / (alpha * m))


def chernoff_certified_far(p_hat: np.ndarray, m: int, delta: float = 0.05) -> np.ndarray:
    """Simultaneous upper bounds U(p_hat(t_j)) = ((sqrt(c) + sqrt(c + 4 p_hat)) / 2)^2,
    c = 2 ln(T/delta) / m, for a grid of T = len(p_hat) thresholds (see the module docstring)."""
    p_hat = np.asarray(p_hat, dtype=np.float64)
    c = 2 * math.log(len(p_hat) / delta) / m
    return ((math.sqrt(c) + np.sqrt(c + 4 * p_hat)) / 2) ** 2


def select_threshold_chernoff(grid: np.ndarray, p_hat: np.ndarray, m: int, alpha: float = 0.05,
                              delta: float = 0.05) -> int | None:
    """Index of the lowest grid threshold with p_hat <= (1 - eta) alpha, or None (also when
    eta >= 1, where no threshold can be certified). ``grid`` must be increasing."""
    grid = np.asarray(grid)
    if np.any(np.diff(grid) <= 0):
        raise ValueError("grid must be strictly increasing")
    eta = chernoff_eta(m, len(grid), alpha, delta)
    if eta >= 1:
        return None
    ok = np.flatnonzero(np.asarray(p_hat) <= (1 - eta) * alpha)
    return int(ok[0]) if len(ok) else None
