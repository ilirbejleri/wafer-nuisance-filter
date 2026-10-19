"""Result 2: generalization bounds for the linear filter x -> sign(w.x), x in R^(k+1) (k
features and the intercept coordinate), labels y in {-1, +1}.

VC bound (linear classifiers with intercept on R^k have VC dimension k + 1; uniform convergence
from symmetrization, Hoeffding and Sauer-Shelah: Pr(max_h |L(h) - L_hat(h)| > eps) <= 4 (2en/d)^d
exp(-n eps^2 / 8)). Setting the right-hand side to delta, with probability >= 1 - delta, for every
w:

    L01(w) <= L_hat01(w) + sqrt( 8 ((k+1) ln(2en/(k+1)) + ln(4/delta)) / n ).

Rademacher bound, for ||w|| <= B and ||x|| <= C. Apply the Rademacher generalization bound
(McDiarmid + symmetrization) to the loss class F = {(x, y) -> l(y w.x) : ||w|| <= B} with l(u) =
ln(1 + e^-u):

* sigma_i y_i is again uniform on {-1, +1}, so the margin class {(x, y) -> y w.x} has the same
  empirical Rademacher complexity as the linear class {x -> w.x}; Cauchy-Schwarz and Jensen bound it
  by (B/n) sqrt(sum_i ||x_i||^2) (<= BC/sqrt(n));
* l is 1-Lipschitz (|l'(u)| = 1/(1 + e^u) <= 1), so by the contraction lemma the loss class has
  at most that complexity;
* the margin lies in [-BC, BC], so l lies in [ln(1 + e^-BC), ln(1 + e^BC)], an interval of
  length exactly BC (ln(1 + e^a) - ln(1 + e^-a) = a): Delta_F = BC;
* a misclassified point has y w.x <= 0, hence l >= ln 2, so 1[y w.x <= 0] <= l(y w.x) / ln 2.

Hence with probability >= 1 - delta, for every ||w|| <= B:

    L01(w) <= (1/ln 2) ( L_hat_log(w) + (2B/n) sqrt(sum_i ||x_i||^2) + 3BC sqrt(ln(2/delta)/(2n)) ).

Clipped-loss Rademacher bound (the one reported). Same class and same steps, with the logistic
loss replaced by psi(u) = min(ln(1 + e^-u), ln 2) in the bound only (training is unchanged):

* if u <= 0 then ln(1 + e^-u) >= ln 2, so psi(u) = ln 2 and 1[u <= 0] <= psi(u) / ln 2;
* psi is the minimum of a 1-Lipschitz function and a constant, hence 1-Lipschitz, so the
  contraction step and the complexity term are unchanged;
* psi takes values in (0, ln 2], so Delta_F <= ln 2 for every B (instead of BC).

Hence with probability >= 1 - delta, for every ||w|| <= B:

    L01(w) <= (1/ln 2) ( L_hat_psi(w) + (2B/n) sqrt(sum_i ||x_i||^2)
                         + 3 ln 2 sqrt(ln(2/delta)/(2n)) ).

For a grid of G values of B, each Rademacher bound is computed at delta/G, so by a union bound
all G hold together with probability >= 1 - delta and B may be chosen after looking at data.
The VC bound already holds uniformly over all w, so it needs no such correction.
"""

from __future__ import annotations

import math

import numpy as np


def vc_bound(train_error, n, k: int, delta: float = 0.05):
    """L_hat01 + sqrt(8 ((k+1) ln(2en/(k+1)) + ln(4/delta)) / n)  (VC dimension k + 1)."""
    n = np.asarray(n, dtype=np.float64)
    d = k + 1
    if np.any(n < d) or not 0 < delta < 1:
        raise ValueError("need n >= k + 1 and 0 < delta < 1")
    return np.asarray(train_error) + np.sqrt(
        8 * (d * np.log(2 * math.e * n / d) + math.log(4 / delta)) / n)


def rademacher_complexity_bound(X: np.ndarray, B: float) -> float:
    """(B/n) sqrt(sum_i ||x_i||^2): the bound on the empirical Rademacher complexity of
    {x -> w.x : ||w|| <= B} (Cauchy-Schwarz, then Jensen)."""
    X = np.asarray(X, dtype=np.float64)
    return B / len(X) * math.sqrt(float((X * X).sum()))


def rademacher_bound(train_log_loss: float, X: np.ndarray, B: float, C: float,
                     delta: float = 0.05) -> float:
    """(1/ln 2)(L_hat_log + 2 Rhat + 3 B C sqrt(ln(2/delta)/(2n)))  (Rademacher, logistic loss).
    Pass delta/G when B comes from a grid of G values."""
    n = len(X)
    if np.linalg.norm(X, axis=1).max() > C * (1 + 1e-9):
        raise ValueError("some ||x_i|| exceeds C")
    return (train_log_loss + 2 * rademacher_complexity_bound(X, B)
            + 3 * B * C * math.sqrt(math.log(2 / delta) / (2 * n))) / math.log(2)


def clipped_logistic_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """L_hat_psi(w) = (1/n) sum_i psi(y_i w.x_i) with psi(u) = min(ln(1 + e^-u), ln 2)."""
    u = y * (np.asarray(X, dtype=np.float64) @ w)
    return float(np.minimum(np.logaddexp(0.0, -u), math.log(2)).mean())


def rademacher_bound_clipped(train_psi_loss: float, X: np.ndarray, B: float, C: float,
                             delta: float = 0.05) -> float:
    """(1/ln 2)(L_hat_psi + 2 Rhat + 3 ln 2 sqrt(ln(2/delta)/(2n)))  (the Rademacher bound
    applied to the clipped loss psi(u) = min(ln(1 + e^-u), ln 2)).

    Derivation: (a) for u <= 0, psi(u) = ln 2, so 1[u <= 0] <= psi(u) / ln 2 and
    L01(w) <= E psi / ln 2; (b) psi = min(1-Lipschitz function, constant) is 1-Lipschitz, so by
    the contraction lemma and the margin / linear-class steps of the logistic bound the loss class
    has Rhat <= (B/n) sqrt(sum ||x_i||^2); (c) psi takes values in (0, ln 2], so the
    McDiarmid range is Delta_F <= ln 2 for every B, and the generalization bound gives
    E psi <= L_hat_psi + 2 Rhat + 3 ln 2 sqrt(ln(2/delta)/(2n)) for every ||w|| <= B at once.
    ``C`` only enters through the check ||x_i|| <= C. Pass delta/G when B comes from a grid of
    G values."""
    n = len(X)
    if np.linalg.norm(X, axis=1).max() > C * (1 + 1e-9):
        raise ValueError("some ||x_i|| exceeds C")
    if not 0 <= train_psi_loss <= math.log(2) * (1 + 1e-12):
        raise ValueError("the clipped loss lies in [0, ln 2]")
    return (train_psi_loss + 2 * rademacher_complexity_bound(X, B)
            + 3 * math.log(2) * math.sqrt(math.log(2 / delta) / (2 * n))) / math.log(2)
