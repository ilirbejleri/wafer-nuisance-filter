"""Error rates of many threshold classifiers on one large evaluation pool, and the pool-induced
noise floor of a measured excess error.

For thresholds theta_1, ..., theta_R and an evaluation pool (x_i, y_i), i = 1..N, with
h_theta(x) = +1 iff x > theta:

    err(theta) = ( #{i: y_i = +1, x_i <= theta} + #{i: y_i = -1, x_i > theta} ) / N,

computed for all thetas at once by binary search in the sorted pool.

Paired differences. err(theta) - err(theta*) only involves the points between the two
thresholds, where the classifiers disagree: each such point contributes +1/N or -1/N. If a
fraction d of the pool lies there, the difference is a mean of N i.i.d. terms in {-1, 0, +1}
with variance at most d / N, so its standard deviation from the finite pool is at most
sqrt(d / N). This is the noise floor below which a measured excess error is not resolved.
"""

from __future__ import annotations

import numpy as np


class ThresholdEvaluator:
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        x, y = np.asarray(x, np.float64), np.asarray(y)
        self.n = len(x)
        self.pos = np.sort(x[y == 1])
        self.neg = np.sort(x[y == -1])
        self.all = np.sort(x)

    def error(self, theta) -> np.ndarray:
        theta = np.asarray(theta, np.float64)
        pos_le = np.searchsorted(self.pos, theta, side="right")
        neg_gt = len(self.neg) - np.searchsorted(self.neg, theta, side="right")
        return (pos_le + neg_gt) / self.n

    def disagreement(self, theta, theta_star: float) -> np.ndarray:
        """Fraction of the pool with x between theta and theta_star, i.e. where h_theta and
        h_theta_star disagree."""
        theta = np.asarray(theta, np.float64)
        lo, hi = np.minimum(theta, theta_star), np.maximum(theta, theta_star)
        cnt = (np.searchsorted(self.all, hi, side="right")
               - np.searchsorted(self.all, lo, side="right"))
        return cnt / self.n


def noise_floor(mean_disagreement: float, n_pool: int) -> float:
    """sqrt(d / N): standard deviation bound of a paired error difference on an N-point pool."""
    return float(np.sqrt(mean_disagreement / n_pool))
