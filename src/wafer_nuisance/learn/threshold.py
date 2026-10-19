"""Exact ERM over the threshold class H = {x -> sign(x - theta)} on a scalar feature.

h_theta(x) = +1 if x > theta, else -1. With the n sample values sorted, every distinct
classifier corresponds to a cut between two consecutive distinct values (or below all / above
all of them), so ERM is a sort and one sweep:

    errors(cut) = #{positives at or left of the cut} + #{negatives right of the cut},

computed for all cuts at once with cumulative sums, O(n log n) in total. theta is the midpoint
of the chosen gap (-inf / +inf for the two outer cuts). If several cuts reach the minimum, the
lowest one is taken, so the output is a deterministic function of the sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdClassifier:
    theta: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.where(np.asarray(x) > self.theta, 1, -1)

    def error(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) != y))


def threshold_errors(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(thetas, training error counts) for every distinct cut, lowest theta first."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    if x.ndim != 1 or x.shape != y.shape or not np.isin(y, (-1, 1)).all():
        raise ValueError("need 1-D x and labels in {-1, +1} of the same length")
    vals, inv = np.unique(x, return_inverse=True)  # sorted distinct values
    pos = np.bincount(inv, weights=(y == 1), minlength=len(vals))
    neg = np.bincount(inv, weights=(y == -1), minlength=len(vals))
    # Cut j (j = 0..m) puts vals[:j] on the -1 side and vals[j:] on the +1 side.
    pos_left = np.concatenate([[0.0], np.cumsum(pos)])
    neg_right = neg.sum() - np.concatenate([[0.0], np.cumsum(neg)])
    errors = pos_left + neg_right
    thetas = np.concatenate([[-np.inf], (vals[:-1] + vals[1:]) / 2, [np.inf]])
    return thetas, errors.astype(np.int64)


def fit_threshold(x: np.ndarray, y: np.ndarray) -> ThresholdClassifier:
    """Exact empirical risk minimizer over H (lowest minimizing cut)."""
    thetas, errors = threshold_errors(x, y)
    return ThresholdClassifier(float(thetas[np.argmin(errors)]))
