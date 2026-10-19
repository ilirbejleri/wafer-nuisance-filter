"""Result 3: the asymptotic learning curve of unregularized logistic regression (asymptotic
normality of M-estimators with the sandwich covariance).

Parameters theta in R^p (p = k + 1 with the intercept), loss
l(theta; x, y) = ln(1 + e^(-y theta.x)), phi = sigmoid, theta* the population minimizer. At theta*:

    H = E[ phi(theta*.x) phi(-theta*.x) x x^T ]  (Hessian of the population loss),
    G = E[ grad l grad l^T ],  grad l = -phi(-y theta*.x) y x.

Asymptotic normality: sqrt(n) (theta_hat_n - theta*) -> N(0, H^-1 G H^-1). A second-order Taylor
expansion of L around theta* (where grad L = 0) then gives

    E[ L(theta_hat_n) - L(theta*) ] ~ (1/2) E[ (theta_hat - theta*)^T H (theta_hat - theta*) ]
                                    ~ Tr(H^-1 G) / (2n).

If the model is well specified (P(y = 1 | x) = phi(theta*.x)), the information identity gives
G = H and the prediction is p / (2n); Tr(H^-1 G) / p measures the misspecification.

Result 3b (plug-in learning-curve prediction). The difference of the asymptotic excess risks at
n and m n labels is the expected improvement from collecting (m - 1) n more:

    E[ L(theta_hat_n) - L(theta_hat_mn) ] ~ Tr(H^-1 G) (1/n - 1/(mn)) / 2,

and with H, G replaced by their estimates at theta_hat_n on the same n labels (consistent:
theta_hat_n -> theta*, the empirical averages converge uniformly near theta*, and the trace of
H^-1 G is continuous where H is invertible) the prediction needs no population data.
"""

from __future__ import annotations

import numpy as np


def sandwich(H: np.ndarray, G: np.ndarray) -> np.ndarray:
    """H^-1 G H^-1: the asymptotic covariance of sqrt(n) (theta_hat - theta*)."""
    Hi = np.linalg.inv(H)
    return Hi @ G @ Hi


def trace_HinvG(H: np.ndarray, G: np.ndarray) -> float:
    return float(np.trace(np.linalg.solve(H, G)))


def predicted_excess(H: np.ndarray, G: np.ndarray, n) -> np.ndarray:
    """Tr(H^-1 G) / (2n)."""
    return trace_HinvG(H, G) / (2 * np.asarray(n, dtype=np.float64))


def misspecification_ratio(H: np.ndarray, G: np.ndarray) -> float:
    """Tr(H^-1 G) / p (= 1 when G = H)."""
    return trace_HinvG(H, G) / H.shape[0]


def predicted_improvement(H: np.ndarray, G: np.ndarray, n, factor: float = 4.0) -> np.ndarray:
    """Expected drop in excess risk from n to factor * n labels:
    Tr(H^-1 G) (1/n - 1/(factor n)) / 2.

    Derivation: E[L(theta_hat_n)] - L(theta*) ~ Tr(H^-1 G) / (2n) (asymptotic normality +
    second-order Taylor, see the module docstring) at both sizes; L(theta*) cancels in the
    difference, so E[L(theta_hat_n) - L(theta_hat_mn)] ~ Tr(H^-1 G) / (2n) - Tr(H^-1 G) / (2mn).
    For m = 4 this is 3 Tr(H^-1 G) / (8n). It predicts the expected improvement, not one run's:
    n (L(theta_hat_n) - L(theta*)) converges in distribution to the non-degenerate (1/2) Z^T H Z,
    Z ~ N(0, H^-1 G H^-1), so a single run's improvement fluctuates on the same 1/n scale. With
    H, G estimated at theta_hat_n from the same n labels this is the plug-in prediction.
    """
    n = np.asarray(n, dtype=np.float64)
    return trace_HinvG(H, G) / 2 * (1 / n - 1 / (factor * n))
