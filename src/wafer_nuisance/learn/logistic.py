"""Logistic regression for stage 2: norm-constrained (Result 2) and unregularized (Result 3).

Labels y in {-1, +1}, margin u = y w.x, logistic loss l(u) = ln(1 + e^(-u)), phi = sigmoid.

    empirical risk   L(w) = (1/n) sum_i l(y_i w.x_i)
    gradient         grad L(w) = -(1/n) sum_i phi(-y_i w.x_i) y_i x_i
    Hessian          H(w) = (1/n) sum_i phi(w.x_i) phi(-w.x_i) x_i x_i^T

Norm-constrained: minimize L(w) subject to ||w|| <= B by projected gradient descent with
Nesterov momentum: step 1/L_s with L_s = lambda_max(X^T X / n) / 4 >= the Lipschitz constant of
grad L, projection onto the ball = rescaling w to norm B when ||w|| > B; whenever the momentum
step increases the objective, momentum is reset and a plain projected step is taken instead.
Stops when the gradient mapping G(w) = L_s (w - P(w - grad L(w) / L_s)) has norm <= tol
(G = 0 exactly at the constrained minimizer).

Unregularized: Newton's method with backtracking line search. The empirical minimizer does not
exist when the training sample is linearly separable (the loss keeps decreasing along a
separating direction while ||w|| -> infinity); such runs are flagged (``separable=True``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, log_expit


def logistic_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    return float(-log_expit(y * (X @ w)).mean())


def logistic_grad(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -(expit(-y * (X @ w)) * y) @ X / len(y)


def logistic_hessian(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """H(w) = mean phi(w.x) phi(-w.x) x x^T (does not depend on the labels)."""
    t = X @ w
    return (X * (expit(t) * expit(-t))[:, None]).T @ X / len(X)


def gradient_outer(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """G(w) = mean grad l_i grad l_i^T with grad l_i = -phi(-y_i w.x_i) y_i x_i."""
    g = X * (expit(-y * (X @ w)) * y)[:, None]
    return g.T @ g / len(X)


def project_ball(w: np.ndarray, B: float) -> np.ndarray:
    """Euclidean projection onto {||w|| <= B}: rescale if outside."""
    nrm = np.linalg.norm(w)
    return w if nrm <= B else w * (B / nrm)


@dataclass(frozen=True)
class FitResult:
    w: np.ndarray
    loss: float  # final empirical logistic loss
    iterations: int
    converged: bool
    grad_norm: float  # norm of the gradient mapping (constrained) or gradient (unconstrained)
    separable: bool = False  # unregularized only: no finite minimizer (see module docstring)


def fit_logistic_ball(X: np.ndarray, y: np.ndarray, B: float, tol: float = 1e-7,
                      max_iter: int = 100_000, w0: np.ndarray | None = None) -> FitResult:
    """argmin of the empirical logistic loss over ||w|| <= B (see the module docstring)."""
    n = len(y)
    Ls = max(np.linalg.eigvalsh(X.T @ X / n)[-1] / 4, 1e-12)
    w = np.zeros(X.shape[1]) if w0 is None else project_ball(np.asarray(w0, float), B)
    z, t, f_prev = w.copy(), 1.0, logistic_loss(w, X, y)
    gm = np.inf
    for it in range(1, max_iter + 1):
        w_new = project_ball(z - logistic_grad(z, X, y) / Ls, B)
        f_new = logistic_loss(w_new, X, y)
        if f_new > f_prev:  # momentum overshot: restart it with a plain projected step from w
            t = 1.0
            w_new = project_ball(w - logistic_grad(w, X, y) / Ls, B)
            f_new = logistic_loss(w_new, X, y)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = w_new + ((t - 1) / t_new) * (w_new - w)
        w, t, f_prev = w_new, t_new, f_new
        if it % 10 == 0 or it == max_iter:
            gm = float(np.linalg.norm(Ls * (w - project_ball(w - logistic_grad(w, X, y) / Ls,
                                                             B))))
            if gm <= tol:
                return FitResult(w, f_prev, it, True, gm)
    return FitResult(w, f_prev, max_iter, False, gm)


def fit_logistic(X: np.ndarray, y: np.ndarray, tol: float = 1e-10, max_iter: int = 100,
                 max_norm: float = 1e4, separable_loss: float = 1e-6) -> FitResult:
    """Unregularized logistic regression by Newton's method with backtracking. Flags the run as
    separable if every training point ends up classified correctly with a near-zero loss, or
    ||w|| exceeds ``max_norm``, or Newton fails to converge (all signs of a minimizer at
    infinity)."""
    w = np.zeros(X.shape[1])
    f = logistic_loss(w, X, y)
    g = logistic_grad(w, X, y)
    for it in range(1, max_iter + 1):
        H = logistic_hessian(w, X)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        a, dec = 1.0, float(g @ step)
        while True:
            w_new = w - a * step
            f_new = logistic_loss(w_new, X, y)
            if f_new <= f - 1e-4 * a * dec or a < 1e-10:
                break
            a /= 2
        w, f = w_new, f_new
        g = logistic_grad(w, X, y)
        gn = float(np.linalg.norm(g))
        if np.linalg.norm(w) > max_norm:
            break
        if gn <= tol:
            sep = bool(f < separable_loss and np.all(y * (X @ w) > 0))
            return FitResult(w, f, it, True, gn, separable=sep)
    return FitResult(w, f, it, False, float(np.linalg.norm(g)), separable=True)


def zero_one_error(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Fraction of points with y w.x <= 0 (ties count as errors)."""
    return float(np.mean(y * (X @ w) <= 0))
