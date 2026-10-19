import numpy as np
import pytest

from wafer_nuisance.learn import (
    fit_logistic,
    fit_logistic_ball,
    gradient_outer,
    logistic_grad,
    logistic_hessian,
    logistic_loss,
    project_ball,
    zero_one_error,
)


def _data(seed=0, n=400, p=5, noise=1.0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    X[:, -1] = 1.0
    X /= np.maximum(1.0, np.linalg.norm(X, axis=1, keepdims=True))
    w = rng.standard_normal(p) * 3
    y = np.where(rng.random(n) < 1 / (1 + np.exp(-(X @ w) / noise)), 1.0, -1.0)
    return X, y


def test_projection_onto_ball():
    rng = np.random.default_rng(0)
    for _ in range(100):
        w, B = rng.standard_normal(6) * rng.uniform(0.1, 5), rng.uniform(0.2, 4)
        p = project_ball(w, B)
        assert np.linalg.norm(p) <= B + 1e-12
        if np.linalg.norm(w) <= B:
            np.testing.assert_array_equal(p, w)
        else:  # closest point of the ball: on the sphere, along w
            assert np.linalg.norm(p) == pytest.approx(B)
            np.testing.assert_allclose(p / np.linalg.norm(p), w / np.linalg.norm(w))
            for _ in range(20):  # no point of the ball is closer
                q = project_ball(rng.standard_normal(6) * B, B)
                assert np.linalg.norm(w - p) <= np.linalg.norm(w - q) + 1e-12


def test_loss_range_is_exactly_BC():
    """For ||w|| <= B, ||x|| <= C the margin lies in [-BC, BC], so l lies in
    [ln(1 + e^-BC), ln(1 + e^BC)], an interval of length exactly BC (Delta_F = BC)."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        B, C = rng.uniform(0.1, 20), rng.uniform(0.1, 5)
        lo, hi = np.log1p(np.exp(-B * C)), np.log1p(np.exp(B * C))
        assert hi - lo == pytest.approx(B * C, rel=1e-12)
        w = project_ball(rng.standard_normal(4) * B, B)
        x = project_ball(rng.standard_normal(4) * C, C)
        y = rng.choice([-1.0, 1.0])
        loss = np.log1p(np.exp(-y * w @ x))
        assert lo - 1e-12 <= loss <= hi + 1e-12


def test_gradient_hessian_and_G_match_finite_differences():
    X, y = _data()
    rng = np.random.default_rng(2)
    w = rng.standard_normal(X.shape[1])
    eps = 1e-6
    E = np.eye(len(w))
    num_g = np.array([(logistic_loss(w + eps * e, X, y) - logistic_loss(w - eps * e, X, y))
                      / (2 * eps) for e in E])
    np.testing.assert_allclose(logistic_grad(w, X, y), num_g, atol=1e-8)
    num_h = np.array([(logistic_grad(w + eps * e, X, y) - logistic_grad(w - eps * e, X, y))
                      / (2 * eps) for e in E])
    np.testing.assert_allclose(logistic_hessian(w, X), num_h, atol=1e-7)
    # G = mean of outer products of per-example gradients (finite-difference gradients).
    per = np.array([[(logistic_loss(w + eps * e, X[i:i + 1], y[i:i + 1])
                      - logistic_loss(w - eps * e, X[i:i + 1], y[i:i + 1])) / (2 * eps)
                     for e in E] for i in range(len(y))])
    np.testing.assert_allclose(gradient_outer(w, X, y), per.T @ per / len(y), atol=1e-8)


def test_constrained_fit_is_optimal_on_the_ball():
    X, y = _data(3)
    for B in (0.5, 2.0, 50.0):
        r = fit_logistic_ball(X, y, B, tol=1e-9)
        assert r.converged and r.grad_norm <= 1e-9
        assert np.linalg.norm(r.w) <= B + 1e-9
        rng = np.random.default_rng(4)
        for _ in range(200):  # no feasible point does better
            v = project_ball(r.w + rng.standard_normal(len(r.w)) * 0.3, B)
            assert logistic_loss(v, X, y) >= r.loss - 1e-10
    # A large ball contains the unconstrained minimizer: same solution as Newton.
    u = fit_logistic(X, y)
    big = fit_logistic_ball(X, y, 10 * np.linalg.norm(u.w), tol=1e-10)
    np.testing.assert_allclose(big.w, u.w, atol=1e-6)


def test_unregularized_fit_and_separability_flag():
    X, y = _data(5)
    r = fit_logistic(X, y)
    assert r.converged and not r.separable and r.grad_norm <= 1e-10
    ys = np.sign(X @ np.array([1.0, -2.0, 0.5, 1.0, 0.3]))
    s = fit_logistic(X, ys)
    assert s.separable and zero_one_error(s.w, X, ys) == 0.0
