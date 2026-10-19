import numpy as np
import pytest

from wafer_nuisance.bounds import agnostic_bound, realizable_bound


def test_result_1_hand_values():
    # n = 1000, delta = 0.05: ln(2e * 1000) = 8.600902, ln(40) = 3.688879, ln(80) = 4.382027.
    assert realizable_bound(1000, 0.05) == pytest.approx(2 / (1000 * 0.6931472) * 12.289782,
                                                          rel=1e-6)
    assert realizable_bound(1000, 0.05) == pytest.approx(0.0354608, rel=1e-5)
    assert agnostic_bound(1000, 0.05) == pytest.approx(np.sqrt(32 * 12.982929 / 1000), rel=1e-6)
    assert agnostic_bound(1000, 0.05) == pytest.approx(0.6445570, rel=1e-6)


def test_result_1_shapes_and_conditions():
    n = np.array([16, 256, 4096])
    r, a = realizable_bound(n), agnostic_bound(n)
    assert r.shape == a.shape == (3,) and np.all(np.diff(r) < 0) and np.all(np.diff(a) < 0)
    with pytest.raises(ValueError):
        realizable_bound(100, delta=0.6)  # the realizable bound needs delta <= 1/2
    with pytest.raises(ValueError):
        agnostic_bound(0)


def test_result_4_hand_values_and_selection():
    from wafer_nuisance.bounds import certification_margin, certified_far, select_threshold

    # m = 1000, T = 50, delta = 0.05: ln(1000) = 6.907755 -> sqrt(6.907755 / 2000) = 0.0587697.
    assert certification_margin(1000, 50, 0.05) == pytest.approx(0.0587697, rel=1e-6)
    grid = np.array([0.1, 0.2, 0.3, 0.4])
    p_hat = np.array([0.5, 0.04, 0.0, 0.0])
    np.testing.assert_allclose(certified_far(p_hat, 1000), p_hat + certification_margin(1000, 4))
    assert select_threshold(grid, p_hat, m=1000, alpha=0.05) == 2  # 0.04 + 0.0546 > 0.05
    assert select_threshold(grid, p_hat, m=10, alpha=0.05) is None


def test_result_4_chernoff_hand_values_and_selection():
    from wafer_nuisance.bounds import (
        chernoff_certified_far,
        chernoff_eta,
        select_threshold_chernoff,
    )

    # m = 1000, T = 50, delta = alpha = 0.05: c = 2 ln(1000) / 1000 = 0.0138155,
    # eta = sqrt(c / 0.05) = 0.5256522, so the rule needs p_hat <= 0.4743478 * 0.05 = 0.0237174.
    assert chernoff_eta(1000, 50, 0.05, 0.05) == pytest.approx(0.5256522, rel=1e-6)
    p_hat = np.array([0.0, 0.01, 0.02] + [0.5] * 47)
    U = chernoff_certified_far(p_hat, 1000, 0.05)
    np.testing.assert_allclose(U[:3], [0.0138155, 0.0305413, 0.0449085], rtol=1e-5)
    grid = np.arange(50.0)
    assert select_threshold_chernoff(grid, p_hat[::-1], m=1000) == 47  # first p_hat = 0.02
    # The rule and the bound agree: certified iff U <= alpha.
    for ph in (0.0237, 0.0238):
        assert (ph <= (1 - chernoff_eta(1000, 50)) * 0.05) == bool(
            chernoff_certified_far(np.full(50, ph), 1000)[0] <= 0.05)
    # eta >= 1 below m = 2 ln(T/delta) / alpha = 276.3: nothing certifiable, even with p_hat = 0.
    assert chernoff_eta(276, 50) >= 1 > chernoff_eta(277, 50)
    assert select_threshold_chernoff(grid, np.zeros(50), m=276) is None
    assert select_threshold_chernoff(grid, np.zeros(50), m=277) == 0


def test_violation_rate_bound_hand_values():
    from wafer_nuisance.bounds import violation_rate_bound

    # v = 0, R = 200, delta = 0.05: ln(20) / 200 = 2.995732 / 200 = 0.0149787.
    assert violation_rate_bound(0, 200, 0.05) == pytest.approx(0.0149787, rel=1e-5)
    # v = 3: 3/200 + sqrt(ln(20) / 400) = 0.015 + sqrt(0.00748933) = 0.015 + 0.0865409.
    assert violation_rate_bound(3, 200, 0.05) == pytest.approx(0.1015409, rel=1e-6)
    # zero errors on 100,000 independent test points: ln(20) / 10^5 = 2.995732e-5.
    assert violation_rate_bound(0, 100_000, 0.05) == pytest.approx(2.995732e-5, rel=1e-6)
    assert violation_rate_bound(200, 200) == 1.0
    with pytest.raises(ValueError):
        violation_rate_bound(5, 4)


def test_result_2_hand_values():
    from wafer_nuisance.bounds import rademacher_bound, rademacher_complexity_bound, vc_bound

    # k = 10, n = 1000: 11 ln(2e 1000 / 11) = 11 * 6.203007 = 68.233079, ln(80) = 4.382027.
    assert vc_bound(0.1, 1000, 10) == pytest.approx(0.1 + np.sqrt(8 * 72.615106 / 1000),
                                                     rel=1e-6)
    assert vc_bound(0.1, 1000, 10) == pytest.approx(0.862182, rel=1e-6)
    # Four unit vectors, B = 2, C = 1: Rhat = (2/4) sqrt(4) = 1;
    # (0.5 + 2 * 1 + 3 * 2 * sqrt(ln(40) / 8)) / ln 2 = (2.5 + 4.074305) / 0.693147 = 9.484715.
    X = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.6, 0.8]])
    assert rademacher_complexity_bound(X, 2.0) == pytest.approx(1.0)
    assert rademacher_bound(0.5, X, B=2.0, C=1.0) == pytest.approx(9.484715, rel=1e-6)
    with pytest.raises(ValueError):
        rademacher_bound(0.5, 2 * X, B=2.0, C=1.0)


def test_result_2_clipped_hand_values():
    from wafer_nuisance.bounds import (
        clipped_logistic_loss,
        rademacher_bound,
        rademacher_bound_clipped,
    )

    # Margins u = y w.x = (-1, 0, 2, 10): psi = (ln 2, ln 2, ln(1 + e^-2), ln(1 + e^-10))
    # = (0.693147, 0.693147, 0.126928, 0.0000454); mean 0.378317.
    X = np.array([[-1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [10.0, 0.0]])
    y = np.ones(4)
    assert clipped_logistic_loss(np.array([1.0, 0.0]), X, y) == pytest.approx(0.378317, rel=1e-5)
    # Same four points as above, B = 2, C = 1, Rhat = 1:
    # (0.5 + 2 * 1 + 3 ln 2 sqrt(ln(40) / 8)) / ln 2 = (2.5 + 3 * 0.693147 * 0.679051) / 0.693147
    # = (2.5 + 1.412038) / 0.693147 = 5.643890. The deviation term no longer grows with B.
    X = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.6, 0.8]])
    assert rademacher_bound_clipped(0.5, X, B=2.0, C=1.0) == pytest.approx(5.643890, rel=1e-6)
    assert rademacher_bound_clipped(0.5, X, B=2.0, C=1.0) < rademacher_bound(0.5, X, 2.0, 1.0)
    with pytest.raises(ValueError):
        rademacher_bound_clipped(0.5, 2 * X, B=2.0, C=1.0)
    with pytest.raises(ValueError):
        rademacher_bound_clipped(0.8, X, B=2.0, C=1.0)  # psi <= ln 2


def test_result_3_hand_values():
    from wafer_nuisance.bounds import (
        misspecification_ratio,
        predicted_excess,
        sandwich,
        trace_HinvG,
    )

    H, G = np.diag([2.0, 1.0]), np.diag([2.0, 3.0])  # H^-1 G = diag(1, 3)
    assert trace_HinvG(H, G) == pytest.approx(4.0)
    assert predicted_excess(H, G, 10) == pytest.approx(0.2)
    assert misspecification_ratio(H, G) == pytest.approx(2.0)
    np.testing.assert_allclose(sandwich(H, G), np.diag([0.5, 3.0]))
    assert misspecification_ratio(H, H) == pytest.approx(1.0)  # well specified: p / (2n)


def test_result_3_predicted_improvement_hand_values():
    from wafer_nuisance.bounds import predicted_improvement

    H, G = np.diag([2.0, 1.0]), np.diag([2.0, 3.0])  # Tr(H^-1 G) = 4
    # 4 / 2 * (1/10 - 1/40) = 2 * 0.075 = 0.15 = 3 * 4 / (8 * 10)
    assert predicted_improvement(H, G, 10) == pytest.approx(0.15)
    # factor 2 at n = 20: 2 * (1/20 - 1/40) = 0.05
    assert predicted_improvement(H, G, 20, factor=2) == pytest.approx(0.05)
    np.testing.assert_allclose(predicted_improvement(np.eye(3), np.eye(3), [10, 100]),
                               [1.5 * 0.075, 1.5 * 0.0075])  # well specified: p / 2 = 1.5
