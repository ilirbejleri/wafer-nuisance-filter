import itertools

import numpy as np
import pytest

from wafer_nuisance.bounds import hoeffding_halfwidth, hoeffding_interval
from wafer_nuisance.learn import fit_threshold, threshold_errors


def brute_force_min_error(x, y):
    """Try theta at every sample value, between them, and outside the range."""
    cands = np.concatenate([[-np.inf, np.inf], x, x + 1e-9, x - 1e-9])
    return min(np.sum(np.where(x > t, 1, -1) != y) for t in cands)


@pytest.mark.parametrize("seed", range(30))
def test_threshold_erm_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 40))
    x = np.round(rng.normal(size=n), 1)  # rounding creates ties in x
    y = np.where(rng.random(n) < 1 / (1 + np.exp(-3 * x)), 1, -1)
    h = fit_threshold(x, y)
    assert np.sum(h.predict(x) != y) == brute_force_min_error(x, y)


def test_threshold_erm_exhaustive_small():
    x = np.array([0.0, 1.0, 1.0, 2.0])
    for labels in itertools.product([-1, 1], repeat=4):
        y = np.array(labels)
        assert np.sum(fit_threshold(x, y).predict(x) != y) == brute_force_min_error(x, y)


def test_realizable_threshold_is_midpoint():
    x = np.array([0.1, 0.4, 0.2, 0.9, 0.7])
    y = np.array([-1, -1, -1, 1, 1])
    h = fit_threshold(x, y)
    assert h.theta == pytest.approx(0.55) and h.error(x, y) == 0


def test_ties_take_the_lowest_cut_and_outer_cuts():
    thetas, errors = threshold_errors(np.array([1.0, 2.0]), np.array([1, -1]))
    np.testing.assert_array_equal(errors, [1, 2, 1])
    assert fit_threshold(np.array([1.0, 2.0]), np.array([1, -1])).theta == -np.inf
    assert fit_threshold(np.array([1.0, 2.0]), np.array([-1, -1])).theta == np.inf


def test_hoeffding_hand_computed():
    # sqrt(ln(40) / 200) = sqrt(3.68888 / 200) = 0.135810
    assert hoeffding_halfwidth(100, 0.05) == pytest.approx(0.1358104, abs=1e-6)
    assert hoeffding_interval(0.05, 100, 0.05) == pytest.approx((0.0, 0.1858104), abs=1e-6)
