"""Result 1: generalization of threshold ERM, H = {x -> sign(x - theta)} on peak contrast.

H has VC dimension d = 1 (one point can be labelled both ways; two points x1 < x2 cannot get
(+1, -1)), so with d = 1 the VC bounds read:

* realizable (symmetrization + Sauer-Shelah; needs 0 < delta <= 1/2 and n >= d): with probability >=
  1 - delta, every ERM hypothesis h_hat satisfies

      L(h_hat) <= (2 / (n ln 2)) * ( ln(2 e n) + ln(2 / delta) );

* agnostic (uniform convergence via symmetrization, Hoeffding, Sauer-Shelah; n >= d): with
  probability >= 1 - delta,

      L(h_hat) - L(h*) <= sqrt( 32 * ( ln(2 e n) + ln(4 / delta) ) / n ).

The general forms have d ln(2en/d) in place of ln(2en). Rates: O~(1/n) realizable, O~(1/sqrt(n))
agnostic; 1/sqrt(n) is the worst case over distributions (the agnostic lower bound), so a particular
distribution may be learned faster.
"""

from __future__ import annotations

import math

import numpy as np


def _check(n, delta: float, d: int) -> np.ndarray:
    n = np.asarray(n, dtype=np.float64)
    if np.any(n < d) or not 0 < delta < 1:
        raise ValueError("need n >= d and 0 < delta < 1")
    return n


def realizable_bound(n, delta: float = 0.05, d: int = 1):
    """(2 / (n ln 2)) (d ln(2en/d) + ln(2/delta)) (realizable; requires delta <= 1/2)."""
    if delta > 0.5:
        raise ValueError("the realizable bound requires delta <= 1/2")
    n = _check(n, delta, d)
    return 2 / (n * math.log(2)) * (d * np.log(2 * math.e * n / d) + math.log(2 / delta))


def agnostic_bound(n, delta: float = 0.05, d: int = 1):
    """sqrt(32 (d ln(2en/d) + ln(4/delta)) / n) on L(h_hat) - L(h*) (agnostic)."""
    n = _check(n, delta, d)
    return np.sqrt(32 * (d * np.log(2 * math.e * n / d) + math.log(4 / delta)) / n)
