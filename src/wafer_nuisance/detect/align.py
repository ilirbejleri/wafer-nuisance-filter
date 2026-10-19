"""Rigid die-to-die alignment: integer cross-correlation peak, then parabolic sub-pixel refinement.

Convention: a shift s = (sy, sx) means test(p) ~ ref(p - s), i.e. the test die is the reference
moved by s. ``warp`` applies s to the reference, so the difference image lives in the test die's
pixel grid (where its ground-truth mask lives).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def cross_correlation(test: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Circular cross-correlation C[s] = sum_p a(p) b(p - s) of the zero-mean images a, b,
    computed with the FFT (correlation theorem: C = IFFT(FFT(a) * conj(FFT(b))))."""
    a = test - test.mean()
    b = ref - ref.mean()
    return np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))))


def _parabola_offset(c_minus: float, c0: float, c_plus: float) -> float:
    """Vertex of the parabola through (-1, c_minus), (0, c0), (1, c_plus)."""
    denom = c_minus - 2 * c0 + c_plus
    if denom >= 0:  # not a strict maximum (flat or convex): no refinement
        return 0.0
    return float(np.clip((c_minus - c_plus) / (2 * denom), -0.5, 0.5))


def estimate_shift(test: np.ndarray, ref: np.ndarray, max_shift: int = 3) -> tuple[float, float]:
    """Shift s with test(p) ~ ref(p - s).

    1. Integer part: the argmax of the cross-correlation over |sy|, |sx| <= max_shift. The search
       window must be smaller than half the smallest pattern pitch, or a periodic pattern can lock
       onto the wrong period.
    2. Sub-pixel part: along each axis, fit a parabola through the peak and its two neighbours
       and move to its vertex.
    """
    c = cross_correlation(test, ref)
    h, w = c.shape
    r = np.arange(-max_shift, max_shift + 1)
    window = c[np.ix_(r % h, r % w)]
    iy, ix = np.unravel_index(np.argmax(window), window.shape)
    sy, sx = int(r[iy]), int(r[ix])
    c0 = c[sy % h, sx % w]
    dy = _parabola_offset(c[(sy - 1) % h, sx % w], c0, c[(sy + 1) % h, sx % w])
    dx = _parabola_offset(c[sy % h, (sx - 1) % w], c0, c[sy % h, (sx + 1) % w])
    return sy + dy, sx + dx


def warp(ref: np.ndarray, shift: tuple[float, float], order: int = 3) -> np.ndarray:
    """ref moved by ``shift``: out(p) = ref(p - shift), by spline interpolation of the given
    order (1 = bilinear, 3 = cubic). Pixels that come from outside the image copy the nearest
    edge pixel."""
    return ndimage.shift(ref, shift, order=order, mode="nearest")
