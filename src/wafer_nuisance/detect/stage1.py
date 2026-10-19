"""Stage 1 (computer vision, no learning): reference die -> difference image -> candidates.

1. Reference: the mean of the reference-pool dies, each aligned to the first one. Averaging m
   dies divides the reference's sensor noise variance by m.
2. Difference image: align the test die to the reference (``align.estimate_shift``), move the
   reference onto the test die's pixel grid, match its gain and offset to the test die by least
   squares, take |test - reference| and smooth it with a Gaussian. A ``border``-pixel frame is
   set to zero (pixels there may come from outside the reference after the shift).
3. Candidates: threshold the difference image at tau, label the 8-connected components. Each
   component is one candidate; its crop is a fixed window centred on the component centroid.

tau is chosen on validation dies for high recall (``choose_threshold``), so most candidates are
nuisance. Label rule (fixed): a candidate is a real defect iff its component overlaps the
ground-truth defect mask in at least one pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from wafer_nuisance.detect.align import estimate_shift, warp

EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class Stage1Config:
    max_shift: int = 3  # px, integer search window for alignment (< half the smallest pitch)
    interp_order: int = 3  # spline order used to move the reference (1 bilinear, 3 cubic)
    smooth_sigma: float = 1.0  # px, Gaussian smoothing of |test - reference|
    border: int = 4  # px, frame of the difference image that is zeroed
    crop: int = 32  # px, side of the fixed crop window centred on each candidate


def build_reference(images: np.ndarray, cfg: Stage1Config) -> np.ndarray:
    """Mean of ``images`` (m, H, W), each moved onto the grid of images[0]."""
    base = images[0].astype(np.float64)
    acc = base.copy()
    for img in images[1:]:
        s = estimate_shift(img, base, cfg.max_shift)
        acc += warp(img.astype(np.float64), (-s[0], -s[1]), cfg.interp_order)
    return acc / len(images)


def photometric_match(test: np.ndarray, ref: np.ndarray, border: int) -> np.ndarray:
    """a * ref + b with (a, b) the least-squares fit of test on ref over the interior, which
    removes a global illumination gain/offset difference between the two dies."""
    sl = (slice(border, -border), slice(border, -border)) if border else (slice(None),) * 2
    x, y = ref[sl].ravel(), test[sl].ravel()
    a = np.cov(x, y, bias=True)[0, 1] / x.var()
    b = y.mean() - a * x.mean()
    return a * ref + b


def difference_image(test: np.ndarray, reference: np.ndarray, cfg: Stage1Config
                     ) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """(smoothed |test - aligned, photometrically matched reference|, that reference, shift)."""
    test = test.astype(np.float64)
    s = estimate_shift(test, reference, cfg.max_shift)
    ref_aligned = photometric_match(test, warp(reference, s, cfg.interp_order), cfg.border)
    diff = ndimage.gaussian_filter(np.abs(test - ref_aligned), cfg.smooth_sigma)
    b = cfg.border
    if b:
        diff[:b], diff[-b:], diff[:, :b], diff[:, -b:] = 0, 0, 0, 0
    return diff, ref_aligned, s


@dataclass(frozen=True)
class Components:
    """Connected components of {diff > tau}. ``labels`` is the label image (0 = background,
    k = component k), arrays below are indexed by k - 1."""

    labels: np.ndarray
    count: int
    area: np.ndarray  # pixels
    peak: np.ndarray  # max of the difference image over the component ("peak contrast")
    centroid: np.ndarray  # (count, 2) row, col


def connected_components(diff: np.ndarray, tau: float) -> Components:
    labels, n = ndimage.label(diff > tau, structure=EIGHT_CONNECTED)
    if n == 0:
        return Components(labels, 0, np.zeros(0, int), np.zeros(0), np.zeros((0, 2)))
    rows, cols = np.nonzero(labels)  # foreground pixels only: a small part of the die
    lab = labels[rows, cols]
    area = np.bincount(lab, minlength=n + 1)[1:]
    peak = np.zeros(n + 1)
    np.maximum.at(peak, lab, diff[rows, cols])
    centroid = np.stack([np.bincount(lab, rows, n + 1)[1:], np.bincount(lab, cols, n + 1)[1:]],
                        axis=1) / area[:, None]
    return Components(labels, n, area, peak[1:], centroid)


def component_is_defect(comps: Components, mask: np.ndarray) -> np.ndarray:
    """Label rule: component k is a real defect iff it overlaps ``mask`` in >= 1 pixel."""
    hits = np.bincount(comps.labels[mask], minlength=comps.count + 1)[1:]
    return hits > 0


def defect_peak(diff: np.ndarray, mask: np.ndarray) -> float:
    """max of the difference image over the defect mask. Stage 1 finds the defect (some component
    overlaps the mask) iff this exceeds tau, so recall(tau) = P(defect_peak > tau)."""
    return float(diff[mask].max()) if mask.any() else float("nan")


def choose_threshold(defect_peaks: np.ndarray, target_recall: float) -> tuple[float, float]:
    """Largest tau with recall >= target_recall on these defective dies: sort the peaks, allow
    the k = floor((1 - target) * N) smallest to be missed, and put tau midway between the k-th
    and (k+1)-th smallest. Returns (tau, achieved recall)."""
    p = np.sort(np.asarray(defect_peaks, dtype=np.float64))
    n = len(p)
    k = int(np.floor((1 - target_recall) * n + 1e-9))
    tau = p[0] / 2 if k == 0 else (p[k - 1] + p[k]) / 2
    return float(tau), float(np.mean(p > tau))


def crop(img: np.ndarray, center: tuple[float, float], size: int, mode: str = "reflect"
         ) -> np.ndarray:
    """size x size window whose centre pixel is the rounded ``center``; padded at the borders."""
    half = size // 2
    r, c = int(round(center[0])), int(round(center[1]))
    padded = np.pad(img, half, mode=mode)
    return padded[r : r + size, c : c + size]


@dataclass(frozen=True)
class Stage1Result:
    diff: np.ndarray  # smoothed difference image
    ref_aligned: np.ndarray  # reference on the test die's grid, photometrically matched
    shift: tuple[float, float]
    comps: Components
    is_defect: np.ndarray  # (count,) bool, label rule applied to the ground-truth mask


@dataclass(frozen=True)
class Stage1:
    """A tuned stage 1: config, reference image and threshold tau (all fixed before any
    candidate is drawn)."""

    cfg: Stage1Config
    reference: np.ndarray
    tau: float

    def run(self, image: np.ndarray, mask: np.ndarray) -> Stage1Result:
        diff, ref_aligned, s = difference_image(image, self.reference, self.cfg)
        comps = connected_components(diff, self.tau)
        return Stage1Result(diff, ref_aligned, s, comps, component_is_defect(comps, mask))
