"""Hand-crafted candidate features (k = 10), computed on the full die so no component is cut off
by the crop window.

Notation: D = test - reference (signed, after alignment and gain/offset match), |D|_s = the smoothed
absolute difference stage 1 thresholds, S = the candidate's component (pixels), p = its peak pixel.

==================  ============================================================================
feature             definition
==================  ============================================================================
log_area            ln |S|
peak_contrast       max_S |D|_s (Result 1's scalar feature; the threshold classifier uses it)
mean_contrast       mean_S |D|_s
signed_mean         mean_S of D smoothed like |D|_s (polarity: brighter or darker than the
                    reference)
eccentricity        sqrt(1 - l2/l1), l1 >= l2 the eigenvalues of the covariance of S's pixel
                    coordinates (0 for a disc, -> 1 for a line)
bbox_aspect         long / short side of S's bounding box (>= 1)
gradient_energy     mean_S |grad D_s|^2, central differences, D_s = D smoothed like |D|_s
log_small           max_S |sigma^2 LoG_sigma(D)| at sigma = 1.5 px (scale-normalised
log_large           Laplacian of Gaussian, the blob detector) and at sigma = 3 px
edge_distance       Euclidean distance (px) from p to the nearest pattern edge of the reference
                    die; pattern edges are the half-intensity contour of the reference (pixels
                    where reference - (min + max)/2 changes sign towards a neighbour)
==================  ============================================================================
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy import ndimage

from wafer_nuisance.detect.stage1 import Stage1, Stage1Result

HAND_FEATURES = ("log_area", "peak_contrast", "mean_contrast", "signed_mean", "eccentricity",
                 "bbox_aspect", "gradient_energy", "log_small", "log_large", "edge_distance")
LOG_SIGMAS = (1.5, 3.0)  # px
MARGIN = 12  # px around the component's bounding box: >= 3 * the largest LoG sigma + 2


def pattern_edges(reference: np.ndarray) -> np.ndarray:
    """Half-intensity contour of the reference: bool map of pixels whose sign of
    reference - (min + max) / 2 differs from a 4-neighbour's."""
    s = reference > (reference.min() + reference.max()) / 2
    e = np.zeros_like(s)
    e[:-1] |= s[:-1] != s[1:]
    e[1:] |= s[1:] != s[:-1]
    e[:, :-1] |= s[:, :-1] != s[:, 1:]
    e[:, 1:] |= s[:, 1:] != s[:, :-1]
    return e


_EDGE_DISTANCE: dict[str, np.ndarray] = {}


def edge_distance_map(reference: np.ndarray) -> np.ndarray:
    """Distance (px) of every pixel of the reference grid to its nearest pattern edge (cached per
    reference image)."""
    key = hashlib.sha1(np.ascontiguousarray(reference).tobytes()).hexdigest()
    if key not in _EDGE_DISTANCE:
        _EDGE_DISTANCE.clear()
        _EDGE_DISTANCE[key] = ndimage.distance_transform_edt(~pattern_edges(reference))
    return _EDGE_DISTANCE[key]


def handcrafted_features(test: np.ndarray, stage1: Stage1, res: Stage1Result, k: int
                         ) -> np.ndarray:
    """The 10 features of ``HAND_FEATURES`` for component k (0-based) of a stage-1 result."""
    rows, cols = np.nonzero(res.comps.labels == k + 1)
    h, w = test.shape
    r0, r1 = max(rows.min() - MARGIN, 0), min(rows.max() + MARGIN + 1, h)
    c0, c1 = max(cols.min() - MARGIN, 0), min(cols.max() + MARGIN + 1, w)
    rr, cc = rows - r0, cols - c0  # component pixels in window coordinates

    d = test[r0:r1, c0:c1].astype(np.float64) - res.ref_aligned[r0:r1, c0:c1]
    d_s = ndimage.gaussian_filter(d, stage1.cfg.smooth_sigma)
    absd = res.diff[rows, cols]
    gy, gx = np.gradient(d_s)
    logs = [np.abs(s**2 * ndimage.gaussian_laplace(d, s)[rr, cc]).max() for s in LOG_SIGMAS]

    area = len(rows)
    if area > 1:
        l2, l1 = np.linalg.eigvalsh(np.cov(np.stack([rows, cols]).astype(np.float64)))
        ecc = float(np.sqrt(max(0.0, 1.0 - l2 / l1))) if l1 > 0 else 0.0
    else:
        ecc = 0.0
    bh, bw = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1

    # Peak pixel on the test grid -> reference grid: test(p) ~ ref(p - shift).
    j = int(np.argmax(absd))
    sy, sx = res.shift
    py = int(np.clip(round(rows[j] - sy), 0, h - 1))
    px = int(np.clip(round(cols[j] - sx), 0, w - 1))

    return np.array([
        np.log(area),
        absd.max(),
        absd.mean(),
        d_s[rr, cc].mean(),
        ecc,
        max(bh, bw) / min(bh, bw),
        (gy[rr, cc] ** 2 + gx[rr, cc] ** 2).mean(),
        logs[0],
        logs[1],
        edge_distance_map(stage1.reference)[py, px],
    ])
