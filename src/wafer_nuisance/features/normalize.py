"""Feature sets for stage 2: PCA, standardization, intercept and norm clipping, all fixed from the
feature-fit pool before any training data are drawn.

For a feature set with raw features r (hand-crafted r in R^10, or CNN r in R^384):

1. PCA (CNN sets with k < 384 only): with mu, Sigma the feature-fit mean and covariance of r and
   U_k the eigenvectors of Sigma for its k largest eigenvalues, z = U_k^T (r - mu). Otherwise
   z = r.
2. Standardize: z_j <- (z_j - m_j) / s_j with feature-fit mean m_j and standard deviation s_j
   (s_j = 1 for a constant feature).
3. Append the intercept coordinate: v = (z, 1) in R^(k+1).
4. Clip: x = v * min(1, C / ||v||), with C the 99th percentile of ||v|| over the feature-fit pool.
   The intercept coordinate is part of v and of the norm.
5. Divide by C, so every feature vector satisfies ||x|| <= 1 (the Rademacher bound's C is then
   1; the raw C of each set is recorded). This is a fixed linear map, so it changes neither the
   linear class nor any 0-1 error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLIP_QUANTILE = 0.99


@dataclass(frozen=True)
class FeatureMap:
    name: str
    source: str  # "hand" or "cnn": which raw features it reads
    mu: np.ndarray | None  # PCA centre (None: no PCA)
    basis: np.ndarray | None  # (d, k) top-k eigenvectors as columns (None: no PCA)
    eigvals: np.ndarray | None  # all eigenvalues of the feature-fit covariance, descending
    m: np.ndarray  # (k,) standardization mean
    s: np.ndarray  # (k,) standardization std
    C: float  # clipping radius of v = (z, 1), before the final division by C

    @property
    def k(self) -> int:
        return len(self.m)

    @property
    def dim(self) -> int:
        """p = k + 1 parameters of a linear classifier (with intercept)."""
        return self.k + 1

    def unclipped(self, raw: np.ndarray) -> np.ndarray:
        """Steps 1-3: v = (standardized z, 1), shape (n, k + 1)."""
        z = np.asarray(raw, dtype=np.float64)
        if self.basis is not None:
            z = (z - self.mu) @ self.basis
        z = (z - self.m) / self.s
        return np.hstack([z, np.ones((len(z), 1))])

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        """Steps 1-5: (n, d_raw) raw features -> (n, k + 1) with every row norm <= 1."""
        v = self.unclipped(raw)
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        return v * np.minimum(1.0, self.C / norm) / self.C


def fit_feature_map(name: str, source: str, raw: np.ndarray, k: int | None = None
                    ) -> FeatureMap:
    """Fit steps 1-4 on the feature-fit pool's raw features (n, d). ``k`` < d applies PCA."""
    raw = np.asarray(raw, dtype=np.float64)
    mu = basis = eigvals = None
    z = raw
    if k is not None and k < raw.shape[1]:
        mu = raw.mean(axis=0)
        eigvals, vecs = np.linalg.eigh(np.cov(raw - mu, rowvar=False))
        order = np.argsort(eigvals)[::-1]
        eigvals, basis = eigvals[order], vecs[:, order[:k]]
        z = (raw - mu) @ basis
    m, s = z.mean(axis=0), z.std(axis=0)
    s = np.where(s > 1e-12 * max(1.0, np.abs(m).max()), s, 1.0)
    fm = FeatureMap(name, source, mu, basis, eigvals, m, s, C=1.0)
    C = float(np.quantile(np.linalg.norm(fm.unclipped(raw), axis=1), CLIP_QUANTILE))
    return FeatureMap(name, source, mu, basis, eigvals, m, s, C)


# name -> (raw source, PCA dimension or None)
FEATURE_SETS: dict[str, tuple[str, int | None]] = {
    "hand": ("hand", None),
    "pca4": ("cnn", 4),
    "pca16": ("cnn", 16),
    "pca64": ("cnn", 64),
    "cnn384": ("cnn", None),
}


def fit_feature_maps(fit_pool: dict[str, np.ndarray]) -> dict[str, FeatureMap]:
    """Every feature set of ``FEATURE_SETS`` from the feature-fit pool's raw arrays."""
    return {name: fit_feature_map(name, src, fit_pool[src], k)
            for name, (src, k) in FEATURE_SETS.items()}
