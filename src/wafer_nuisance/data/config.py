"""Generator regimes and the independent data pools.

A regime fixes the nuisance distribution and the defect distribution. The two regimes share the
nuisance distribution (so their nuisance candidates have the same distribution) and differ only
in the defect contrast range:

* ``separable``: every defect is strong enough that, on the population pool, one threshold on
  peak contrast separates real defects from nuisance with zero errors (verified, not assumed:
  ``experiments/stage1_report.py``);
* ``overlapping``: faint defects reach down into the nuisance range, so the best threshold errs.

Stage 1's threshold is tuned once, for recall on the overlapping regime's validation dies, and
used unchanged for both regimes: the inspection tool is set up for the faint defects it must
find, then the process is changed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from typing import Literal

import numpy as np

from wafer_nuisance.data.synthetic import DEFECT_KINDS, DefectKind, DefectSpec, NuisanceConfig


@dataclass(frozen=True)
class DefectDistribution:
    """Each die is defective with probability ``defect_prob`` and then carries one defect of a
    uniformly chosen kind. ``contrast`` ~ U[contrast_min, contrast_max] for every kind: the
    additive reflectance change for particles (random sign) and scratches (dark), the fill
    fraction for bridges, opens and missing contacts (see ``synthetic.DefectSpec``)."""

    contrast_min: float
    contrast_max: float = 1.0
    defect_prob: float = 0.5
    kinds: tuple[DefectKind, ...] = DEFECT_KINDS
    particle_radius: tuple[float, float] = (2.0, 5.0)  # px
    scratch_width: tuple[float, float] = (1.5, 3.0)  # px
    scratch_length: tuple[float, float] = (20.0, 60.0)  # px
    pattern_extent: tuple[float, float] = (3.0, 8.0)  # px, bridge / open length along the line
    missing_count: tuple[int, int] = (1, 3)  # adjacent contacts removed (inclusive)


def draw_defect_spec(rng: np.random.Generator, dist: DefectDistribution) -> DefectSpec:
    kind = dist.kinds[rng.integers(len(dist.kinds))]
    c = rng.uniform(dist.contrast_min, dist.contrast_max)
    if kind == "particle":
        return DefectSpec("particle", size=rng.uniform(*dist.particle_radius),
                          contrast=rng.choice([-1.0, 1.0]) * c,
                          aspect=rng.uniform(1.0, 2.0), irregularity=rng.uniform(0.0, 0.3))
    if kind == "scratch":
        return DefectSpec("scratch", size=rng.uniform(*dist.scratch_width), contrast=-c,
                          length=rng.uniform(*dist.scratch_length))
    if kind in ("bridge", "open"):
        return DefectSpec(kind, size=rng.uniform(*dist.pattern_extent), contrast=min(c, 1.0))
    lo, hi = dist.missing_count
    return DefectSpec("missing_contact", size=int(rng.integers(lo, hi + 1)), contrast=min(c, 1.0))


# Nuisance shared by both regimes (strengths chosen so that a defect-free die gives a few
# candidates at the recall-tuned threshold; see experiments/stage1_report.py).
NUISANCE = NuisanceConfig(
    shift_std=0.3,
    cd_bias_std=0.06,
    ler_std=0.03,
    warp_std=0.06,
    blob_rate=3.0,
    blob_amp_min=0.02,
    blob_amp_max=0.15,
    blob_min_gap=3.0,
)


@dataclass(frozen=True)
class Regime:
    name: str
    defects: DefectDistribution
    nuisance: NuisanceConfig = NUISANCE
    design_seed: int = 0  # one product layout for every pool


REGIMES: dict[str, Regime] = {
    "separable": Regime("separable", DefectDistribution(contrast_min=0.9, contrast_max=1.5)),
    "overlapping": Regime("overlapping", DefectDistribution(contrast_min=0.15, contrast_max=0.6)),
}
STAGE1_TARGET_RECALL = 0.99
STAGE1_REGIME = "overlapping"  # regime whose validation dies set the stage-1 threshold


@dataclass(frozen=True)
class PoolSpec:
    """An independent pool: item i of pool p is generated from (p.seed, p.name, i) only.

    ``unit`` "candidates": each item is one candidate drawn from a fresh die (``sampler``).
    ``unit`` "dies": each item is a whole die (all its candidates are scored).
    ``size`` is the default size (per repeat for train and certification).
    """

    name: str
    seed: int
    size: int
    unit: Literal["candidates", "dies"]
    defect_free: bool = False
    all_defective: bool = False  # every die carries a defect (stage-1 recall tuning)
    uses: str = ""


POOLS: dict[str, PoolSpec] = {p.name: p for p in (
    PoolSpec("reference", 1001, 32, "dies", defect_free=True, uses="stage-1 reference die"),
    PoolSpec("stage1_tune", 1002, 4000, "dies", all_defective=True,
             uses="stage-1 threshold: defective validation dies (own seed; tau must be fixed "
                  "before the validation candidates, which depend on it, are drawn)"),
    PoolSpec("feature_fit", 1003, 20_000, "candidates", uses="PCA basis, normalization, C"),
    PoolSpec("train", 1004, 2**15, "candidates", uses="training subsamples (per repeat)"),
    PoolSpec("validation", 1005, 10_000, "candidates", uses="B, score grid"),
    PoolSpec("population", 1006, 200_000, "candidates", uses="L(h*), theta*, H, G"),
    PoolSpec("test", 1007, 100_000, "candidates", uses="measured errors"),
    PoolSpec("certification", 1008, 8_000, "dies", defect_free=True,
             uses="Result 4 certification (per repeat)"),
    PoolSpec("far_test", 1009, 100_000, "dies", defect_free=True,
             uses="true die-level false-alarm rate for Result 4"),
    PoolSpec("stage1_check", 1010, 20_000, "dies", all_defective=True,
             uses="out-of-sample stage-1 recall (measurement only)"),
)}


def _jsonable(x):
    if is_dataclass(x):
        return {k: _jsonable(v) for k, v in asdict(x).items()}
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def config_hash(*objs, length: int = 12) -> str:
    """Stable short hash of dataclasses / plain values (cache keys)."""
    blob = json.dumps([_jsonable(o) for o in objs], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:length]


__all__ = ["DefectDistribution", "NUISANCE", "POOLS", "PoolSpec", "REGIMES", "Regime",
           "STAGE1_REGIME", "STAGE1_TARGET_RECALL", "config_hash", "draw_defect_spec"]
