"""One candidate per fresh die: the sampling unit of every training, validation and test example.

Candidates from one die are dependent (they share the die's nuisance draw), so an example is
never "a candidate" but "one candidate drawn uniformly at random from a freshly generated die":

    for attempt a = 0, 1, 2, ...:
        draw a fresh die D_a (nuisance; with probability defect_prob one defect)
        run stage 1 -> candidate set C(D_a)
        if C(D_a) is non-empty: pick J uniformly from C(D_a) and stop
        (otherwise redraw; the number of redraws is recorded)

Induced distribution. Write D for a die from the regime and C(D) for its candidates. The example
(x, y) = (features of J, label of J) has distribution

    Q(A) = E[ |C(D) ∩ A| / |C(D)|  ;  |C(D)| >= 1 ] / P(|C(D)| >= 1),

i.e. the die is drawn from its distribution conditioned on having at least one candidate, and
the candidate is uniform within that die. Q weighs dies equally, not candidates: a candidate on a
die with many candidates is less likely to be chosen than one on a die with a single candidate,
so Q is not the distribution of a random element of the stream of all candidates. Examples with
different indices come from disjoint, independently seeded dies, so they are i.i.d. draws from
Q; every error rate in this project is a probability under Q.

Seeding: die (pool p, index i, attempt a) is keyed by (p.seed, f"{p.name}/a{a}", i) only, so item
i never depends on how many items are requested, and pools with different seeds are independent.
The two regimes use the same keys: die i of a pool has the same nuisance, defect presence, kind
and placement in both, and only the defect contrast differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from wafer_nuisance.data.config import PoolSpec, Regime, draw_defect_spec
from wafer_nuisance.data.synthetic import (
    DEFECT_KINDS,
    DefectSpec,
    Die,
    DieDesign,
    SyntheticWafer,
    make_layout_design,
)
from wafer_nuisance.detect.stage1 import Stage1, Stage1Result, crop
from wafer_nuisance.features.handcrafted import handcrafted_features
from wafer_nuisance.utils.seeding import item_rng

MAX_ATTEMPTS = 1000


@lru_cache(maxsize=4)
def design_for(seed: int) -> DieDesign:
    return make_layout_design(item_rng(seed, "design", 0))


def make_die(regime: Regime, pool: PoolSpec, index: int, attempt: int = 0
             ) -> tuple[Die, DefectSpec | None]:
    """Fresh die (pool, index, attempt) and the defect spec it carries (None if defect-free)."""
    wafer = SyntheticWafer(design_for(regime.design_seed), regime.nuisance, pool.seed)
    stream = f"{pool.name}/a{attempt}"
    rng = item_rng(pool.seed, f"{stream}/spec", index)
    defective = pool.all_defective or rng.random() < regime.defects.defect_prob
    if defective and not pool.defect_free:
        spec = draw_defect_spec(rng, regime.defects)
        return wafer.defective_die(index, spec, stream), spec
    return wafer.normal_die(index, stream), None


@dataclass(frozen=True)
class Candidate:
    """One example: a candidate picked uniformly from the first die with >= 1 candidate."""

    pool: str
    index: int
    attempts: int  # dies drawn; attempts - 1 had zero candidates and were redrawn
    n_candidates: int  # |C(D)| on the die the candidate came from
    label: int  # +1 real defect, -1 nuisance (label rule: overlaps the defect mask)
    die_defective: bool
    defect_found: bool  # some candidate on the die overlaps the defect mask
    defect_kind: int  # index into DEFECT_KINDS, -1 if the die is defect-free
    defect_contrast: float  # nan if defect-free
    peak: float  # peak contrast: max of the difference image over the component
    area: int
    centroid: tuple[float, float]
    crops: dict[str, np.ndarray]  # test, reference, diff (float32), component (bool)
    hand: np.ndarray | None  # (10,) ``features.handcrafted.HAND_FEATURES`` (None if not asked)


def candidate_crops(die: Die, res: Stage1Result, k: int, size: int) -> dict[str, np.ndarray]:
    """Fixed size x size windows centred on component k's centroid."""
    c = res.comps.centroid[k]
    return {
        "test": crop(die.image, c, size).astype(np.float32),
        "reference": crop(res.ref_aligned, c, size).astype(np.float32),
        "diff": crop(res.diff, c, size).astype(np.float32),
        "component": crop(res.comps.labels == k + 1, c, size, mode="constant"),
    }


def sample_candidate(regime: Regime, stage1: Stage1, pool: PoolSpec, index: int,
                     features: bool = False) -> Candidate:
    """Example ``index`` of a candidate pool (see the module docstring for its distribution).
    ``features``: also compute the hand-crafted features (they do not change the draw)."""
    for attempt in range(MAX_ATTEMPTS):
        die, spec = make_die(regime, pool, index, attempt)
        res = stage1.run(die.image, die.mask)
        n = res.comps.count
        if n == 0:
            continue
        k = int(item_rng(pool.seed, f"{pool.name}/a{attempt}/pick", index).integers(n))
        return Candidate(
            pool=pool.name, index=index, attempts=attempt + 1, n_candidates=n,
            label=1 if res.is_defect[k] else -1,
            die_defective=spec is not None,
            defect_found=bool(res.is_defect.any()),
            defect_kind=DEFECT_KINDS.index(spec.kind) if spec else -1,
            defect_contrast=abs(spec.contrast) if spec else float("nan"),
            peak=float(res.comps.peak[k]), area=int(res.comps.area[k]),
            centroid=(float(res.comps.centroid[k, 0]), float(res.comps.centroid[k, 1])),
            crops=candidate_crops(die, res, k, stage1.cfg.crop),
            hand=handcrafted_features(die.image, stage1, res, k) if features else None,
        )
    raise RuntimeError(f"{pool.name}[{index}]: no candidates in {MAX_ATTEMPTS} dies")


def die_candidates(regime: Regime, stage1: Stage1, pool: PoolSpec, index: int
                   ) -> tuple[Die, Stage1Result]:
    """Die ``index`` of a die pool with all its candidates (possibly none: then the die cannot be
    flagged). Used for die-level false-alarm rates, where every candidate is scored."""
    die, _ = make_die(regime, pool, index)
    return die, stage1.run(die.image, die.mask)
