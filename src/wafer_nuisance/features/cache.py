"""Disk cache for everything that is expensive and deterministic: the tuned stage 1 and per-pool
candidate arrays.

Layout: ``cache/<kind>/<name>-<hash>/...npz`` where the hash covers every input that changes the
output (regime, stage-1 config and tau, pool identity, row-function name and version). Pools
are generated in chunks of ``CHUNK`` items by a process pool; a chunk is computed once and then
read from disk, so growing a pool (or asking for a prefix of it) never recomputes existing
items. Row functions run in worker processes; a row kind may also have a post-processing step that
runs on each finished chunk in the main process (the CNN, which uses the GPU).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np

from wafer_nuisance.data.config import POOLS, PoolSpec, Regime, config_hash
from wafer_nuisance.data.sampler import die_candidates, make_die, sample_candidate
from wafer_nuisance.data.synthetic import DEFECT_KINDS
from wafer_nuisance.detect.stage1 import (
    Stage1,
    Stage1Config,
    build_reference,
    choose_threshold,
    defect_peak,
    difference_image,
)

CACHE_DIR = Path(__file__).resolve().parents[3] / "cache"
CHUNK = 256


def _cached(path: Path, compute: Callable[[], dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    out = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    tmp.replace(path)
    return out


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


# ---- stage 1 ---------------------------------------------------------------------------------


def _defect_peak_job(args) -> float:
    regime, pool, cfg, reference, i = args
    die, _ = make_die(regime, pool, i)
    diff, _, _ = difference_image(die.image, reference, cfg)
    return defect_peak(diff, die.mask)


def tune_stage1(regime: Regime, cfg: Stage1Config, target_recall: float,
                n_tune: int | None = None, workers: int | None = None) -> tuple[Stage1, dict]:
    """Reference from the reference pool, then tau = the largest threshold with recall >=
    ``target_recall`` on the stage-1 tuning dies (all defective). Returns the tuned stage 1 and
    a record of what was used."""
    ref_pool, tune_pool = POOLS["reference"], POOLS["stage1_tune"]
    n_tune = n_tune or tune_pool.size
    key = config_hash(regime, cfg, ref_pool, tune_pool, target_recall, n_tune)

    def compute() -> dict[str, np.ndarray]:
        imgs = np.stack([make_die(regime, ref_pool, i)[0].image for i in range(ref_pool.size)])
        reference = build_reference(imgs, cfg)
        jobs = [(regime, tune_pool, cfg, reference, i) for i in range(n_tune)]
        with ProcessPoolExecutor(workers or default_workers()) as ex:
            peaks = np.array(list(ex.map(_defect_peak_job, jobs, chunksize=64)))
        tau, recall = choose_threshold(peaks, target_recall)
        return {"reference": reference, "tau": np.array(tau), "recall": np.array(recall),
                "defect_peaks": peaks}

    out = _cached(CACHE_DIR / "stage1" / key / "stage1.npz", compute)
    info = {"tau": float(out["tau"]), "tuning_recall": float(out["recall"]),
            "target_recall": target_recall, "n_tune_dies": n_tune,
            "n_reference_dies": ref_pool.size, "cache_key": key}
    return Stage1(cfg, out["reference"], float(out["tau"])), info


# ---- pools ---------------------------------------------------------------------------------

# A row function maps (regime, stage1, pool, index) to a dict of fixed-shape arrays for item
# ``index`` of the pool. Bump its version when its output changes, so stale caches are not reused.
RowFn = Callable[[Regime, Stage1, PoolSpec, int], dict[str, np.ndarray]]
META_FIELDS = ("index", "attempts", "n_candidates", "label", "die_defective", "defect_found",
               "defect_kind", "defect_contrast", "peak", "area")


def meta_row(regime: Regime, stage1: Stage1, pool: PoolSpec, i: int) -> dict[str, np.ndarray]:
    """Scalar description of candidate i (one candidate from a fresh die)."""
    c = sample_candidate(regime, stage1, pool, i)
    return {f: np.asarray(getattr(c, f)) for f in META_FIELDS}


def recall_row(regime: Regime, stage1: Stage1, pool: PoolSpec, i: int) -> dict[str, np.ndarray]:
    """Die i of an all-defective die pool: is its defect found by stage 1?"""
    die, spec = make_die(regime, pool, i)
    diff, _, _ = difference_image(die.image, stage1.reference, stage1.cfg)
    peak = defect_peak(diff, die.mask)
    return {"defect_peak": np.asarray(peak), "found": np.asarray(peak > stage1.tau),
            "defect_kind": np.asarray(DEFECT_KINDS.index(spec.kind)),
            "defect_contrast": np.asarray(abs(spec.contrast))}


def features_row(regime: Regime, stage1: Stage1, pool: PoolSpec, i: int
                 ) -> dict[str, np.ndarray]:
    """Meta fields, the hand-crafted features and the crops the CNN step needs."""
    c = sample_candidate(regime, stage1, pool, i, features=True)
    return {**{f: np.asarray(getattr(c, f)) for f in META_FIELDS}, "hand": c.hand,
            "crop_test": c.crops["test"], "crop_reference": c.crops["reference"],
            "crop_component": c.crops["component"]}


def cnn_post(chunk: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Replace the crops by the 384-dim CNN features (main process), stored as float16 to keep
    the multi-million-candidate pools on disk small; the stored values are the features."""
    from wafer_nuisance.features.cnn import cnn_features

    crops = {k: chunk.pop(k) for k in ("crop_test", "crop_reference", "crop_component")}
    chunk["cnn"] = cnn_features(crops["crop_test"], crops["crop_reference"],
                                crops["crop_component"]).astype(np.float16)
    return chunk


def die_score_row(regime: Regime, stage1: Stage1, pool: PoolSpec, i: int, scorer
                  ) -> dict[str, np.ndarray]:
    """Die i of a die pool scored by a frozen stage-2 scorer ``(feature_map, w)`` on the
    hand-crafted features: the die's maximum candidate score (-inf if it has no candidates, so
    it is never flagged), its candidate count, and whether it carries a defect that some
    candidate overlaps."""
    from wafer_nuisance.features.handcrafted import handcrafted_features

    fm, w = scorer
    die, res = die_candidates(regime, stage1, pool, i)
    n = res.comps.count
    if n == 0:
        top = -np.inf
    else:
        raw = np.stack([handcrafted_features(die.image, stage1, res, k) for k in range(n)])
        top = float((fm(raw) @ w).max())
    return {"max_score": np.asarray(top), "n_candidates": np.asarray(n),
            "die_defective": np.asarray(bool(die.mask.any())),
            "defect_found": np.asarray(bool(res.is_defect.any()))}


# name -> (row function, version, pool unit it applies to, main-process post step or None)
ROWS: dict[str, tuple[RowFn, int, str, Callable | None]] = {
    "meta": (meta_row, 1, "candidates", None),
    "recall": (recall_row, 1, "dies", None),
    "features": (features_row, 2, "candidates", cnn_post),
    "die_score": (die_score_row, 1, "dies", None),
}


def _chunk_job(args) -> dict[str, np.ndarray]:
    regime, stage1, pool, rows, start, stop, *payload = args
    fn = ROWS[rows][0]
    extra = tuple(payload[:1]) if payload and payload[0] is not None else ()
    out = [fn(regime, stage1, pool, i, *extra) for i in range(start, stop)]
    return {k: np.stack([r[k] for r in out]) for k in out[0]}


def payload_hash(payload) -> str:
    """Hash of a row parameter made of arrays and plain values (exact bytes of every array)."""
    h = hashlib.sha1()
    for x in payload if isinstance(payload, tuple) else (payload,):
        if is_dataclass(x):
            x = tuple(getattr(x, f.name) for f in fields(x))
            h.update(payload_hash(x).encode())
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype, x.shape)).encode() + np.ascontiguousarray(x).tobytes())
        else:
            h.update(repr(x).encode())
    return h.hexdigest()[:12]


def pool_arrays(regime: Regime, stage1: Stage1, pool: PoolSpec, n: int, rows: str = "meta",
                start: int = 0, workers: int | None = None, payload=None
                ) -> dict[str, np.ndarray]:
    """Row arrays for items start, ..., start + n - 1 of ``pool`` (cached per chunk).
    ``payload`` is an extra argument for row functions that take one (e.g. a frozen scorer);
    its exact contents are part of the cache key."""
    _, version, unit, post = ROWS[rows]
    if pool.unit != unit:
        raise ValueError(f"rows {rows!r} need a {unit} pool; {pool.name} is a {pool.unit} pool")
    # Item i depends on the pool's identity, not its default size.
    ident = (pool.name, pool.seed, pool.defect_free, pool.all_defective)
    # The payload hash is appended only when there is one, so payload-free keys are unchanged.
    extra = () if payload is None else (payload_hash(payload),)
    key = config_hash(regime, stage1.cfg, stage1.tau, ident, rows, version, *extra)
    base = CACHE_DIR / "pools" / f"{regime.name}-{pool.name}-{rows}-{key}"
    c0, c1 = start // CHUNK, (start + n - 1) // CHUNK
    paths = {c: base / f"chunk{c:06d}.npz" for c in range(c0, c1 + 1)}
    missing = [c for c, p in paths.items() if not p.exists()]
    if missing:
        jobs = [(regime, stage1, pool, rows, c * CHUNK, (c + 1) * CHUNK, payload)
                for c in missing]
        with ProcessPoolExecutor(workers or default_workers()) as ex:
            for c, out in zip(missing, ex.map(_chunk_job, jobs), strict=True):
                _cached(paths[c], lambda out=out: post(out) if post else out)
    parts = [_cached(paths[c], dict) for c in range(c0, c1 + 1)]
    full = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    lo = start - c0 * CHUNK
    return {k: v[lo : lo + n] for k, v in full.items()}


# ---- feature sets --------------------------------------------------------------------------


def pool_features(regime: Regime, stage1: Stage1, pool: PoolSpec, n: int, start: int = 0,
                  workers: int | None = None) -> dict[str, np.ndarray]:
    """Meta fields, ``hand`` (n, 10) and ``cnn`` (n, 384) raw features of candidates start, ...,
    start + n - 1 of a candidate pool."""
    return pool_arrays(regime, stage1, pool, n, "features", start, workers)


def feature_maps(regime: Regime, stage1: Stage1, n_fit: int | None = None,
                 workers: int | None = None) -> tuple[dict, dict[str, np.ndarray]]:
    """Every feature set's map (PCA, standardization, clipping), fitted on the feature-fit pool
    only. Returns (maps, the feature-fit pool's arrays)."""
    from wafer_nuisance.features.normalize import fit_feature_maps

    pool = POOLS["feature_fit"]
    fit = pool_features(regime, stage1, pool, n_fit or pool.size, workers=workers)
    return fit_feature_maps(fit), fit
