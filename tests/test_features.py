import numpy as np
import pytest

from wafer_nuisance.detect.stage1 import Stage1, Stage1Config, Stage1Result, connected_components
from wafer_nuisance.features.cnn import footprint_cells
from wafer_nuisance.features.handcrafted import HAND_FEATURES, handcrafted_features, pattern_edges
from wafer_nuisance.features.normalize import fit_feature_map


def _result(test, ref, tau=0.05):
    diff = np.abs(test - ref)
    comps = connected_components(diff, tau)
    return Stage1Result(diff, ref, (0.0, 0.0), comps, np.zeros(comps.count, bool))


def test_handcrafted_features_on_hand_built_components():
    ref = np.full((64, 64), 0.2)
    ref[:, 40:] = 0.8  # one vertical pattern edge between columns 39 and 40
    test = ref.copy()
    test[10:13, 10:13] += 0.5  # 3 x 3 square, bright
    test[30, 5:25] -= 0.3  # 1 x 20 line, dark
    s1 = Stage1(Stage1Config(smooth_sigma=0.5), ref, 0.05)
    res = _result(test, ref)
    assert res.comps.count == 2
    f = {k: dict(zip(HAND_FEATURES, handcrafted_features(test, s1, res, k), strict=True))
         for k in range(2)}
    sq, ln = (f[0], f[1]) if f[0]["log_area"] == pytest.approx(np.log(9)) else (f[1], f[0])
    assert ln["log_area"] == pytest.approx(np.log(20))
    assert sq["peak_contrast"] == pytest.approx(0.5) and sq["mean_contrast"] == pytest.approx(0.5)
    assert sq["signed_mean"] > 0 > ln["signed_mean"]
    assert sq["eccentricity"] == pytest.approx(0.0, abs=1e-9) and ln["eccentricity"] == 1.0
    assert sq["bbox_aspect"] == 1.0 and ln["bbox_aspect"] == 20.0
    # Distance from the peak pixel to the edge contour (columns 39 and 40 are edge pixels).
    assert sq["edge_distance"] == pytest.approx(39 - 11, abs=1.0)
    assert sq["gradient_energy"] > 0 and sq["log_small"] > 0 and sq["log_large"] > 0


def test_pattern_edges_are_the_half_intensity_contour():
    ref = np.zeros((8, 8))
    ref[:, 4:] = 1.0
    e = pattern_edges(ref)
    assert e[:, 3].all() and e[:, 4].all() and e.sum() == 16


def test_footprint_cells():
    comp = np.zeros((2, 32, 32), bool)
    comp[0, 5, 9] = True  # cell (1, 2)
    cells = footprint_cells(comp)
    assert cells.shape == (2, 8, 8) and cells[0].sum() == 1 and cells[0, 1, 2]
    assert cells[1].all()  # empty footprint -> every cell


def test_feature_map_standardizes_clips_and_bounds_norm():
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((5000, 12)) @ rng.standard_normal((12, 12)) + 3.0
    fm = fit_feature_map("t", "cnn", raw, k=4)
    v = fm.unclipped(raw)
    assert v.shape == (5000, 5) and np.all(v[:, -1] == 1.0)
    np.testing.assert_allclose(v[:, :-1].mean(0), 0, atol=1e-9)
    np.testing.assert_allclose(v[:, :-1].std(0), 1, atol=1e-9)
    x = fm(raw)
    norms = np.linalg.norm(x, axis=1)
    assert norms.max() <= 1 + 1e-12
    assert np.mean(norms >= 1 - 1e-12) == pytest.approx(0.01, abs=0.002)  # 99th percentile C
    # PCA basis: top eigenvectors of the covariance, orthonormal, eigenvalues descending.
    np.testing.assert_allclose(fm.basis.T @ fm.basis, np.eye(4), atol=1e-10)
    assert np.all(np.diff(fm.eigvals) <= 0)
    cov = np.cov(raw, rowvar=False)
    np.testing.assert_allclose(cov @ fm.basis, fm.basis * fm.eigvals[:4], atol=1e-8)


def test_die_score_rows_are_keyed_by_the_scorer():
    from wafer_nuisance.data.config import POOLS, REGIMES
    from wafer_nuisance.data.sampler import die_candidates, make_die
    from wafer_nuisance.features.cache import die_score_row, payload_hash

    regime = REGIMES["overlapping"]
    imgs = np.stack([make_die(regime, POOLS["reference"], i)[0].image for i in range(4)])
    from wafer_nuisance.detect.stage1 import build_reference

    s1 = Stage1(Stage1Config(), build_reference(imgs, Stage1Config()), 0.06)
    raw = np.random.default_rng(0).standard_normal((500, len(HAND_FEATURES)))
    fm = fit_feature_map("hand", "hand", raw)
    w = np.linspace(-1, 1, fm.dim)
    assert payload_hash((fm, w)) != payload_hash((fm, w + 1e-9))
    assert payload_hash((fm, w)) == payload_hash((fm, w.copy()))
    pool = POOLS["far_test"]
    for i in range(6):
        row = die_score_row(regime, s1, pool, i, (fm, w))
        die, res = die_candidates(regime, s1, pool, i)
        assert int(row["n_candidates"]) == res.comps.count and not bool(row["die_defective"])
        if res.comps.count == 0:
            assert np.isneginf(row["max_score"])
        else:
            feats = np.stack([handcrafted_features(die.image, s1, res, k)
                              for k in range(res.comps.count)])
            assert float(row["max_score"]) == pytest.approx((fm(feats) @ w).max())


def test_payload_free_cache_keys_are_stable(tmp_path, monkeypatch):
    """A row type without a payload is keyed by exactly (regime, stage-1 config, tau, pool
    identity, rows, version): adding the payload argument must not move existing caches."""
    from wafer_nuisance.data.config import POOLS, REGIMES, config_hash
    from wafer_nuisance.features import cache

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    regime, pool = REGIMES["overlapping"], POOLS["train"]
    s1 = Stage1(Stage1Config(), np.zeros((8, 8)), 0.06)
    ident = (pool.name, pool.seed, pool.defect_free, pool.all_defective)
    key = config_hash(regime, s1.cfg, s1.tau, ident, "meta", cache.ROWS["meta"][1])
    path = tmp_path / "pools" / f"overlapping-train-meta-{key}" / "chunk000000.npz"
    path.parent.mkdir(parents=True)
    np.savez(path, x=np.arange(cache.CHUNK))
    out = cache.pool_arrays(regime, s1, pool, 5, "meta", start=3)
    assert out["x"].tolist() == [3, 4, 5, 6, 7]
