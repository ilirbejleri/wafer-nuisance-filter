from dataclasses import replace

import numpy as np
import pytest

from wafer_nuisance.data.config import POOLS, REGIMES, config_hash
from wafer_nuisance.data.sampler import make_die, sample_candidate
from wafer_nuisance.data.synthetic import NuisanceConfig, SyntheticWafer, make_layout_design
from wafer_nuisance.detect.stage1 import Stage1, Stage1Config, build_reference
from wafer_nuisance.utils.seeding import item_rng

REGIME = REGIMES["overlapping"]
POOL = POOLS["validation"]


@pytest.fixture(scope="module")
def stage1():
    cfg = Stage1Config()
    imgs = np.stack([make_die(REGIME, POOLS["reference"], i)[0].image for i in range(4)])
    return Stage1(cfg, build_reference(imgs, cfg), tau=0.06)


def test_sampler_returns_one_candidate_from_the_first_die_with_candidates(stage1):
    """Each example is exactly one component of one fresh die: the dies drawn before it had no
    candidates, and the returned candidate is one of the components of the die it came from."""
    for i in range(12):
        c = sample_candidate(REGIME, stage1, POOL, i)
        for a in range(c.attempts - 1):
            die, _ = make_die(REGIME, POOL, i, a)
            assert stage1.run(die.image, die.mask).comps.count == 0
        die, spec = make_die(REGIME, POOL, i, c.attempts - 1)
        res = stage1.run(die.image, die.mask)
        assert res.comps.count == c.n_candidates >= 1
        k = [j for j in range(res.comps.count)
             if np.allclose(res.comps.centroid[j], c.centroid)]
        assert len(k) == 1
        assert c.label == (1 if res.is_defect[k[0]] else -1)
        assert c.peak == pytest.approx(res.comps.peak[k[0]])
        assert c.die_defective == (spec is not None)
        assert c.crops["component"].dtype == bool and c.crops["component"].any()


def test_sampler_is_deterministic_and_items_are_distinct_dies(stage1):
    a = sample_candidate(REGIME, stage1, POOL, 3)
    b = sample_candidate(REGIME, stage1, POOL, 3)
    assert a.centroid == b.centroid and a.peak == b.peak
    imgs = [make_die(REGIME, POOL, i)[0].image for i in range(3)]
    assert not np.allclose(imgs[0], imgs[1]) and not np.allclose(imgs[1], imgs[2])


def test_pools_are_independent_and_regimes_share_dies():
    """Different pools never share a die; the two regimes share nuisance and defect placement
    for the same (pool, index) and differ only in contrast."""
    d_val, s_val = make_die(REGIME, POOLS["validation"], 0)
    d_test, _ = make_die(REGIME, POOLS["test"], 0)
    assert not np.allclose(d_val.image, d_test.image)
    for i in range(20):
        d_o, s_o = make_die(REGIMES["overlapping"], POOL, i)
        d_s, s_s = make_die(REGIMES["separable"], POOL, i)
        assert (s_o is None) == (s_s is None)
        if s_o is not None:
            assert s_o.kind == s_s.kind
            np.testing.assert_array_equal(d_o.mask, d_s.mask)
        else:
            np.testing.assert_array_equal(d_o.image, d_s.image)


def test_defect_free_pools_have_no_defects():
    for i in range(10):
        die, spec = make_die(REGIME, POOLS["far_test"], i)
        assert spec is None and not die.mask.any()
    for i in range(10):
        die, spec = make_die(REGIME, POOLS["stage1_tune"], i)
        assert spec is not None and die.mask.any()


def test_pool_seeds_are_distinct():
    seeds = [p.seed for p in POOLS.values()]
    assert len(seeds) == len(set(seeds))


def test_config_hash_tracks_changes():
    assert config_hash(REGIME) == config_hash(REGIMES["overlapping"])
    assert config_hash(REGIME) != config_hash(replace(REGIME, design_seed=1))


# ---- new nuisance sources in the generator ---------------------------------------------------


def test_blobs_change_intensity_not_mask():
    design = make_layout_design(item_rng(0, "design", 0))
    base = NuisanceConfig(noise_std=0)
    plain = SyntheticWafer(design, base, 0)
    blobby = SyntheticWafer(design, replace(base, blob_rate=5.0, blob_amp_min=0.1,
                                            blob_amp_max=0.2), 0)
    draw = blobby.draw(0, "x")
    assert len(draw.blobs) > 0
    a, b = plain.normal_die(0, "x"), blobby.normal_die(0, "x")
    assert not b.mask.any()
    d = np.abs(b.image - a.image)
    assert 0.05 < d.max() <= 0.2 * len(draw.blobs) + 1e-6


def test_warp_moves_pattern_locally():
    design = make_layout_design(item_rng(0, "design", 0))
    still = NuisanceConfig(shift_std=0, cd_bias_std=0, ler_std=0, psf_sigma_std=0, gain_std=0,
                           offset_std=0, noise_std=0)
    flat = SyntheticWafer(design, still, 0).normal_die(0).image
    warped_w = SyntheticWafer(design, replace(still, warp_std=0.2), 0)
    warped = warped_w.normal_die(0).image
    diff = np.abs(warped - flat)
    assert diff.max() > 0.01  # pattern moved
    # The warp field is smooth: its control values are N(0, warp_std^2) on a coarse grid.
    from wafer_nuisance.data.synthetic import _warp_field

    wu, wv = _warp_field(flat.shape, warped_w.draw(0, "reference"))
    assert np.abs(np.diff(wu, axis=1)).max() < 0.05 and wu.std() > 0.05


def test_blob_min_gap_keeps_blobs_apart():
    from wafer_nuisance.data.synthetic import DIE_SIZE, draw_nuisance

    cfg = NuisanceConfig(blob_rate=6.0, blob_amp_min=0.1, blob_amp_max=0.2, blob_min_gap=3.0)
    n_pairs = 0
    for i in range(200):
        blobs = draw_nuisance(cfg, item_rng(0, "gap", i)).blobs
        for j, a in enumerate(blobs):
            for b in blobs[:j]:
                n_pairs += 1
                d = DIE_SIZE * np.hypot(a.fy - b.fy, a.fx - b.fx)
                assert d >= 3.0 * (a.sigma + b.sigma)
    assert n_pairs > 100


def test_features_do_not_change_the_draw(stage1):
    from wafer_nuisance.features.handcrafted import HAND_FEATURES

    for i in range(5):
        a = sample_candidate(REGIME, stage1, POOL, i)
        b = sample_candidate(REGIME, stage1, POOL, i, features=True)
        assert a.hand is None and b.hand.shape == (len(HAND_FEATURES),)
        assert a.centroid == b.centroid and a.label == b.label
        f = dict(zip(HAND_FEATURES, b.hand, strict=True))
        assert f["peak_contrast"] == pytest.approx(b.peak)
        assert f["log_area"] == pytest.approx(np.log(b.area))
