from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import binary_dilation

from wafer_nuisance.data.synthetic import (
    DEFECT_KINDS,
    DefectSpec,
    NuisanceConfig,
    SyntheticWafer,
    die_coordinates,
    make_layout_design,
    make_uniform_design,
    random_defect_spec,
    render_die,
)
from wafer_nuisance.utils.seeding import item_rng

STILL = NuisanceConfig(shift_std=0, cd_bias_std=0, ler_std=0, psf_sigma_std=0, gain_std=0,
                       offset_std=0, noise_std=0)


@pytest.fixture(scope="module")
def wafer():
    return SyntheticWafer.from_seed(7)


def spec_for(kind, **kw):
    size = {"missing_contact": 2}.get(kind, 4.0)
    contrast = {"scratch": -0.6}.get(kind, 0.8)
    return DefectSpec(kind, size=kw.pop("size", size), contrast=kw.pop("contrast", contrast), **kw)


# ---- reproducibility -------------------------------------------------------------------------


def test_same_seed_same_wafer():
    a, b = SyntheticWafer.from_seed(3), SyntheticWafer.from_seed(3)
    assert a.design == b.design
    np.testing.assert_array_equal(a.normal_images(2), b.normal_images(2))
    np.testing.assert_array_equal(a.defective_die(0, spec_for("bridge")).image,
                                  b.defective_die(0, spec_for("bridge")).image)


def test_reference_sets_are_nested(wafer):
    """n-sweeps require the size-n reference set to be a prefix of the size-n' set."""
    np.testing.assert_array_equal(wafer.normal_images(5)[:3], wafer.normal_images(3))
    np.testing.assert_array_equal(wafer.normal_images(2, start=3), wafer.normal_images(5)[3:])


def test_streams_are_distinct(wafer):
    assert not np.allclose(wafer.normal_die(0, "reference").image,
                           wafer.normal_die(0, "calibration").image)


def test_dies_vary_under_nuisance(wafer):
    imgs = wafer.normal_images(4)
    assert all(not np.allclose(imgs[0], imgs[k]) for k in range(1, 4))


# ---- design ----------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_layout_tiles_die_and_has_required_blocks(seed):
    d = make_layout_design(item_rng(seed, "design", 0))
    assert (d.block_index_map() >= 0).all()
    assert sum((b.x1 - b.x0) * (b.y1 - b.y0) for b in d.blocks) == d.height * d.width
    assert d.blocks_of_kind("lines") and d.blocks_of_kind("contacts")


@pytest.mark.parametrize("kind,orientation", [("lines", "v"), ("lines", "h"), ("contacts", "v")])
def test_pattern_is_periodic(kind, orientation):
    p = 12
    w = SyntheticWafer(make_uniform_design(kind, pitch=p, orientation=orientation), STILL, 0)
    img = w.normal_die(0).image.astype(np.float64)
    inner = (slice(20, -20), slice(20, -20))
    axes = [1] if (kind == "lines" and orientation == "v") else \
        [0] if kind == "lines" else [0, 1]
    for ax in axes:
        np.testing.assert_allclose(np.roll(img, p, axis=ax)[inner], img[inner], atol=1e-6)


def test_shift_by_one_pitch_is_invisible():
    w = SyntheticWafer(make_uniform_design("lines", pitch=12), STILL, 0)
    draw = w.draw(0, "reference")
    a = w.normal_die(0).image
    b = render_die(w.design, replace(draw, dx=draw.dx + 12.0)).image
    # Away from the die border: the shift moves the last 12 columns off the die (reflectance 0
    # there, like a scribe street), which is intended.
    np.testing.assert_allclose(a[:, 20:-20], b[:, 20:-20], atol=1e-5)


def test_subpixel_shift_changes_image():
    w = SyntheticWafer(make_uniform_design("lines", pitch=12), STILL, 0)
    draw = w.draw(0, "reference")
    b = render_die(w.design, replace(draw, dx=0.3)).image
    assert np.abs(w.normal_die(0).image - b).max() > 1e-3


def test_cd_bias_drift_increases_material():
    base = SyntheticWafer(make_uniform_design("lines", pitch=12), STILL, 0)
    drifted = replace(base, nuisance=replace(STILL, cd_bias_mean=1.0))
    assert drifted.normal_die(0).image.mean() > base.normal_die(0).image.mean() + 0.02


def test_noise_level():
    cfg = NuisanceConfig(noise_std=0.05)
    w = SyntheticWafer.from_seed(1, nuisance=cfg)
    resid = w.normal_die(0).image - w.normal_die(0, noiseless=True).image
    assert abs(resid.std() - 0.05) < 0.0025


# ---- normal dies -----------------------------------------------------------------------------


def test_normal_die_is_clean(wafer):
    die = wafer.normal_die(0)
    assert die.image.shape == (256, 256) and die.image.dtype == np.float32
    assert np.isfinite(die.image).all()
    assert not die.mask.any() and not die.is_defective


# ---- defects ---------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", DEFECT_KINDS)
def test_defect_is_localised(wafer, kind):
    """The noiseless image changes only near the ground-truth mask, and it does change."""
    die = wafer.defective_die(0, spec_for(kind), noiseless=True)
    clean = wafer.normal_die(0, stream="test", noiseless=True)
    assert die.mask.any()
    changed = np.abs(die.image - clean.image) > 1e-3 * die.draw.gain
    assert changed.any()
    # PSF reach: ~4 sigma plus the defect's soft edge and the bridge/open overlap.
    reach = int(np.ceil(4 * die.draw.psf_sigma)) + 3
    near = binary_dilation(die.mask, iterations=reach)
    assert not (changed & ~near).any()


@pytest.mark.parametrize("kind", DEFECT_KINDS)
def test_zero_contrast_is_invisible_but_mask_kept(wafer, kind):
    d0 = wafer.defective_die(1, spec_for(kind, contrast=0.0), noiseless=True)
    d1 = wafer.defective_die(1, spec_for(kind), noiseless=True)
    clean = wafer.normal_die(1, stream="test", noiseless=True)
    np.testing.assert_allclose(d0.image, clean.image, atol=1e-6)
    np.testing.assert_array_equal(d0.mask, d1.mask)


@pytest.mark.parametrize("kind,sizes", [
    ("particle", (1.5, 3.0, 6.0)),
    ("scratch", (1.0, 2.0, 4.0)),
    ("bridge", (2.0, 4.0, 8.0)),
    ("open", (2.0, 4.0, 8.0)),
    ("missing_contact", (1, 2, 3)),
])
def test_mask_area_grows_with_size(wafer, kind, sizes):
    areas = [wafer.defective_die(2, spec_for(kind, size=s)).mask.sum() for s in sizes]
    assert all(a < b for a, b in zip(areas, areas[1:], strict=False)), areas


@pytest.mark.parametrize("kind,sign", [("bridge", +1), ("open", -1), ("missing_contact", -1)])
def test_material_defects_change_reflectance_in_right_direction(wafer, kind, sign):
    die = wafer.defective_die(3, spec_for(kind, contrast=1.0), noiseless=True)
    clean = wafer.normal_die(3, stream="test", noiseless=True)
    assert sign * (die.image - clean.image)[die.mask].mean() > 0


@pytest.mark.parametrize("kind,block_kind", [("bridge", "lines"), ("open", "lines"),
                                             ("missing_contact", "contacts")])
def test_pattern_defects_sit_in_matching_block(wafer, kind, block_kind):
    idx = wafer.design.block_index_map()
    for i in range(10):
        die = wafer.defective_die(i, spec_for(kind))
        kinds = {wafer.design.blocks[k].kind for k in np.unique(idx[die.mask])}
        assert kinds == {block_kind}


def test_bridge_mask_excludes_lines(wafer):
    """A bridge's footprint overlaps the neighbouring lines, but only the space it fills is
    ground truth, so the mask is narrower than the footprint."""
    die = wafer.defective_die(4, spec_for("bridge"))
    fp = die.defects[0].footprint(*die_coordinates(wafer.design, die.draw))
    assert die.mask.sum() < (fp >= 0.5).sum()


@pytest.mark.parametrize("seed", range(5))
def test_random_specs_always_place(seed):
    w = SyntheticWafer.from_seed(seed)
    rng = np.random.default_rng(seed)
    for i in range(40):
        die = w.defective_die(i, random_defect_spec(rng))
        assert die.mask.any()


def test_multiple_defects_union_mask(wafer):
    a = wafer.defective_die(5, spec_for("particle"))
    both = wafer.defective_die(5, (spec_for("particle"), spec_for("open")))
    assert both.mask.sum() > a.mask.sum()
    assert len(both.defects) == 2


@pytest.mark.parametrize("bad", [
    dict(kind="crater", size=1.0),
    dict(kind="particle", size=0.0),
    dict(kind="missing_contact", size=1.5),
    dict(kind="bridge", size=3.0, contrast=1.5),
    dict(kind="particle", size=2.0, aspect=0.5),
])
def test_spec_validation(bad):
    with pytest.raises(ValueError):
        DefectSpec(**bad)
