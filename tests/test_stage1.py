from dataclasses import replace

import numpy as np
import pytest

from wafer_nuisance.data.synthetic import (
    DefectSpec,
    NuisanceConfig,
    SyntheticWafer,
    make_layout_design,
    render_die,
)
from wafer_nuisance.detect.align import estimate_shift, warp
from wafer_nuisance.detect.stage1 import (
    Stage1Config,
    build_reference,
    choose_threshold,
    component_is_defect,
    connected_components,
    crop,
    defect_peak,
    difference_image,
    photometric_match,
)
from wafer_nuisance.utils.seeding import item_rng

STILL = NuisanceConfig(shift_std=0, cd_bias_std=0, ler_std=0, psf_sigma_std=0, gain_std=0,
                       offset_std=0, noise_std=0)


@pytest.fixture(scope="module")
def design():
    return make_layout_design(item_rng(0, "design", 0))


# ---- alignment -------------------------------------------------------------------------------


@pytest.mark.parametrize("dy, dx", [(0.37, -0.61), (-1.25, 0.8), (0.0, 0.45), (2.1, -2.4)])
def test_alignment_recovers_known_subpixel_shift(design, dy, dx):
    """A die rendered with misalignment (dx, dy) is the reference moved by s = (-dy, -dx)
    (die coordinates are pixel coordinates plus the misalignment)."""
    wafer = SyntheticWafer(design, STILL, 0)
    ref = wafer.normal_die(0).image
    moved = render_die(design, replace(wafer.draw(0, "reference"), dx=dx, dy=dy)).image
    sy, sx = estimate_shift(moved, ref, max_shift=3)
    assert abs(sy - (-dy)) < 0.05 and abs(sx - (-dx)) < 0.05


def test_warp_convention():
    img = np.zeros((32, 32))
    img[10, 12] = 1.0
    out = warp(img, (3.0, -2.0), order=1)
    assert np.unravel_index(out.argmax(), out.shape) == (13, 10)


def test_photometric_match_removes_gain_and_offset():
    rng = np.random.default_rng(0)
    ref = rng.random((40, 40))
    test = 1.3 * ref + 0.2
    np.testing.assert_allclose(photometric_match(test, ref, border=2), test, atol=1e-12)


def test_defect_free_difference_is_small_and_defect_shows(design):
    """With no nuisance at all the difference image is ~0 away from defects."""
    wafer = SyntheticWafer(design, STILL, 0)
    cfg = Stage1Config()
    ref = build_reference(np.stack([wafer.normal_die(i).image for i in range(3)]), cfg)
    diff, _, _ = difference_image(wafer.normal_die(5, "t").image, ref, cfg)
    assert diff.max() < 1e-3
    die = wafer.defective_die(5, DefectSpec("particle", size=3.0, contrast=0.8), "t")
    diff, _, _ = difference_image(die.image, ref, cfg)
    assert defect_peak(diff, die.mask) > 0.2


# ---- components and the label rule -----------------------------------------------------------


def _three_blobs():
    diff = np.zeros((20, 20))
    diff[2:5, 2:5] = 0.5  # component A
    diff[10:13, 10:13] = 0.9  # component B
    diff[16:18, 2:4] = 0.3  # component C
    diff[13, 13] = 0.7  # touches B diagonally -> same component under 8-connectivity
    return diff


def test_connected_components_stats():
    c = connected_components(_three_blobs(), 0.1)
    assert c.count == 3
    order = np.argsort(c.centroid[:, 0])  # A, B, C by row
    np.testing.assert_array_equal(c.area[order], [9, 10, 4])
    np.testing.assert_allclose(c.peak[order], [0.5, 0.9, 0.3])
    np.testing.assert_allclose(c.centroid[order][0], [3.0, 3.0])


def test_label_rule_on_hand_built_mask():
    diff = _three_blobs()
    c = connected_components(diff, 0.1)
    mask = np.zeros_like(diff, dtype=bool)
    mask[14, 14] = True  # background pixel diagonally next to B: no overlap
    assert not component_is_defect(c, mask).any()
    mask[13, 13] = True  # exactly one pixel of component B
    lab_b = c.labels[11, 11]
    expected = np.zeros(c.count, dtype=bool)
    expected[lab_b - 1] = True
    np.testing.assert_array_equal(component_is_defect(c, mask), expected)
    mask[:] = False
    mask[5, 5] = True  # diagonal neighbour of A but not part of it
    assert not component_is_defect(c, mask).any()


def test_recall_identity():
    """Stage 1 finds the defect iff defect_peak > tau."""
    diff = _three_blobs()
    mask = np.zeros_like(diff, dtype=bool)
    mask[16, 2] = True
    for tau in (0.2, 0.29, 0.31, 0.6):
        c = connected_components(diff, tau)
        assert component_is_defect(c, mask).any() == (defect_peak(diff, mask) > tau)


def test_choose_threshold():
    peaks = np.arange(1, 101) / 100  # 0.01 ... 1.00
    tau, recall = choose_threshold(peaks, 0.95)
    assert recall == pytest.approx(0.95) and 0.05 < tau < 0.06
    tau, recall = choose_threshold(peaks, 1.0)
    assert recall == 1.0 and tau < 0.01


def test_crop_is_centred_and_padded():
    img = np.arange(100.0).reshape(10, 10)
    w = crop(img, (4.2, 5.7), 4)
    assert w.shape == (4, 4) and w[2, 2] == img[4, 6]
    edge = crop(img, (0, 0), 4, mode="constant")
    assert edge.shape == (4, 4) and edge[2, 2] == img[0, 0] and edge[0, 0] == 0
