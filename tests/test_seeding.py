import numpy as np

from wafer_nuisance.utils.seeding import item_rng


def test_same_key_same_stream():
    a = item_rng(0, "reference", 3).standard_normal(10)
    b = item_rng(0, "reference", 3).standard_normal(10)
    np.testing.assert_array_equal(a, b)


def test_keys_are_distinct():
    base = item_rng(0, "reference", 3).standard_normal(10)
    for other in (item_rng(1, "reference", 3), item_rng(0, "calibration", 3),
                  item_rng(0, "reference", 4)):
        assert not np.allclose(base, other.standard_normal(10))
