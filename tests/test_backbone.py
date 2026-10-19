"""Backbone tests use random weights (pretrained=False) so the suite runs offline; they check
plumbing (shapes, freezing, position selection), not feature quality."""

import numpy as np
import pytest
import torch

from wafer_nuisance.features import BackboneConfig, PatchFeatureExtractor, extract_features

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def extractor():
    torch.manual_seed(0)
    return PatchFeatureExtractor(BackboneConfig(pretrained=False))


def test_shape_and_dim(extractor):
    out = extractor(torch.rand(2, 256, 256))
    assert out.shape == (2, 384, 32, 32) and extractor.dim == 384


def test_frozen(extractor):
    assert not any(p.requires_grad for p in extractor.parameters())
    extractor.train()
    assert not extractor.training


def test_deterministic(extractor):
    x = torch.rand(1, 256, 256)
    torch.testing.assert_close(extractor(x), extractor(x))


def test_position_selection(extractor):
    imgs = [np.random.default_rng(i).random((256, 256), dtype=np.float32) for i in range(3)]
    full = extract_features(extractor, imgs, batch_size=2, device=CPU)
    pos = np.array([0, 5, 1023])
    sub = extract_features(extractor, imgs, batch_size=2, device=CPU, positions=pos)
    assert full.shape == (3, 32, 32, 384) and sub.shape == (3, 3, 384)
    np.testing.assert_allclose(sub, full.reshape(3, -1, 384)[:, pos], rtol=1e-5, atol=1e-5)


def test_rgb_input_matches_replicated_gray(extractor):
    g = torch.rand(2, 64, 64)
    torch.testing.assert_close(extractor(g), extractor(g[:, None].expand(-1, 3, -1, -1)))


def test_cnn_candidate_features_shape(monkeypatch):
    """Plumbing of features.cnn with random weights: (N, 384), finite, zero for identical crops."""
    from wafer_nuisance.features import cnn

    torch.manual_seed(0)
    net = PatchFeatureExtractor(BackboneConfig(pretrained=False))
    monkeypatch.setattr(cnn, "extractor", lambda cfg=None: net)
    rng = np.random.default_rng(0)
    test = rng.random((3, 32, 32), dtype=np.float32)
    comp = np.zeros((3, 32, 32), bool)
    comp[:, 14:18, 14:18] = True
    f = cnn.cnn_features(test, test.copy(), comp)
    assert f.shape == (3, 384) and np.allclose(f, 0)
    g = cnn.cnn_features(test, rng.random((3, 32, 32), dtype=np.float32), comp)
    assert np.isfinite(g).all() and np.abs(g).max() > 0
