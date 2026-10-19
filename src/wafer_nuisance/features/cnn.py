"""Frozen-CNN candidate features (d = 384).

The 32 x 32 test and reference crops are upsampled 2x (bilinear) to 64 x 64 and embedded by the
frozen ResNet-18 layer2 + layer3 extractor of ``backbone`` (local 3 x 3 average pooling with
``count_include_pad=True``), giving an 8 x 8 grid of 384-dim embeddings per crop; each grid cell
covers a 4 x 4 pixel block of the crop. The candidate's feature is the die-to-die difference of
embeddings averaged over its footprint:

    x_cnn = mean over footprint cells g of ( phi_test(g) - phi_reference(g) ),

where the footprint is the set of cells containing at least one pixel of the component (all
cells, if none of the component falls inside the crop).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from wafer_nuisance.features.backbone import BackboneConfig, PatchFeatureExtractor, default_device

UPSAMPLE = 2
CELL = 8 // UPSAMPLE  # crop pixels per grid cell side (layer2 has stride 8)

_EXTRACTOR: dict[str, PatchFeatureExtractor] = {}


def extractor(cfg: BackboneConfig | None = None) -> PatchFeatureExtractor:
    cfg = cfg or BackboneConfig()
    key = repr(cfg)
    if key not in _EXTRACTOR:
        _EXTRACTOR[key] = PatchFeatureExtractor(cfg).to(default_device())
    return _EXTRACTOR[key]


def footprint_cells(component: np.ndarray) -> np.ndarray:
    """(N, s, s) bool crops -> (N, s / CELL, s / CELL) bool: cells containing a component pixel."""
    n, s, _ = component.shape
    cells = component.reshape(n, s // CELL, CELL, s // CELL, CELL).any(axis=(2, 4))
    cells[~cells.any(axis=(1, 2))] = True
    return cells


@torch.inference_mode()
def cnn_features(test: np.ndarray, reference: np.ndarray, component: np.ndarray,
                 batch_size: int = 256, cfg: BackboneConfig | None = None) -> np.ndarray:
    """(N, 32, 32) test / reference crops and bool component crops -> (N, 384) float32."""
    net = extractor(cfg)
    dev = next(net.parameters()).device
    cells = footprint_cells(np.asarray(component, dtype=bool))
    out = []
    for i in range(0, len(test), batch_size):
        sl = slice(i, i + batch_size)
        x = torch.from_numpy(np.concatenate([test[sl], reference[sl]])).to(dev, torch.float32)
        x = F.interpolate(x[:, None], scale_factor=UPSAMPLE, mode="bilinear",
                          align_corners=False)
        f = net(x)  # (2B, 384, 8, 8)
        b = len(f) // 2
        delta = (f[:b] - f[b:]).float()
        m = torch.from_numpy(cells[sl]).to(dev, torch.float32)[:, None]  # (B, 1, 8, 8)
        out.append(((delta * m).sum((2, 3)) / m.sum((2, 3))).cpu().numpy())
    return np.concatenate(out).astype(np.float32)
