"""Frozen ImageNet backbone -> patch embeddings phi(x).

PatchCore-style recipe, kept deliberately small so every step is explainable:

1. Image in [0, 1] (grayscale dies are replicated to 3 channels; RGB is used as is) -> ImageNet
   mean/std normalisation.
2. Run the frozen network up to layer3 (layer4 and the classifier are never used: they are
   ImageNet-semantic and too coarse spatially for small defects).
3. Take layer2 (stride 8) and layer3 (stride 16) feature maps.
4. Optional local average pooling (``local_pool`` x ``local_pool``, stride 1) on each map, which
   aggregates a small neighbourhood into each embedding and makes it tolerant to sub-pixel
   misalignment between dies. Zero padding counts in the average (``count_include_pad=True``),
   as in PatchCore's unfold-based aggregation. Excluding it gives border embeddings a larger
   variance than interior ones, which makes border patches stand out on defect-free images.
5. Upsample layer3 bilinearly to layer2's grid and concatenate channels.

For ResNet-18 at 256x256 this gives a 32x32 grid of 128 + 256 = 384-dim embeddings; each grid cell
is one "patch" (an 8x8 pixel cell at stride 8, with a much larger receptive field). The backbone is
never trained: parameters are frozen and the module is always in eval mode (BatchNorm uses its
ImageNet running statistics).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    Wide_ResNet50_2_Weights,
    resnet18,
    wide_resnet50_2,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BackboneName = Literal["resnet18", "wide_resnet50_2"]


def default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass(frozen=True)
class BackboneConfig:
    name: BackboneName = "resnet18"
    local_pool: int = 3  # 1 disables neighbourhood aggregation
    pretrained: bool = True  # False only for offline shape tests


class PatchFeatureExtractor(nn.Module):
    def __init__(self, cfg: BackboneConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or BackboneConfig()
        if cfg.name == "resnet18":
            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained else None)
        elif cfg.name == "wide_resnet50_2":
            net = wide_resnet50_2(
                weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1 if cfg.pretrained else None
            )
        else:
            raise ValueError(f"unknown backbone {cfg.name!r}")
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> PatchFeatureExtractor:
        # Frozen: BatchNorm must keep using ImageNet statistics no matter what the caller does.
        return super().train(False)

    @property
    def dim(self) -> int:
        return self.layer2[-1].conv1.in_channels + self.layer3[-1].conv1.in_channels

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, H, W) / (B, 1, H, W) grayscale or (B, 3, H, W) RGB in [0, 1] -> (B, C, H/8, W/8)."""
        if images.ndim == 3:
            images = images[:, None]
        x = (images.expand(-1, 3, -1, -1) - self.mean) / self.std
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        if self.cfg.local_pool > 1:
            k = self.cfg.local_pool
            f2 = F.avg_pool2d(f2, k, stride=1, padding=k // 2, count_include_pad=True)
            f3 = F.avg_pool2d(f3, k, stride=1, padding=k // 2, count_include_pad=True)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)


def extract_features(
    extractor: PatchFeatureExtractor,
    images: Iterable[np.ndarray],
    batch_size: int = 64,
    device: torch.device | None = None,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    """Embed a stream of (H, W) grayscale or (3, H, W) RGB images.

    Returns (N, h, w, C) float32, or (N, len(positions), C) if ``positions`` (flat indices into the
    h*w grid) is given, which lets large sweeps keep only the cells they analyse.
    """
    device = device or default_device()
    extractor = extractor.to(device)
    out = []
    for batch in _batched(images, batch_size):
        x = torch.from_numpy(np.stack(batch)).to(device=device, dtype=torch.float32)
        f = extractor(x).permute(0, 2, 3, 1)  # (B, h, w, C)
        if positions is not None:
            f = f.reshape(f.shape[0], -1, f.shape[-1])[:, torch.as_tensor(positions, device=device)]
        out.append(f.float().cpu().numpy())
    return np.concatenate(out)


def _batched(items: Iterable[np.ndarray], n: int) -> Iterator[list[np.ndarray]]:
    batch: list[np.ndarray] = []
    for item in items:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch
