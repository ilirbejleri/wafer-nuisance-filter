"""Seeding helpers.

Two mechanisms, used for different things:

* ``item_rng(base_seed, stream, index)`` gives every generated item (die, defect, ...) its own
  independent RNG derived from a fixed key. Item ``i`` is therefore identical no matter how many
  items are requested or in what order, so a reference set of size n is a prefix of the set of
  size n' > n. Sweeps over n rely on this nesting.
* ``seed_everything(seed)`` seeds the global RNGs (``random``, NumPy legacy, torch if installed)
  for code paths we don't control, e.g. library internals.
"""

from __future__ import annotations

import random
import zlib

import numpy as np


def _stream_key(stream: str) -> int:
    # Stable across processes and Python versions (unlike hash()).
    return zlib.crc32(stream.encode("utf-8"))


def item_rng(base_seed: int, stream: str, index: int) -> np.random.Generator:
    """Independent RNG for item ``index`` of named ``stream`` under ``base_seed``."""
    if index < 0:
        raise ValueError(f"index must be non-negative, got {index}")
    ss = np.random.SeedSequence([base_seed, _stream_key(stream), index])
    return np.random.default_rng(ss)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - intentionally seeds the legacy global RNG
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
