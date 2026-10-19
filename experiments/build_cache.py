"""Fill the feature cache for the pools the E2-E4 experiments read, block by block (so memory
stays bounded). Optional: every experiment computes missing chunks itself; this script only lets
the long generation run once, up front, in the background.

    uv run python experiments/build_cache.py            # everything below
    uv run python experiments/build_cache.py --quick    # the sizes the --quick runs read

Sizes (candidates, overlapping regime, "features" rows):
  feature_fit 20,000 | validation 10,000 | test 100,000 |
  population 1,000,000 (E3 estimates theta* on it; see e3_learning_curve.py) |
  train 5,200,000 = 200 E3b repeats x 26,000 (= 4 x 100 x 65); E3 uses the first 200 x 19,500
  (= 300 x 65), E2 the first 50 x 2^15.
"""

from __future__ import annotations

import argparse
import time

from wafer_nuisance.data.config import POOLS, REGIMES, STAGE1_REGIME, STAGE1_TARGET_RECALL
from wafer_nuisance.detect.stage1 import Stage1Config
from wafer_nuisance.features.cache import pool_features, tune_stage1

BLOCK = 2**16
FULL = {"feature_fit": 20_000, "validation": 10_000, "test": 100_000,
        "population": 1_000_000, "train": 5_200_000}
QUICK = {"feature_fit": 2048, "validation": 2048, "test": 4096, "population": 8192,
         "train": 52_000}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    stage1, _ = tune_stage1(REGIMES[STAGE1_REGIME], Stage1Config(), STAGE1_TARGET_RECALL)
    regime = REGIMES["overlapping"]
    for name, n in (QUICK if args.quick else FULL).items():
        t0 = time.perf_counter()
        for start in range(0, n, BLOCK):
            pool_features(regime, stage1, POOLS[name], min(BLOCK, n - start), start)
            print(f"{name}: {min(start + BLOCK, n):,} / {n:,} "
                  f"({time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
