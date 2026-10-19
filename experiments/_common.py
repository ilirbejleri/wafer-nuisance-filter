"""Shared helpers for the experiment scripts: results I/O (stamped with provenance) and plotting.

Pool construction, sampling and feature caching live in the package (``wafer_nuisance.data``,
``wafer_nuisance.features``), so every experiment draws from the same pools.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator

from wafer_nuisance.utils.provenance import provenance

ROOT = Path(__file__).resolve().parents[1]


def loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def write_results(tag: str, cfg, summary: dict, rows: list[dict], device) -> dict:
    out = {"config": asdict(cfg), "provenance": provenance(str(device)), "summary": summary}
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / f"{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    if rows:
        with open(ROOT / "results" / f"{tag}_raw.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    return out


def read_results(tag: str) -> dict:
    return json.loads((ROOT / "results" / f"{tag}.json").read_text())


def log_ticks(ax, x: Sequence[float], y_log: bool = True) -> None:
    """Ticks at the swept values, plain numbers instead of 2x10^0-style labels."""
    ax.set_xticks(list(x), [f"{int(v)}" for v in x])
    ax.xaxis.set_minor_locator(NullLocator())
    if y_log:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
