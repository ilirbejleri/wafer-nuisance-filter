"""Metadata stamped into every results file so each number traces to code + environment."""

from __future__ import annotations

import platform
import subprocess
from datetime import UTC, datetime

import numpy as np
import torch


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(device: str | None = None) -> dict:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        # Code changes only: untracked files and the experiment outputs themselves (results/,
        # figures/, which a run may be overwriting) are not code changes.
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no", "--", ".",
                               ":!results", ":!figures")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": device,
    }
