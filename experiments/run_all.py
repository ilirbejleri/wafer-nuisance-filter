"""Reproduce everything: every experiment, the theory numbers, the headline figure, the README and
the theory PDF.

    uv run python experiments/run_all.py              # full run
    uv run python experiments/run_all.py --quick      # smoke run of every experiment

Run it from a clean git tree: every results file records the commit it came from and whether
the tree had uncommitted code changes. In --quick mode the experiments write separate *_quick
files, and the write-up steps (which read the full results) are skipped.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time

from _common import ROOT

# build_cache only fills the feature cache in parallel up front (every experiment would compute
# missing chunks itself, more slowly); it writes no results.
EXPERIMENTS: tuple[str, ...] = ("stage1_report", "features_report", "e1_thresholds", "build_cache",
                                "e2_linear_bounds", "e3_learning_curve", "e3b_plugin_curve",
                                "e4_false_alarms")
WRITEUP = ("theory_tables", "headline", "build_readme")


def run(script: str, *args: str) -> None:
    t0 = time.perf_counter()
    print(f"== {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "experiments" / f"{script}.py"), *args],
                   check=True, cwd=ROOT)
    print(f"   done in {time.perf_counter() - t0:.0f} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".",
                            ":!results", ":!figures"], capture_output=True, text=True, cwd=ROOT)
    if dirty.stdout.strip():
        print("warning: uncommitted code changes; results will be stamped git_dirty=True",
              flush=True)

    extra = ("--quick",) if args.quick else ()
    for script in EXPERIMENTS:
        run(script, *extra)
    if args.quick:
        print("quick mode: skipping the write-up steps (they read the full results)")
        return
    for script in WRITEUP:
        run(script)
    if shutil.which("latexmk"):
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                        "main.tex"], check=True, cwd=ROOT / "theory",
                       stdout=subprocess.DEVNULL)
        print("== theory/main.pdf rebuilt")
    else:
        print("latexmk not found: theory/main.pdf not rebuilt")


if __name__ == "__main__":
    main()
