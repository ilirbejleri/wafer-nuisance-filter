"""README.md is generated from docs/README.template.md and results/theory_constants.json; this
fails if it is stale (e.g. results were regenerated but the README was not rebuilt)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_is_up_to_date():
    spec = importlib.util.spec_from_file_location("build_readme",
                                                  ROOT / "experiments" / "build_readme.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (ROOT / "README.md").read_text() == mod.render(), (
        "README.md is stale: run `uv run python experiments/build_readme.py`")
