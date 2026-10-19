"""What stage 1 produces. Candidates per die, positive rate, stage-1 recall, and the check
that the two regimes are what they claim to be.

    uv run python experiments/stage1_report.py              # full run (population pool)
    uv run python experiments/stage1_report.py --quick      # small pools, *_quick outputs
    uv run python experiments/stage1_report.py --plot-only  # redraw figures from results/

Outputs: results/stage1.json, results/stage1_gallery.npz (the gallery crops),
figures/stage1_regimes.png, figures/candidate_gallery.png.

Regime check (definition, fixed in advance): on the population pool, the best threshold on
peak contrast makes zero errors in the separable regime and errs 5-20% in the overlapping one.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import matplotlib
import numpy as np
from _common import ROOT, read_results, write_results

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wafer_nuisance.bounds import hoeffding_interval  # noqa: E402
from wafer_nuisance.data.config import (  # noqa: E402
    POOLS,
    REGIMES,
    STAGE1_REGIME,
    STAGE1_TARGET_RECALL,
)
from wafer_nuisance.data.sampler import sample_candidate  # noqa: E402
from wafer_nuisance.data.synthetic import DEFECT_KINDS  # noqa: E402
from wafer_nuisance.detect.stage1 import Stage1Config  # noqa: E402
from wafer_nuisance.eval.plotting import (  # noqa: E402
    REFERENCE,
    SERIES,
    TEXT_SECONDARY,
    apply_style,
)
from wafer_nuisance.features.cache import pool_arrays, tune_stage1  # noqa: E402
from wafer_nuisance.learn import fit_threshold, threshold_errors  # noqa: E402

PEAK_BINS = np.linspace(0.0, 0.6, 61)
MAX_COUNT_BIN = 12  # candidates-per-die histogram: 1..11 and ">= 12"


@dataclass
class Config:
    regimes: tuple[str, ...] = ("separable", "overlapping")
    n_population: int = POOLS["population"].size
    n_recall_dies: int = POOLS["stage1_check"].size
    n_tune: int = POOLS["stage1_tune"].size
    target_recall: float = STAGE1_TARGET_RECALL
    stage1_regime: str = STAGE1_REGIME
    stage1: Stage1Config = field(default_factory=Stage1Config)
    n_gallery_scan: int = 600  # validation candidates scanned for gallery examples
    delta: float = 0.05
    quick: bool = False


def quick_config() -> Config:
    return Config(n_population=4096, n_recall_dies=2048, n_tune=1000, n_gallery_scan=200,
                  quick=True)


def _rate(k: float, n: int, delta: float) -> dict:
    p = k / n
    return {"value": p, "n": n, "hoeffding_ci": hoeffding_interval(p, n, delta)}


def regime_summary(a: dict, rec: dict, delta: float) -> dict:
    y, peak, ncand = a["label"], a["peak"], a["n_candidates"]
    n = len(y)
    h = fit_threshold(peak, y)
    _, errors = threshold_errors(peak, y)
    defective = a["die_defective"].astype(bool)
    counts = np.bincount(np.minimum(ncand, MAX_COUNT_BIN), minlength=MAX_COUNT_BIN + 1)[1:]
    total_attempts = int(a["attempts"].sum())
    found = rec["found"].astype(bool)
    return {
        "n_candidates_sampled": n,
        "positive_rate": _rate(int((y == 1).sum()), n, delta),
        "positive_rate_on_defective_dies": float(np.mean(y[defective] == 1)),
        "frac_examples_from_defective_dies": float(defective.mean()),
        "candidates_per_die": {
            "mean": float(ncand.mean()), "median": float(np.median(ncand)),
            "q90": float(np.quantile(ncand, 0.9)), "q99": float(np.quantile(ncand, 0.99)),
            "max": int(ncand.max()), "hist_1_to_ge12": counts.tolist(),
            "note": "per die that produced the example, i.e. dies with >= 1 candidate",
        },
        "redraws": {
            "frac_examples_with_redraw": float(np.mean(a["attempts"] > 1)),
            "dies_drawn": total_attempts,
            # Attempts are geometric with success probability P(|C(D)| >= 1).
            "zero_candidate_die_rate": _rate(total_attempts - n, total_attempts, delta),
        },
        "recall": {
            "overall": _rate(int(found.sum()), len(found), delta),
            "by_kind": {k: float(found[rec["defect_kind"] == j].mean())
                        for j, k in enumerate(DEFECT_KINDS)},
            "note": "out-of-sample, on the stage1_check pool (all-defective dies)",
        },
        "threshold_check": {
            "best_theta": h.theta,
            "best_errors": int(errors.min()),
            "best_error_rate": _rate(int(errors.min()), n, delta),
            "max_nuisance_peak": float(peak[y == -1].max()),
            "min_defect_peak": float(peak[y == 1].min()),
            "separable": bool(errors.min() == 0),
            "in_5_to_20_percent": bool(0.05 <= errors.min() / n <= 0.20),
        },
        "peak_hist": {
            "bins": PEAK_BINS.tolist(),
            "defect": np.histogram(np.clip(peak[y == 1], 0, PEAK_BINS[-1] - 1e-9),
                                   PEAK_BINS)[0].tolist(),
            "nuisance": np.histogram(np.clip(peak[y == -1], 0, PEAK_BINS[-1] - 1e-9),
                                     PEAK_BINS)[0].tolist(),
        },
    }


def gallery(regime_name: str, stage1, cfg: Config) -> dict[str, np.ndarray]:
    """Two real defects of each kind (faintest and strongest seen) and ten nuisance candidates
    spread over the nuisance peak range, from the first validation candidates."""
    regime, pool = REGIMES[regime_name], POOLS["validation"]
    cands = [sample_candidate(regime, stage1, pool, i) for i in range(cfg.n_gallery_scan)]
    real = []
    for j in range(len(DEFECT_KINDS)):
        ks = sorted((c for c in cands if c.label == 1 and c.defect_kind == j),
                    key=lambda c: c.defect_contrast)
        real += [ks[0], ks[-1]] if len(ks) >= 2 else ks
    nuis = sorted((c for c in cands if c.label == -1), key=lambda c: c.peak)
    nuis = [nuis[int(round(q))] for q in np.linspace(0, len(nuis) - 1, 10)]
    picks = real + nuis
    return {
        "test": np.stack([c.crops["test"] for c in picks]).astype(np.float16),
        "diff": np.stack([c.crops["diff"] for c in picks]).astype(np.float16),
        "component": np.stack([c.crops["component"] for c in picks]),
        "label": np.array([c.label for c in picks]),
        "kind": np.array([c.defect_kind for c in picks]),
        "contrast": np.array([c.defect_contrast for c in picks]),
        "peak": np.array([c.peak for c in picks]),
        "index": np.array([c.index for c in picks]),
    }


def run(cfg: Config, tag: str) -> None:
    stage1, s1_info = tune_stage1(REGIMES[cfg.stage1_regime], cfg.stage1, cfg.target_recall,
                                  n_tune=cfg.n_tune)
    print(f"stage 1: tau = {stage1.tau:.4f}, tuning recall = {s1_info['tuning_recall']:.4f}",
          flush=True)
    summary = {"stage1": s1_info, "regimes": {}}
    for name in cfg.regimes:
        regime = REGIMES[name]
        a = pool_arrays(regime, stage1, POOLS["population"], cfg.n_population, "meta")
        rec = pool_arrays(regime, stage1, POOLS["stage1_check"], cfg.n_recall_dies, "recall")
        s = regime_summary(a, rec, cfg.delta)
        summary["regimes"][name] = s
        tc = s["threshold_check"]
        print(f"{name}: positive rate {s['positive_rate']['value']:.3f}, "
              f"candidates/die {s['candidates_per_die']['mean']:.2f}, "
              f"recall {s['recall']['overall']['value']:.4f}, "
              f"best threshold errors {tc['best_errors']} / {len(a['label'])} "
              f"(max nuisance peak {tc['max_nuisance_peak']:.3f}, "
              f"min defect peak {tc['min_defect_peak']:.3f})", flush=True)
    g = gallery(cfg.stage1_regime, stage1, cfg)
    np.savez_compressed(ROOT / "results" / f"{tag}_gallery.npz", **g)
    write_results(tag, cfg, summary, [], "cpu")
    plot(tag)


# ---- figures ---------------------------------------------------------------------------------


def plot(tag: str) -> None:
    apply_style()
    r = read_results(tag)
    s = r["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    tau = s["stage1"]["tau"]
    for ax, name in zip(axes[:2], ("separable", "overlapping"), strict=True):
        rs = s["regimes"][name]
        ph = rs["peak_hist"]
        bins = np.array(ph["bins"])
        for j, (cls, lab) in enumerate((("nuisance", "nuisance (−1)"),
                                        ("defect", "real defect (+1)"))):
            ax.stairs(ph[cls], bins, fill=True, alpha=0.35, color=SERIES[j], label=lab)
            ax.stairs(ph[cls], bins, color=SERIES[j], lw=1.2)
        tc = rs["threshold_check"]
        ax.axvline(tau, color=REFERENCE, ls=":", lw=1.5, label=f"stage-1 τ = {tau:.3f}")
        ax.axvline(tc["best_theta"], color=TEXT_SECONDARY, ls="--", lw=1.5,
                   label=f"best threshold ({tc['best_errors']} errors)")
        ax.annotate(f"last bin: ≥ {bins[-2]:g}", (bins[-2], 0), xycoords=("data", "axes fraction"),
                    xytext=(-2, 4), textcoords="offset points", ha="right", fontsize=8,
                    color=TEXT_SECONDARY)
        n = rs["n_candidates_sampled"]
        ax.set(yscale="log", xlabel="peak contrast (difference image)",
               ylabel=f"candidates (of {n:,})",
               title=f"{name}: best 1-D threshold errs {tc['best_errors']:,} / {n:,} "
                     f"({tc['best_error_rate']['value']:.3g})")
        ax.legend(loc="upper right")
    ax = axes[2]
    w = 0.38
    for j, name in enumerate(("separable", "overlapping")):
        cp = s["regimes"][name]["candidates_per_die"]
        h = np.array(cp["hist_1_to_ge12"], dtype=float)
        x = np.arange(1, len(h) + 1)
        ax.bar(x + (j - 0.5) * w, h / h.sum(), w, color=SERIES[j + 2],
               label=f"{name} (mean {cp['mean']:.2f})")
    ax.set_xticks(np.arange(1, MAX_COUNT_BIN + 1),
                  [str(k) for k in range(1, MAX_COUNT_BIN)] + [f"≥{MAX_COUNT_BIN}"])
    ax.set(xlabel="candidates on the die (dies with ≥ 1)", ylabel="fraction of examples",
           title="candidates per die")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"{tag}_regimes.png", dpi=130)
    plt.close(fig)
    plot_gallery(tag)
    print("wrote figures for", tag)


def plot_gallery(tag: str) -> None:
    g = np.load(ROOT / "results" / f"{tag}_gallery.npz")
    lab = g["label"]
    groups = (("Real defects (+1): faintest and strongest of each kind",
               np.flatnonzero(lab == 1)),
              ("Nuisance (−1): spread over the nuisance peak-contrast range",
               np.flatnonzero(lab == -1)))
    ncol = max(len(ix) for _, ix in groups)
    vmax = float(g["diff"].max())
    fig = plt.figure(figsize=(1.5 * ncol, 7.6))
    subfigs = fig.subfigures(2, 1, hspace=0.04)
    for sf, (title, ix) in zip(subfigs, groups, strict=True):
        axes = sf.subplots(2, ncol)
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        for col in range(ncol):
            if col >= len(ix):
                axes[0, col].axis("off")
                axes[1, col].axis("off")
                continue
            k = ix[col]
            top, bot = axes[0, col], axes[1, col]
            top.imshow(g["test"][k].astype(float), cmap="gray", vmin=0.1, vmax=0.9)
            bot.imshow(g["diff"][k].astype(float), cmap="magma", vmin=0, vmax=vmax)
            bot.contour(g["component"][k], levels=[0.5], colors="#1baf7a", linewidths=0.8)
            if lab[k] == 1:
                top.set_title(f"{DEFECT_KINDS[g['kind'][k]].replace('_', ' ')}\n"
                              f"c={g['contrast'][k]:.2f}", fontsize=7)
            bot.set_title(f"peak {g['peak'][k]:.3f}", fontsize=7)
        axes[0, 0].set_ylabel("test crop", fontsize=8)
        axes[1, 0].set_ylabel("|difference|", fontsize=8)
        sf.suptitle(title, fontsize=10, x=0.01, ha="left", fontweight="bold")
    fig.text(0.99, 0.003, "overlapping regime, validation pool; 32 × 32 crops centred on the "
             "candidate; green = its connected component", fontsize=8, ha="right",
             color=TEXT_SECONDARY)
    out = ("candidate_gallery.png" if tag == "stage1" else f"{tag}_candidate_gallery.png")
    fig.savefig(ROOT / "figures" / out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    tag = "stage1_quick" if args.quick else "stage1"
    if args.plot_only:
        plot(tag)
        return
    run(quick_config() if args.quick else Config(), tag)


if __name__ == "__main__":
    main()
