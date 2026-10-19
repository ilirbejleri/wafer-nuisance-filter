"""The stage-2 feature sets. Hand-crafted (k = 10) and frozen-CNN (d = 384, PCA to 4, 16, 64)
features, standardized, with an intercept, clipped to ||x|| <= C (feature-fit pool), then scaled
to ||x|| <= 1.

    uv run python experiments/features_report.py              # full run
    uv run python experiments/features_report.py --quick      # small pools, *_quick outputs
    uv run python experiments/features_report.py --plot-only  # redraw figures from results/

Outputs: results/features.json, figures/features.png.

Main-setting choice (rule fixed before this script was first run): the linear-filter experiments
(Results 2, 3, 4) use the overlapping regime if a linear classifier on the hand-crafted features
errs 5-20% on validation candidates. The check fits unregularized logistic regression on the
first half of the validation pool and measures its 0-1 error on the second half (validation
pool only, so no choice sees the test pool). The other feature sets are reported for information.
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
from wafer_nuisance.detect.stage1 import Stage1Config  # noqa: E402
from wafer_nuisance.eval.plotting import SERIES, TEXT_SECONDARY, apply_style  # noqa: E402
from wafer_nuisance.features.cache import (  # noqa: E402
    feature_maps,
    pool_features,
    tune_stage1,
)
from wafer_nuisance.features.handcrafted import HAND_FEATURES  # noqa: E402
from wafer_nuisance.learn import fit_logistic, fit_threshold, zero_one_error  # noqa: E402

MAIN_ERROR_RANGE = (0.05, 0.20)


@dataclass
class Config:
    regime: str = "overlapping"
    n_fit: int = POOLS["feature_fit"].size
    n_validation: int = POOLS["validation"].size
    target_recall: float = STAGE1_TARGET_RECALL
    stage1_regime: str = STAGE1_REGIME
    stage1: Stage1Config = field(default_factory=Stage1Config)
    delta: float = 0.05
    quick: bool = False


def quick_config() -> Config:
    return Config(n_fit=2048, n_validation=2048, quick=True)


def run(cfg: Config, tag: str) -> None:
    stage1, s1_info = tune_stage1(REGIMES[cfg.stage1_regime], cfg.stage1, cfg.target_recall)
    regime = REGIMES[cfg.regime]
    maps, fit = feature_maps(regime, stage1, cfg.n_fit)
    val = pool_features(regime, stage1, POOLS["validation"], cfg.n_validation)
    y = val["label"].astype(np.float64)
    half = len(y) // 2
    summary = {"stage1": s1_info, "regime": cfg.regime, "n_fit": int(len(fit["label"])),
               "n_validation": int(len(y)), "validation_split": [half, len(y) - half],
               "hand_features": list(HAND_FEATURES), "sets": {}}

    h = fit_threshold(val["peak"][:half], y[:half])
    e = float(h.error(val["peak"][half:], y[half:]))
    summary["threshold_peak_validation_error"] = {
        "value": e, "hoeffding_ci": hoeffding_interval(e, len(y) - half, cfg.delta)}

    for name, fm in maps.items():
        raw_fit, raw_val = fit[fm.source], val[fm.source]
        norms_fit = np.linalg.norm(fm.unclipped(raw_fit), axis=1)
        X = fm(raw_val)
        r = fit_logistic(X[:half], y[:half])
        err = zero_one_error(r.w, X[half:], y[half:])
        s = {"k": fm.k, "p": fm.dim, "source": fm.source, "C": fm.C,
             "fit_fraction_clipped": float(np.mean(norms_fit > fm.C)),
             "validation_fraction_clipped": float(np.mean(
                 np.linalg.norm(fm.unclipped(raw_val), axis=1) > fm.C)),
             "fit_norm_hist": np.histogram(norms_fit / fm.C, bins=np.linspace(0, 2, 41))[0]
             .tolist(),
             # mean ||x||^2 after scaling to ||x|| <= 1 (enters the Rademacher complexity term)
             "validation_mean_sq_norm": float((X * X).sum(1).mean()),
             "linear_validation_error": {
                 "value": err, "hoeffding_ci": hoeffding_interval(err, len(y) - half, cfg.delta)},
             "logistic_converged": r.converged, "logistic_separable": r.separable}
        if fm.eigvals is not None:
            ev = np.clip(fm.eigvals, 0, None)
            s["explained_variance"] = float(ev[:fm.k].sum() / ev.sum())
        summary["sets"][name] = s
        print(f"{name:7s} k={fm.k:3d} C={fm.C:7.3f} clipped(fit)={s['fit_fraction_clipped']:.3f}"
              f" linear val error={err:.4f}" + (" SEPARABLE" if r.separable else ""), flush=True)

    ev = np.clip(maps["pca4"].eigvals, 0, None)
    summary["cnn_cumulative_variance"] = (np.cumsum(ev) / ev.sum()).tolist()
    hand_err = summary["sets"]["hand"]["linear_validation_error"]["value"]
    summary["main_setting"] = {
        "rule": "overlapping if the hand-crafted linear validation error is in [0.05, 0.20]",
        "hand_linear_validation_error": hand_err,
        "in_range": bool(MAIN_ERROR_RANGE[0] <= hand_err <= MAIN_ERROR_RANGE[1]),
    }
    # Standardized hand-crafted features per class (validation), for the figure.
    Z = maps["hand"].unclipped(val["hand"])[:, :-1]
    bins = np.linspace(-4, 6, 51)
    summary["hand_hist"] = {"bins": bins.tolist(), "features": {
        f: {"nuisance": np.histogram(Z[y < 0, j], bins)[0].tolist(),
            "defect": np.histogram(Z[y > 0, j], bins)[0].tolist()}
        for j, f in enumerate(HAND_FEATURES)}}
    print(f"main setting: hand-crafted linear error {hand_err:.4f}, in 5-20%: "
          f"{summary['main_setting']['in_range']}")
    write_results(tag, cfg, summary, [], "mps/cpu")
    plot(tag)


def plot(tag: str) -> None:
    apply_style()
    s = read_results(tag)["summary"]
    fig = plt.figure(figsize=(16, 7.5), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.0, 1.0])
    axes = top.subplots(2, 5, sharey=False)
    bins = np.array(s["hand_hist"]["bins"])
    for ax, (f, hst) in zip(axes.ravel(), s["hand_hist"]["features"].items(), strict=True):
        for j, cls in enumerate(("nuisance", "defect")):
            c = np.array(hst[cls], float)
            ax.stairs(c / max(c.sum(), 1), bins, fill=True, alpha=0.35, color=SERIES[j],
                      label=cls)
        ax.set_title(f.replace("_", " "), fontsize=9)
        ax.set_yticks([])
    axes[0, 0].legend(fontsize=7)
    top.suptitle("hand-crafted features, standardized (validation candidates)", fontsize=10,
                 x=0.01, ha="left", fontweight="bold")

    a1, a2, a3 = bottom.subplots(1, 3)
    cum = np.array(s["cnn_cumulative_variance"])
    a1.plot(np.arange(1, len(cum) + 1), cum, color=SERIES[0])
    for k in (4, 16, 64):
        a1.axvline(k, color=TEXT_SECONDARY, ls=":", lw=1)
        a1.text(k, 0.03, f" k={k}", fontsize=8, color=TEXT_SECONDARY)
    a1.set(xscale="log", xlabel="PCA components", ylabel="cumulative variance",
           title="CNN features: PCA spectrum (feature-fit pool)", ylim=(0, 1.02))
    nb = np.linspace(0, 2, 41)
    for j, (name, st) in enumerate(s["sets"].items()):
        a2.stairs(np.array(st["fit_norm_hist"]) / sum(st["fit_norm_hist"]), nb,
                  color=SERIES[j % len(SERIES)], lw=1.4, label=f"{name} (C = {st['C']:.2f})")
    a2.axvline(1.0, color=TEXT_SECONDARY, ls="--", lw=1)
    a2.set(xlabel="‖v‖ / C before clipping", ylabel="fraction", title="norms (feature-fit pool)")
    a2.legend(fontsize=7)
    names = list(s["sets"])
    errs = [s["sets"][n]["linear_validation_error"] for n in names]
    lo = [e["value"] - e["hoeffding_ci"][0] for e in errs]
    hi = [e["hoeffding_ci"][1] - e["value"] for e in errs]
    a3.bar(names, [e["value"] for e in errs], yerr=[lo, hi], color=SERIES[0], alpha=0.7,
           capsize=3)
    t = s["threshold_peak_validation_error"]["value"]
    a3.axhline(t, color=SERIES[1], ls="--", lw=1.2, label=f"threshold on peak ({t:.3f})")
    a3.axhspan(*MAIN_ERROR_RANGE, color=TEXT_SECONDARY, alpha=0.08, label="5-20%")
    a3.set(ylabel="0-1 error (validation, second half)",
           title="linear classifier (unregularized logistic)")
    a3.legend(fontsize=7)
    out = "features.png" if tag == "features" else f"{tag}.png"
    fig.savefig(ROOT / "figures" / out, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    tag = "features_quick" if args.quick else "features"
    if args.plot_only:
        plot(tag)
        return
    run(quick_config() if args.quick else Config(), tag)


if __name__ == "__main__":
    main()
