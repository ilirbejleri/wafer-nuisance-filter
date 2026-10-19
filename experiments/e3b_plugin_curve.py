"""E3b (Result 3b): plug-in prediction of the learning-curve improvement from n to 4n labels,
using only the n labels already collected.

    uv run python experiments/e3b_plugin_curve.py              # full run
    uv run python experiments/e3b_plugin_curve.py --quick      # small pools, *_quick outputs
    uv run python experiments/e3b_plugin_curve.py --plot-only  # redraw the figure from results/

Outputs: results/e3b_plugin_curve.json, results/e3b_plugin_curve_raw.csv,
figures/e3b_plugin_curve.png.

Protocol (same setup as E3: overlapping regime, PCA-k CNN features, k in {4, 16, 64}, p = k + 1).

* Repeat r (200 repeats) draws a fresh training block of 4 * 100 * 65 = 26,000 candidates (train
  pool items r * 26,000, ..., (r + 1) * 26,000 - 1). For n in {10, 20, 50, 100} x p the training
  sets are nested: the n-sample is the first n candidates of the 4n-sample. theta_hat_n and
  theta_hat_4n = Newton's method, no regularization.
* Plug-in prediction, from the n labels only: H_n = mean phi phi x x^T and G_n = mean grad l
  grad l^T at theta_hat_n on the n-sample, predicted improvement
  Tr(H_n^-1 G_n) / 2 * (1/n - 1/(4n)) (bounds.predicted_improvement).
* Measured improvement: L_test(theta_hat_n) - L_test(theta_hat_4n), both on the same test pool
  (100,000 candidates); its pool noise floor is sd_i / sqrt(N_test) of the paired per-point
  differences. The measured 0-1 improvement (test error at n minus at 4n) is reported too, with
  no theory attached.
* References: the same prediction with the population H, G (E3's theta*, fitted on the first
  1,000,000 population candidates, H and G on the test pool) and the well-specified shortcut
  with p in place of Tr(H^-1 G).
* Runs whose n-sample is linearly separable (then theta_hat_n does not exist) are excluded and
  counted; a separable 4n-sample makes its n-prefix separable too.

Check (fixed before running): mean plug-in prediction / mean measured improvement in
[0.8, 1.25] at the largest n for each k. The spread of the per-run errors (prediction minus
measurement) is reported next to it: the theory predicts the expected improvement, not one
run's.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import matplotlib
import numpy as np
from _common import ROOT, read_results, write_results
from e3_learning_curve import _P, per_point_loss, population_quantities

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wafer_nuisance.bounds import predicted_improvement, trace_HinvG  # noqa: E402
from wafer_nuisance.data.config import (  # noqa: E402
    POOLS,
    REGIMES,
    STAGE1_REGIME,
    STAGE1_TARGET_RECALL,
)
from wafer_nuisance.detect.stage1 import Stage1Config  # noqa: E402
from wafer_nuisance.eval.plotting import REFERENCE, SERIES, TEXT_SECONDARY, apply_style  # noqa: E402
from wafer_nuisance.features.cache import feature_maps, pool_features, tune_stage1  # noqa: E402
from wafer_nuisance.learn import (  # noqa: E402
    fit_logistic,
    gradient_outer,
    logistic_hessian,
    zero_one_error,
)

RATIO_RANGE = (0.8, 1.25)


@dataclass
class Config:
    regime: str = "overlapping"
    feature_sets: tuple[str, ...] = ("pca4", "pca16", "pca64")
    n_over_p: tuple[int, ...] = (10, 20, 50, 100)
    factor: int = 4
    repeats: int = 200
    n_fit: int = POOLS["feature_fit"].size
    n_population: int = 1_000_000
    n_test: int = POOLS["test"].size
    target_recall: float = STAGE1_TARGET_RECALL
    stage1_regime: str = STAGE1_REGIME
    stage1: Stage1Config = field(default_factory=Stage1Config)
    quick: bool = False


def quick_config() -> Config:
    return Config(n_over_p=(10, 20), repeats=10, n_fit=2048, n_population=8192, n_test=4096,
                  quick=True)


def block_size(cfg: Config) -> int:
    return cfg.factor * max(cfg.n_over_p) * max(_P[s] for s in cfg.feature_sets)


# ---- worker --------------------------------------------------------------------------------

_W: dict = {}


def _setup(cfg: Config, maps: dict) -> None:
    stage1, _ = tune_stage1(REGIMES[cfg.stage1_regime], cfg.stage1, cfg.target_recall)
    test = pool_features(REGIMES[cfg.regime], stage1, POOLS["test"], cfg.n_test, workers=1)
    yt = test["label"].astype(float)
    _W.update(cfg=cfg, maps=maps, stage1=stage1, regime=REGIMES[cfg.regime],
              test={s: (maps[s](test["cnn"]), yt) for s in cfg.feature_sets})


def _task(r: int) -> list[dict]:
    cfg, maps = _W["cfg"], _W["maps"]
    nb = block_size(cfg)
    tr = pool_features(_W["regime"], _W["stage1"], POOLS["train"], nb, start=r * nb, workers=1)
    y_all = tr["label"].astype(float)
    rows = []
    for s in cfg.feature_sets:
        fm = maps[s]
        X_all = fm(tr["cnn"])
        Xt, yt = _W["test"][s]
        p = fm.dim
        for m in cfg.n_over_p:
            n = m * p
            X, y = X_all[:n], y_all[:n]
            small = fit_logistic(X, y)
            big = fit_logistic(X_all[:cfg.factor * n], y_all[:cfg.factor * n])
            row = {"set": s, "p": p, "n_over_p": m, "n": n, "repeat": r,
                   "separable_n": small.separable, "separable_4n": big.separable,
                   "converged_n": small.converged, "converged_4n": big.converged}
            if not small.separable and not big.separable:
                d = per_point_loss(small.w, Xt, yt) - per_point_loss(big.w, Xt, yt)
                H, G = logistic_hessian(small.w, X), gradient_outer(small.w, X, y)
                e_n, e_4n = zero_one_error(small.w, Xt, yt), zero_one_error(big.w, Xt, yt)
                row.update(improvement=float(d.mean()),
                           improvement_floor=float(d.std(ddof=1) / np.sqrt(len(d))),
                           plugin_trace=trace_HinvG(H, G),
                           plugin_prediction=float(predicted_improvement(H, G, n, cfg.factor)),
                           test_01_n=e_n, test_01_4n=e_4n, improvement_01=e_n - e_4n)
            rows.append(row)
    return rows


# ---- driver --------------------------------------------------------------------------------


def summarize(cfg: Config, rows: list[dict], popq: dict) -> dict:
    out = {}
    for s in cfg.feature_sets:
        pq = popq[s]
        per_n = []
        for m in cfg.n_over_p:
            R = [r for r in rows if r["set"] == s and r["n_over_p"] == m]
            ok = [r for r in R if not (r["separable_n"] or r["separable_4n"])]
            n = R[0]["n"]
            meas = np.array([r["improvement"] for r in ok])
            pred = np.array([r["plugin_prediction"] for r in ok])
            imp01 = np.array([r["improvement_01"] for r in ok])
            scale = 1 / n - 1 / (cfg.factor * n)
            pop_pred, p_pred = pq["trace_HinvG"] / 2 * scale, pq["p"] / 2 * scale
            mm = float(meas.mean())
            err = (pred - meas) / mm  # per-run prediction error, relative to the mean improvement
            per_n.append({
                "n_over_p": m, "n": n, "runs": len(R),
                "separable_runs": len(R) - len(ok),
                "separable_runs_4n": sum(r["separable_4n"] for r in R),
                "nonconverged_runs": sum(not (r["converged_n"] and r["converged_4n"]) for r in ok),
                "mean_improvement": mm,
                "se_improvement": float(meas.std(ddof=1) / np.sqrt(len(meas))),
                "sd_improvement": float(meas.std(ddof=1)),
                "fraction_negative_improvement": float(np.mean(meas < 0)),
                "mean_floor": float(np.mean([r["improvement_floor"] for r in ok])),
                "mean_plugin_prediction": float(pred.mean()),
                "sd_plugin_prediction": float(pred.std(ddof=1)),
                "plugin_prediction_q10_q90": np.quantile(pred, [0.1, 0.9]).tolist(),
                "mean_plugin_trace": float(np.mean([r["plugin_trace"] for r in ok])),
                "plugin_trace_ratio": float(np.mean([r["plugin_trace"] for r in ok])
                                            / pq["trace_HinvG"]),
                "population_prediction": pop_pred, "well_specified_prediction": p_pred,
                "ratio_plugin": float(pred.mean() / mm),
                "ratio_population": pop_pred / mm, "ratio_well_specified": p_pred / mm,
                "relative_error_sd": float(err.std(ddof=1)),
                "relative_error_q10_q50_q90": np.quantile(err, [0.1, 0.5, 0.9]).tolist(),
                "mean_improvement_01": float(imp01.mean()),
                "se_improvement_01": float(imp01.std(ddof=1) / np.sqrt(len(imp01))),
                "mean_test_01_n": float(np.mean([r["test_01_n"] for r in ok])),
                "mean_test_01_4n": float(np.mean([r["test_01_4n"] for r in ok])),
            })
        lo, hi = RATIO_RANGE
        last = per_n[-1]["ratio_plugin"]
        out[s] = {"p": pq["p"], "trace_HinvG": pq["trace_HinvG"],
                  "misspecification_ratio": pq["misspecification_ratio"], "per_n": per_n,
                  "checks": {"ratio_range": list(RATIO_RANGE), "ratio_at_largest_n": last,
                             "pass": bool(lo <= last <= hi)}}
    return out


def run(cfg: Config, tag: str) -> None:
    stage1, s1_info = tune_stage1(REGIMES[cfg.stage1_regime], cfg.stage1, cfg.target_recall)
    regime = REGIMES[cfg.regime]
    maps, _ = feature_maps(regime, stage1, cfg.n_fit)
    print("population quantities (as in E3):", flush=True)
    _, popq = population_quantities(cfg, maps, regime, stage1)
    rows: list[dict] = []
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(workers, initializer=_setup, initargs=(cfg, maps)) as ex:
        for k, out in enumerate(ex.map(_task, range(cfg.repeats))):
            rows += out
            if (k + 1) % 20 == 0 or k + 1 == cfg.repeats:
                print(f"  {k + 1} / {cfg.repeats} repeats", flush=True)
    summary = {"stage1": s1_info, "regime": cfg.regime, "sets": summarize(cfg, rows, popq),
               "theta_star_pool": f"population[0:{cfg.n_population}]",
               "H_G_pool": f"test[0:{cfg.n_test}]",
               "train_protocol": f"fresh dies per repeat: repeat r uses train-pool items "
                                 f"[r * {block_size(cfg)}, (r + 1) * {block_size(cfg)}); the "
                                 f"n-sample is a prefix of the {cfg.factor}n-sample"}
    for s, v in summary["sets"].items():
        q = v["per_n"][-1]
        print(f"{s}: plug-in / measured at n/p = {q['n_over_p']}: {q['ratio_plugin']:.3f} "
              f"(pass {v['checks']['pass']}); population {q['ratio_population']:.3f}, "
              f"p/2 {q['ratio_well_specified']:.3f}; per-run error sd "
              f"{q['relative_error_sd']:.2f}; separable runs "
              f"{sum(x['separable_runs'] for x in v['per_n'])}", flush=True)
    write_results(tag, cfg, summary, rows, "cpu")
    plot(tag)


def plot(tag: str) -> None:
    apply_style()
    res = read_results(tag)
    s, factor = res["summary"]["sets"], res["config"]["factor"]
    fig, axes = plt.subplots(1, len(s), figsize=(4.6 * len(s), 4.3), layout="constrained")
    for ax, (j, (name, v)) in zip(np.atleast_1d(axes), enumerate(s.items()), strict=True):
        q = v["per_n"]
        n = np.array([x["n"] for x in q], float)
        meas = n * np.array([x["mean_improvement"] for x in q])
        se = n * np.array([x["se_improvement"] for x in q])
        pred = n * np.array([x["mean_plugin_prediction"] for x in q])
        band = n[:, None] * np.array([x["plugin_prediction_q10_q90"] for x in q])
        c = (1 - 1 / factor) / 2
        ax.errorbar(n, meas, yerr=1.96 * se, fmt="o-", color=SERIES[j], capsize=3,
                    label=f"measured: n × improvement, n → {factor}n (mean ± 1.96 se)")
        ax.plot(n, pred, "s--", color=REFERENCE, ms=4, lw=1.4, label="plug-in prediction (mean)")
        ax.fill_between(n, band[:, 0], band[:, 1], color=REFERENCE, alpha=0.12, lw=0,
                        label="plug-in, 10–90% of runs")
        ax.axhline(c * v["trace_HinvG"], color=TEXT_SECONDARY, ls="-.", lw=1.1,
                   label=f"population H, G: {c * v['trace_HinvG']:.2f}")
        ax.axhline(c * v["p"], color=TEXT_SECONDARY, ls=":", lw=1.2,
                   label=f"p in place of Tr(H⁻¹G): {c * v['p']:.2f}")
        ax.set(xscale="log", xlabel="labels already collected n", title=f"{name} (p = {v['p']})")
        ax.set_xticks(n, [f"{int(x)}\n({c_['n_over_p']}p)" for x, c_ in zip(n, q, strict=True)],
                      fontsize=8)
        ax.minorticks_off()
        ax.legend(fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel(f"n × improvement from n to {factor}n labels")
    out = "e3b_plugin_curve.png" if tag == "e3b_plugin_curve" else f"{tag}.png"
    fig.savefig(ROOT / "figures" / out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    tag = "e3b_plugin_curve_quick" if args.quick else "e3b_plugin_curve"
    if args.plot_only:
        plot(tag)
        return
    run(quick_config() if args.quick else Config(), tag)


if __name__ == "__main__":
    main()
