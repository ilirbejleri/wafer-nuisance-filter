"""E3 (Result 3): the learning curve of unregularized logistic regression vs the
asymptotic prediction Tr(H^-1 G) / (2n).

    uv run python experiments/e3_learning_curve.py              # full run
    uv run python experiments/e3_learning_curve.py --quick      # small pools, *_quick outputs
    uv run python experiments/e3_learning_curve.py --plot-only  # redraw the figure from results/

Outputs: results/e3_learning_curve.json, results/e3_learning_curve_raw.csv,
figures/e3_learning_curve.png.

Protocol (overlapping regime; PCA-k CNN features, k in {4, 16, 64}, p = k + 1 parameters with
the intercept; ||x|| <= 1 as in features_report.py).

* theta* = unregularized logistic minimizer on the first 1,000,000 candidates of the population
  pool. The pool is larger than its 200,000 default because the finite-pool error of theta*
  enters the measured excess at relative size ~ n / N_population (asymptotic normality
  applied to theta* itself); at the largest n = 300 * 65 = 19,500 that is ~2% with N = 10^6 instead
  of ~10% with 2 x 10^5 (decided before any E3 run).
* H = mean phi(theta*.x) phi(-theta*.x) x x^T and G = mean grad l grad l^T at theta*, both on the
  test pool (100,000 candidates, disjoint from the population and training pools). As a check on
  their precision the prediction is also computed on each half of the test pool.
* Repeat r (200 repeats) draws a fresh training sample of 300 * 65 = 19,500 candidates (train
  pool items r * 19,500, ..., (r + 1) * 19,500 - 1); size n uses its first n, n in
  {10, 20, 50, 100, 300} x p. theta_hat = Newton's method, no regularization;
  runs whose training sample is linearly separable are flagged and excluded.
* Measured: n * (L_test(theta_hat) - L_test(theta*)), the excess logistic loss with both terms on
  the same test pool, and n * Cov(theta_hat) over the repeats, compared with Tr(H^-1 G) / 2 and
  H^-1 G H^-1 (p / 2 is the well-specified value). Wald coverage: per run, the interval
  theta_hat_j +- 1.96 sqrt([H_n^-1 G_n H_n^-1]_jj / n) with H_n, G_n estimated on the training
  sample at theta_hat (sandwich covariance); coverage = fraction of (run, coordinate) pairs whose
  interval contains theta*_j.

Pool noise floor: the per-run measured excess is a mean of N_test paired differences
l_i(theta_hat) - l_i(theta*); its standard deviation from the finite pool is estimated by
sd_i / sqrt(N_test) and reported (times n, and averaged over runs) next to n * excess.

Check (fixed before running): measured / predicted in [0.9, 1.1] at the largest n for each
k; report the smallest n/p from which the ratio stays in range, and the Wald coverage.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import matplotlib
import numpy as np
from _common import ROOT, read_results, write_results

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.special import log_expit  # noqa: E402

from wafer_nuisance.bounds import (  # noqa: E402
    misspecification_ratio,
    sandwich,
    trace_HinvG,
)
from wafer_nuisance.data.config import (  # noqa: E402
    POOLS,
    REGIMES,
    STAGE1_REGIME,
    STAGE1_TARGET_RECALL,
)
from wafer_nuisance.detect.stage1 import Stage1Config  # noqa: E402
from wafer_nuisance.eval.plotting import REFERENCE, SERIES, TEXT_SECONDARY, apply_style  # noqa: E402
from wafer_nuisance.features.cache import feature_maps, pool_features, tune_stage1  # noqa: E402
from wafer_nuisance.learn import fit_logistic, gradient_outer, logistic_hessian  # noqa: E402

RATIO_RANGE = (0.9, 1.1)
Z95 = 1.959963984540054  # standard normal 97.5% quantile


@dataclass
class Config:
    regime: str = "overlapping"
    feature_sets: tuple[str, ...] = ("pca4", "pca16", "pca64")
    n_over_p: tuple[int, ...] = (10, 20, 50, 100, 300)
    repeats: int = 200
    n_fit: int = POOLS["feature_fit"].size
    n_population: int = 1_000_000
    n_test: int = POOLS["test"].size
    target_recall: float = STAGE1_TARGET_RECALL
    stage1_regime: str = STAGE1_REGIME
    stage1: Stage1Config = field(default_factory=Stage1Config)
    quick: bool = False


def quick_config() -> Config:
    return Config(n_over_p=(10, 20, 50), repeats=10, n_fit=2048, n_population=8192,
                  n_test=4096, quick=True)


def block_size(cfg: Config) -> int:
    return max(cfg.n_over_p) * max(_P[s] for s in cfg.feature_sets)


_P = {"pca4": 5, "pca16": 17, "pca64": 65}


def per_point_loss(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -log_expit(y * (X @ theta))


# ---- worker --------------------------------------------------------------------------------

_W: dict = {}


def _setup(cfg: Config, maps: dict, theta_star: dict) -> None:
    stage1, _ = tune_stage1(REGIMES[cfg.stage1_regime], cfg.stage1, cfg.target_recall)
    regime = REGIMES[cfg.regime]
    test = pool_features(regime, stage1, POOLS["test"], cfg.n_test, workers=1)
    yt = test["label"].astype(float)
    _W.update(cfg=cfg, maps=maps, stage1=stage1, regime=regime, theta_star=theta_star,
              test={s: (maps[s](test["cnn"]), yt) for s in cfg.feature_sets},
              loss_star={s: per_point_loss(theta_star[s], maps[s](test["cnn"]), yt)
                         for s in cfg.feature_sets})


def _task(r: int) -> list[dict]:
    cfg, maps = _W["cfg"], _W["maps"]
    nb = block_size(cfg)
    tr = pool_features(_W["regime"], _W["stage1"], POOLS["train"], nb, start=r * nb, workers=1)
    y_all = tr["label"].astype(float)
    rows = []
    for s in cfg.feature_sets:
        fm, ts = maps[s], _W["theta_star"][s]
        X_all = fm(tr["cnn"])
        Xt, yt = _W["test"][s]
        p = fm.dim
        for m in cfg.n_over_p:
            n = m * p
            X, y = X_all[:n], y_all[:n]
            res = fit_logistic(X, y)
            row = {"set": s, "p": p, "n_over_p": m, "n": n, "repeat": r,
                   "separable": res.separable, "converged": res.converged,
                   "newton_iters": res.iterations, "grad_norm": res.grad_norm}
            if not res.separable:
                d = per_point_loss(res.w, Xt, yt) - _W["loss_star"][s]
                cov_hat = sandwich(logistic_hessian(res.w, X), gradient_outer(res.w, X, y)) / n
                half = Z95 * np.sqrt(np.diag(cov_hat))
                row.update(n_excess=float(n * d.mean()),
                           n_floor=float(n * d.std(ddof=1) / np.sqrt(len(d))),
                           covered=int(np.sum(np.abs(res.w - ts) <= half)),
                           delta=(res.w - ts).tolist())
            rows.append(row)
    return rows


# ---- driver --------------------------------------------------------------------------------


def population_quantities(cfg: Config, maps: dict, regime, stage1) -> tuple[dict, dict]:
    pop = pool_features(regime, stage1, POOLS["population"], cfg.n_population)
    test = pool_features(regime, stage1, POOLS["test"], cfg.n_test)
    yp, yt = pop["label"].astype(float), test["label"].astype(float)
    theta_star, out = {}, {}
    half = len(yt) // 2
    for s in cfg.feature_sets:
        fm = maps[s]
        res = fit_logistic(fm(pop["cnn"]), yp)
        th = res.w
        Xt = fm(test["cnn"])
        H, G = logistic_hessian(th, Xt), gradient_outer(th, Xt, yt)
        halves = [trace_HinvG(logistic_hessian(th, Xt[sl]), gradient_outer(th, Xt[sl], yt[sl]))
                  for sl in (slice(0, half), slice(half, None))]
        theta_star[s] = th
        out[s] = {"p": fm.dim, "theta_star_converged": res.converged,
                  "theta_star_separable": res.separable, "theta_star_norm": float(
                      np.linalg.norm(th)),
                  "trace_HinvG": trace_HinvG(H, G), "trace_HinvG_test_halves": halves,
                  "predicted_n_excess": trace_HinvG(H, G) / 2,
                  "well_specified_n_excess": fm.dim / 2,
                  "misspecification_ratio": misspecification_ratio(H, G),
                  "sandwich": sandwich(H, G).tolist()}
        print(f"  {s}: Tr(H^-1 G)/2 = {out[s]['predicted_n_excess']:.3f} (p/2 = {fm.dim / 2}), "
              f"halves {[round(h / 2, 3) for h in halves]}", flush=True)
    return theta_star, out


def summarize(cfg: Config, rows: list[dict], popq: dict) -> dict:
    out = {}
    for s in cfg.feature_sets:
        pq = popq[s]
        V = np.array(pq["sandwich"])
        per_n = []
        for m in cfg.n_over_p:
            R = [r for r in rows if r["set"] == s and r["n_over_p"] == m]
            ok = [r for r in R if not r["separable"]]
            n = R[0]["n"]
            ne = np.array([r["n_excess"] for r in ok])
            D = np.array([r["delta"] for r in ok])
            nc = n * np.cov(D, rowvar=False)
            per_n.append({
                "n_over_p": m, "n": n, "runs": len(R), "separable_runs": len(R) - len(ok),
                "nonconverged_runs": sum(not r["converged"] for r in R),
                "mean_n_excess": float(ne.mean()),
                "se_n_excess": float(ne.std(ddof=1) / np.sqrt(len(ne))),
                "mean_n_floor": float(np.mean([r["n_floor"] for r in ok])),
                "ratio": float(ne.mean() / pq["predicted_n_excess"]),
                "n_cov_trace_ratio": float(np.trace(nc) / np.trace(V)),
                "n_cov_rel_frobenius_error": float(np.linalg.norm(nc - V) / np.linalg.norm(V)),
                "wald_coverage": float(sum(r["covered"] for r in ok) / (len(ok) * pq["p"])),
            })
        lo, hi = RATIO_RANGE
        inr = [lo <= q["ratio"] <= hi for q in per_n]
        from_idx = next((i for i in range(len(inr)) if all(inr[i:])), None)
        out[s] = {**{k: v for k, v in pq.items() if k != "sandwich"}, "per_n": per_n,
                  "checks": {"ratio_range": list(RATIO_RANGE), "ratio_at_largest_n":
                             per_n[-1]["ratio"], "pass": bool(inr[-1]),
                             "smallest_n_over_p_in_range_from_then_on":
                                 None if from_idx is None else per_n[from_idx]["n_over_p"]}}
    return out


def run(cfg: Config, tag: str) -> None:
    stage1, s1_info = tune_stage1(REGIMES[cfg.stage1_regime], cfg.stage1, cfg.target_recall)
    regime = REGIMES[cfg.regime]
    maps, _ = feature_maps(regime, stage1, cfg.n_fit)
    print("population quantities:", flush=True)
    theta_star, popq = population_quantities(cfg, maps, regime, stage1)
    rows: list[dict] = []
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(workers, initializer=_setup,
                             initargs=(cfg, maps, theta_star)) as ex:
        for k, out in enumerate(ex.map(_task, range(cfg.repeats))):
            rows += out
            if (k + 1) % 20 == 0 or k + 1 == cfg.repeats:
                print(f"  {k + 1} / {cfg.repeats} repeats", flush=True)
    summary = {"stage1": s1_info, "regime": cfg.regime, "sets": summarize(cfg, rows, popq),
               "theta_star_pool": f"population[0:{cfg.n_population}]",
               "H_G_pool": f"test[0:{cfg.n_test}]",
               "train_protocol": f"fresh dies per repeat: repeat r uses train-pool items "
                                 f"[r * {block_size(cfg)}, (r + 1) * {block_size(cfg)}), "
                                 f"nested prefixes"}
    for s, v in summary["sets"].items():
        q = v["per_n"][-1]
        print(f"{s}: ratio at n/p = {q['n_over_p']}: {q['ratio']:.3f} (pass {v['checks']['pass']})"
              f"; in range from n/p = {v['checks']['smallest_n_over_p_in_range_from_then_on']};"
              f" Wald coverage {q['wald_coverage']:.3f}; separable runs "
              f"{sum(x['separable_runs'] for x in v['per_n'])}", flush=True)
    flat = [{k: v for k, v in r.items() if k != "delta"} for r in rows]
    write_results(tag, cfg, summary, flat, "cpu")
    plot(tag)


def plot(tag: str) -> None:
    apply_style()
    s = read_results(tag)["summary"]["sets"]
    fig, axes = plt.subplots(1, len(s), figsize=(4.6 * len(s), 4.3), layout="constrained")
    for ax, (j, (name, v)) in zip(np.atleast_1d(axes), enumerate(s.items()), strict=True):
        q = v["per_n"]
        n = np.array([x["n"] for x in q], float)
        m = np.array([x["mean_n_excess"] for x in q])
        se = np.array([x["se_n_excess"] for x in q])
        ax.errorbar(n, m, yerr=1.96 * se, fmt="o-", color=SERIES[j], capsize=3,
                    label="n · excess logistic loss (mean ± 1.96 se)")
        ax.axhline(v["predicted_n_excess"], color=REFERENCE, ls="--", lw=1.6,
                   label=f"Tr(H⁻¹G)/2 = {v['predicted_n_excess']:.2f}")
        ax.axhspan(RATIO_RANGE[0] * v["predicted_n_excess"],
                   RATIO_RANGE[1] * v["predicted_n_excess"], color=REFERENCE, alpha=0.12, lw=0,
                   label="±10% (pass band)")
        ax.axhline(v["well_specified_n_excess"], color=TEXT_SECONDARY, ls=":", lw=1.2,
                   label=f"p/2 = {v['well_specified_n_excess']:g} (well specified)")
        ax.plot(n, [x["mean_n_floor"] for x in q], "x:", color=TEXT_SECONDARY, ms=4, lw=0.8,
                label="n · pool noise floor")
        ax.set(xscale="log", xlabel="training candidates n",
               title=f"{name} (p = {v['p']})")
        ax.set_xticks(n, [f"{int(x)}\n({c['n_over_p']}p)" for x, c in zip(n, q, strict=True)],
                      fontsize=8)
        ax.minorticks_off()
        ax.legend(fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("n × excess risk")
    out = "e3_learning_curve.png" if tag == "e3_learning_curve" else f"{tag}.png"
    fig.savefig(ROOT / "figures" / out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    tag = "e3_learning_curve_quick" if args.quick else "e3_learning_curve"
    if args.plot_only:
        plot(tag)
        return
    run(quick_config() if args.quick else Config(), tag)


if __name__ == "__main__":
    main()
