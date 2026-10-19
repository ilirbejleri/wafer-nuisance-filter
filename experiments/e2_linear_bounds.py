"""E2a + E2b (Result 2): the linear filter's test error vs the VC and Rademacher bounds, and the
estimation / approximation trade-off across feature sets.

    uv run python experiments/e2_linear_bounds.py              # full run
    uv run python experiments/e2_linear_bounds.py --quick      # small pools, *_quick outputs
    uv run python experiments/e2_linear_bounds.py --plot-only  # redraw the figures from results/

Outputs: results/e2_linear_bounds.json, results/e2_linear_bounds_raw.csv (one row per set, n,
repeat), figures/e2a_bounds.png, figures/e2b_curves.png.

Protocol (overlapping regime, the main setting; see features_report.py). Repeat r draws a fresh
training sample of 2^15 candidates (train-pool items r * 2^15, ..., (r + 1) * 2^15 - 1); size n uses
its first n. For each feature set (feature maps fitted on the feature-fit pool, ||x|| <= 1, so
C = 1) and each B in the grid, w_B = argmin of the empirical logistic loss over ||w|| <= B
(projected gradient, warm-started along the increasing B grid). B is then chosen on the
validation pool (lowest validation 0-1 error, ties to the smaller B; every choice is made on
validation data), and the selected w is the filter of that run. Measured: its 0-1 error on the
test pool.

Bounds for the selected w: the VC bound (holds for every w at delta) and the Rademacher bound at
delta / G for its B (all G of them hold together, so choosing B afterwards is allowed). The
reported Rademacher bound uses the clipped loss psi(u) = min(ln(1 + e^-u), ln 2)
(``rademacher_bound_clipped``); training still minimizes the logistic loss, only the bound's
evaluation changes. The unclipped (logistic-loss) bound is recorded alongside for the looseness
comparison. Also recorded per run: the "certified" filter w_B with the B that minimizes its own
Rademacher bound (again allowed by the delta / G union bound), with its bound and test error,
for both versions of the bound; and the mean squared feature norm of the training sample.

E2b: L(h*) per feature set is estimated by the 0-1 test error of the unregularized logistic
minimizer on the population pool (exact 0-1 ERM over half-spaces is intractable; this h* is in
the class, so the estimate is an upper estimate of the best-in-class error). Estimation error =
L(h_hat) - L(h*) (approximation / estimation decomposition). Crossover n for a pair of sets: the
first grid n at which the larger set's mean test error drops below the smaller set's after being
above it.

Check (fixed before running): both bounds above the measured test error in >= 95% of runs
(per feature set). The write-up (theory_tables.py) reports the first n at which each bound's mean
over repeats beats the trivial classifier, i.e. is below min(pi, 1 - pi) with pi the positive rate
under Q on the population pool (results/stage1.json); the E2a figure draws that level.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import matplotlib
import numpy as np
from _common import ROOT, log_ticks, read_results, write_results

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wafer_nuisance.bounds import (  # noqa: E402
    clipped_logistic_loss,
    hoeffding_interval,
    rademacher_bound,
    rademacher_bound_clipped,
    vc_bound,
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
from wafer_nuisance.learn import (  # noqa: E402
    fit_logistic,
    fit_logistic_ball,
    logistic_loss,
    zero_one_error,
)

PASS_FRACTION = 0.95


@dataclass
class Config:
    regime: str = "overlapping"
    feature_sets: tuple[str, ...] = ("hand", "pca16", "pca64", "cnn384")
    log2_n: tuple[int, ...] = tuple(range(7, 16))
    repeats: int = 50
    # A first grid {0.5, ..., 16}, extended by powers of 2 to 256: with every x scaled
    # to ||x|| <= 1, B = 16 corresponds to B = 16 / C_raw in the raw standardized units (0.4
    # for CNN-384) and cut the CNN-384 filter's validation error from 0.07 (B = 256) to 0.12
    # (checked on feature-fit training data and the validation pool before any E2 run).
    B_grid: tuple[float, ...] = tuple(2.0**k for k in range(-1, 9))
    n_fit: int = POOLS["feature_fit"].size
    n_validation: int = POOLS["validation"].size
    n_population: int = POOLS["population"].size
    n_test: int = POOLS["test"].size
    tol: float = 1e-8  # gradient-mapping norm at which projected gradient stops
    max_iter: int = 100_000
    target_recall: float = STAGE1_TARGET_RECALL
    stage1_regime: str = STAGE1_REGIME
    stage1: Stage1Config = field(default_factory=Stage1Config)
    delta: float = 0.05
    quick: bool = False


def quick_config() -> Config:
    return Config(log2_n=tuple(range(7, 12)), repeats=4, n_fit=2048, n_validation=2048,
                  n_population=8192, n_test=4096, quick=True)


# ---- one (repeat, feature set) task, run in a worker process -------------------------------

_POOLS: dict = {}


def _setup(cfg: Config, maps: dict) -> None:
    stage1, _ = tune_stage1(REGIMES[cfg.stage1_regime], cfg.stage1, cfg.target_recall)
    regime = REGIMES[cfg.regime]
    val = pool_features(regime, stage1, POOLS["validation"], cfg.n_validation, workers=1)
    test = pool_features(regime, stage1, POOLS["test"], cfg.n_test, workers=1)
    _POOLS.update(cfg=cfg, maps=maps, stage1=stage1, regime=regime,
                  val={s: (maps[s](val[maps[s].source]), val["label"].astype(float))
                       for s in cfg.feature_sets},
                  test={s: (maps[s](test[maps[s].source]), test["label"].astype(float))
                        for s in cfg.feature_sets})


def _task(args) -> list[dict]:
    r, set_name = args
    cfg, fm = _POOLS["cfg"], _POOLS["maps"][set_name]
    n_max = 2 ** max(cfg.log2_n)
    tr = pool_features(_POOLS["regime"], _POOLS["stage1"], POOLS["train"], n_max,
                       start=r * n_max, workers=1)
    X_all, y_all = fm(tr[fm.source]), tr["label"].astype(float)
    Xv, yv = _POOLS["val"][set_name]
    Xt, yt = _POOLS["test"][set_name]
    G = len(cfg.B_grid)
    rows = []
    for log2n in cfg.log2_n:
        n = 2**log2n
        X, y = X_all[:n], y_all[:n]
        w0, per_b = None, []
        for B in cfg.B_grid:
            res = fit_logistic_ball(X, y, B, tol=cfg.tol, max_iter=cfg.max_iter, w0=w0)
            w0 = res.w
            psi = clipped_logistic_loss(res.w, X, y)
            per_b.append({"B": B, "w": res.w, "train_log": res.loss, "train_psi": psi,
                          "train_01": zero_one_error(res.w, X, y),
                          "val_01": zero_one_error(res.w, Xv, yv),
                          "test_01": zero_one_error(res.w, Xt, yt),
                          "rad": rademacher_bound_clipped(psi, X, B, 1.0, cfg.delta / G),
                          "rad_unclipped": rademacher_bound(res.loss, X, B, 1.0, cfg.delta / G),
                          "converged": res.converged, "grad_map": res.grad_norm,
                          "iters": res.iterations})
        j = min(range(G), key=lambda i: (per_b[i]["val_01"], i))
        sel = per_b[j]
        test_err = sel["test_01"]
        jr = min(range(G), key=lambda i: (per_b[i]["rad"], i))  # Rademacher-certified filter
        ju = min(range(G), key=lambda i: (per_b[i]["rad_unclipped"], i))
        rows.append({
            "set": set_name, "n": n, "repeat": r, "B": sel["B"], "test_error": test_err,
            "train_01": sel["train_01"], "train_log": sel["train_log"],
            "train_psi": sel["train_psi"], "mean_sq_norm": float((X * X).sum(1).mean()),
            "val_01": sel["val_01"],
            # n < k + 1: the Sauer-Shelah form (2en/d)^d does not apply; the bound is vacuous.
            "vc_bound": float(vc_bound(sel["train_01"], n, fm.k, cfg.delta)) if n > fm.k
            else float("inf"),
            "rad_bound": sel["rad"],
            "rad_bound_unclipped": sel["rad_unclipped"],
            "rad_cert_B": per_b[jr]["B"],
            "rad_cert_bound": per_b[jr]["rad"],
            "rad_cert_test_error": per_b[jr]["test_01"],
            "rad_unclipped_cert_B": per_b[ju]["B"],
            "rad_unclipped_cert_bound": per_b[ju]["rad_unclipped"],
            "rad_unclipped_cert_test_error": per_b[ju]["test_01"],
            "converged_all_B": all(b["converged"] for b in per_b),
            "max_grad_map": max(b["grad_map"] for b in per_b),
            "max_iters": max(b["iters"] for b in per_b),
        })
    return rows


# ---- driver --------------------------------------------------------------------------------


def best_in_class(cfg: Config, maps: dict, regime, stage1) -> dict:
    pop = pool_features(regime, stage1, POOLS["population"], cfg.n_population)
    test = pool_features(regime, stage1, POOLS["test"], cfg.n_test)
    out = {}
    for s in cfg.feature_sets:
        fm = maps[s]
        X, y = fm(pop[fm.source]), pop["label"].astype(float)
        res = fit_logistic(X, y)
        Xt, yt = fm(test[fm.source]), test["label"].astype(float)
        e = zero_one_error(res.w, Xt, yt)
        out[s] = {"L_star_test": {"value": e, "hoeffding_ci": hoeffding_interval(e, len(yt),
                                                                                  cfg.delta)},
                  "L_star_population_01": zero_one_error(res.w, X, y),
                  "population_log_loss": logistic_loss(res.w, X, y),
                  "population_psi_loss": clipped_logistic_loss(res.w, X, y),
                  "converged": res.converged, "separable": res.separable,
                  "norm_w": float(np.linalg.norm(res.w))}
        print(f"  L(h*) {s}: {e:.4f} (||w|| = {out[s]['norm_w']:.1f})", flush=True)
    return out


def summarize(cfg: Config, rows: list[dict], lstar: dict) -> dict:
    ns = [2**k for k in cfg.log2_n]
    sets = {}
    for s in cfg.feature_sets:
        R = [r for r in rows if r["set"] == s]
        by_n = {n: [r for r in R if r["n"] == n] for n in ns}

        def col(key, by_n=by_n):
            return np.array([[r[key] for r in by_n[n]] for n in ns])  # (len(ns), repeats)

        err, vc, rad = col("test_error"), col("vc_bound"), col("rad_bound")
        rad_u = col("rad_bound_unclipped")
        mean_vc, mean_rad = vc.mean(1), rad.mean(1)
        Bsel = col("B")
        sets[s] = {
            "n": ns, "mean_test_error": err.mean(1).tolist(),
            "std_test_error": err.std(1, ddof=1).tolist(),
            "mean_vc_bound": mean_vc.tolist(), "mean_rad_bound": mean_rad.tolist(),
            "mean_rad_bound_unclipped": rad_u.mean(1).tolist(),
            "mean_rad_cert_bound": col("rad_cert_bound").mean(1).tolist(),
            "mean_rad_cert_test_error": col("rad_cert_test_error").mean(1).tolist(),
            "mean_rad_unclipped_cert_bound": col("rad_unclipped_cert_bound").mean(1).tolist(),
            "mean_rad_unclipped_cert_test_error":
                col("rad_unclipped_cert_test_error").mean(1).tolist(),
            "mean_train_log": col("train_log").mean(1).tolist(),
            "mean_train_psi": col("train_psi").mean(1).tolist(),
            "mean_sq_norm": col("mean_sq_norm").mean(1).tolist(),
            "rad_cert_B_counts": {str(b): (col("rad_cert_B") == b).sum(1).tolist()
                                  for b in cfg.B_grid},
            "mean_train_01": col("train_01").mean(1).tolist(),
            "B_selected_counts": {str(b): (Bsel == b).sum(1).tolist() for b in cfg.B_grid},
            "estimation_error": (err.mean(1) - lstar[s]["L_star_test"]["value"]).tolist(),
            "fraction_vc_above": float((vc > err).mean()),
            "fraction_rad_above": float((rad > err).mean()),
            "fraction_rad_unclipped_above": float((rad_u > err).mean()),
            "fraction_runs_all_B_converged": float(col("converged_all_B").mean()),
            "max_grad_map": float(col("max_grad_map").max()),
            "L_star": lstar[s],
        }
        sets[s]["checks"] = {
            "vc_above_in_95pct": sets[s]["fraction_vc_above"] >= PASS_FRACTION,
            "rad_above_in_95pct": sets[s]["fraction_rad_above"] >= PASS_FRACTION,
        }
    names = list(cfg.feature_sets)
    cross = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ea, eb = np.array(sets[a]["mean_test_error"]), np.array(sets[b]["mean_test_error"])
            n_cross = next((ns[j] for j in range(1, len(ns))
                            if eb[j] < ea[j] and eb[j - 1] >= ea[j - 1]), None)
            cross[f"{a}_vs_{b}"] = n_cross
    return {"sets": sets, "crossover_n": cross}


def run(cfg: Config, tag: str) -> None:
    stage1, s1_info = tune_stage1(REGIMES[cfg.stage1_regime], cfg.stage1, cfg.target_recall)
    regime = REGIMES[cfg.regime]
    maps, _ = feature_maps(regime, stage1, cfg.n_fit)
    print("best-in-class (population pool):", flush=True)
    lstar = best_in_class(cfg, maps, regime, stage1)
    tasks = [(r, s) for r in range(cfg.repeats) for s in reversed(cfg.feature_sets)]
    rows: list[dict] = []
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(workers, initializer=_setup, initargs=(cfg, maps)) as ex:
        for k, out in enumerate(ex.map(_task, tasks)):
            rows += out
            if (k + 1) % 10 == 0 or k + 1 == len(tasks):
                print(f"  {k + 1} / {len(tasks)} tasks", flush=True)
    summary = summarize(cfg, rows, lstar)
    summary.update(stage1=s1_info, regime=cfg.regime,
                   C={s: maps[s].C for s in cfg.feature_sets},
                   train_protocol="fresh dies per repeat: repeat r uses train-pool items "
                                  "[r * 2^15, (r + 1) * 2^15), nested prefixes")
    for s, v in summary["sets"].items():
        print(f"{s}: VC above in {v['fraction_vc_above']:.3f} of runs, Rademacher above in "
              f"{v['fraction_rad_above']:.3f}", flush=True)
    print("crossovers:", summary["crossover_n"])
    write_results(tag, cfg, summary, rows, "cpu")
    plot(tag)


# ---- figures -------------------------------------------------------------------------------


def plot(tag: str) -> None:
    apply_style()
    s = read_results(tag)["summary"]
    sets = s["sets"]
    trivial = trivial_error(quick=tag != "e2_linear_bounds")
    fig, axes = plt.subplots(1, len(sets), figsize=(4.4 * len(sets), 4.9), sharey=True,
                             layout="constrained")
    for ax, (name, v) in zip(np.atleast_1d(axes), sets.items(), strict=True):
        ns = np.array(v["n"], float)
        ax.plot(ns, v["mean_vc_bound"], "--", color=SERIES[1], label="VC bound")
        ax.plot(ns, v["mean_rad_bound"], "--", color=SERIES[2],
                label="Rademacher bound (clipped loss)")
        m, sd = np.array(v["mean_test_error"]), np.array(v["std_test_error"])
        ax.fill_between(ns, m - sd, m + sd, color=SERIES[0], alpha=0.15, lw=0)
        ax.plot(ns, m, "o-", color=SERIES[0], label="test error (mean ± sd)")
        ax.plot(ns, v["mean_rad_cert_bound"], ":", color=SERIES[2], lw=1.6,
                label="Rademacher, B minimizing it")
        ax.plot(ns, v["mean_rad_cert_test_error"], "s:", ms=3.5, color=SERIES[3],
                label="test error of that filter")
        ax.axhline(v["L_star"]["L_star_test"]["value"], color=REFERENCE, ls=":", lw=1.2,
                   label="L(h*) estimate")
        ax.axhline(trivial, color=TEXT_SECONDARY, lw=1.0, ls="-.",
                   label=f"trivial classifier min(π, 1 − π) = {trivial:.3f}")
        ax.set(xscale="log", yscale="log", xlabel="training candidates n",
               title=f"{name} (k = {_k(name)})")
        log_ticks(ax, ns[::2])
    np.atleast_1d(axes)[0].set_ylabel("0-1 error / bound")
    fig.legend(*np.atleast_1d(axes)[0].get_legend_handles_labels(), loc="outside lower center",
               ncol=4, fontsize=8)
    sfx = "" if tag == "e2_linear_bounds" else "_quick"
    fig.savefig(ROOT / "figures" / f"e2a_bounds{sfx}.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
    for j, (name, v) in enumerate(sets.items()):
        ns = np.array(v["n"], float)
        ax.plot(ns, v["mean_test_error"], "o-", color=SERIES[j], label=name)
        ax.axhline(v["L_star"]["L_star_test"]["value"], color=SERIES[j], ls=":", lw=1.2)
    ax.set(xscale="log", xlabel="training candidates n", ylabel="mean test 0-1 error",
           title="learning curves; dotted = L(h*) estimate per feature set")
    log_ticks(ax, np.array(next(iter(sets.values()))["n"], float), y_log=False)
    ax.legend()
    fig.savefig(ROOT / "figures" / f"e2b_curves{sfx}.png", dpi=130)
    plt.close(fig)


def trivial_error(quick: bool = False) -> float:
    """min(pi, 1 - pi): the 0-1 error of always predicting the majority label, with pi the
    positive rate under Q on the main regime's population pool (results/stage1.json, or
    stage1_quick.json for a --quick run)."""
    s = read_results("stage1_quick" if quick else "stage1")["summary"]
    pi = s["regimes"]["overlapping"]["positive_rate"]["value"]
    return min(pi, 1 - pi)


def _k(name: str) -> int:
    return {"hand": 10, "pca4": 4, "pca16": 16, "pca64": 64, "cnn384": 384}[name]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    tag = "e2_linear_bounds_quick" if args.quick else "e2_linear_bounds"
    if args.plot_only:
        plot(tag)
        return
    run(quick_config() if args.quick else Config(), tag)


if __name__ == "__main__":
    main()
