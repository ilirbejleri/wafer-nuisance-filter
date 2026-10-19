"""E1 (Result 1): learning curves of threshold ERM on peak contrast vs the VC rates.

    uv run python experiments/e1_thresholds.py              # full run
    uv run python experiments/e1_thresholds.py --quick      # small pools, *_quick outputs
    uv run python experiments/e1_thresholds.py --plot-only  # redraw the figure from results/

Outputs: results/e1_thresholds.json, results/e1_thresholds_raw.csv (one row per regime, n, repeat),
figures/e1_thresholds.png.

Protocol. Repeat r draws a fresh training sample of 2^14 candidates (train pool items
r * 2^14, ..., (r + 1) * 2^14 - 1: disjoint dies for every repeat, so repeats are independent);
the sample of size n is its first n candidates (nested curves within a repeat). h_hat = exact
threshold ERM (``learn.threshold``). theta* = threshold ERM on the population pool (200,000
candidates). Both h_hat and h* are evaluated on the same test pool (100,000 candidates) so that
most of the pool's sampling noise cancels in the difference.

* separable regime: measured quantity = test error of h_hat; theory = realizable bound
  (d = 1);
* overlapping regime: measured quantity = excess error err_test(h_hat) - err_test(h*); theory =
  agnostic bound (d = 1).

Realizability (separable regime). The realizable bound assumes L(h*) = 0. The generator does not
guarantee it (sensor noise and the nuisance parameters are Gaussian), and the population pool's zero
errors also chose theta*. Recorded instead: theta*'s errors on the independent test pool (an upper
bound on L(h*) follows from ``bounds.violation_rate_bound``), and, per run, the ERM's training
error. The proof of the realizable bound shows
P(h_hat consistent with S and L(h_hat) > eps) <= delta without using realizability, so conditional
on a consistent sample the bound holds at level delta / P(consistent).

Noise floor (``eval.errors``): the measured excess at n is a mean over repeats of paired
differences on one N-point test pool; its pool-induced standard deviation is at most
sqrt(dbar(n) / N), dbar(n) = mean fraction of the test pool between theta_hat and theta*. Slopes
are fitted (least squares on log mean vs log n) only at the n where the mean is >= 5 x floor,
and only if there are at least 3 such n.

Checks (fixed before running): separable slope in [-1.15, -0.85]; overlapping slope in
[-1.15, -0.45]; the bound is above the mean curve at every n.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import matplotlib
import numpy as np
from _common import ROOT, log_ticks, loglog_slope, read_results, write_results

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wafer_nuisance.bounds import (  # noqa: E402
    agnostic_bound,
    hoeffding_interval,
    realizable_bound,
)
from wafer_nuisance.data.config import (  # noqa: E402
    POOLS,
    REGIMES,
    STAGE1_REGIME,
    STAGE1_TARGET_RECALL,
)
from wafer_nuisance.detect.stage1 import Stage1Config  # noqa: E402
from wafer_nuisance.eval.errors import ThresholdEvaluator, noise_floor  # noqa: E402
from wafer_nuisance.eval.plotting import (  # noqa: E402
    REFERENCE,
    SERIES,
    TEXT_SECONDARY,
    apply_style,
)
from wafer_nuisance.features.cache import pool_arrays, tune_stage1  # noqa: E402
from wafer_nuisance.learn import fit_threshold  # noqa: E402

SLOPE_RANGES = {"separable": (-1.15, -0.85), "overlapping": (-1.15, -0.45)}
FLOOR_MULTIPLE = 5.0
MIN_FIT_POINTS = 3


@dataclass
class Config:
    regimes: tuple[str, ...] = ("separable", "overlapping")
    log2_n: tuple[int, ...] = tuple(range(4, 15))
    repeats: int = 100
    n_population: int = POOLS["population"].size
    n_test: int = POOLS["test"].size
    target_recall: float = STAGE1_TARGET_RECALL
    stage1_regime: str = STAGE1_REGIME
    stage1: Stage1Config = field(default_factory=Stage1Config)
    delta: float = 0.05
    quick: bool = False


def quick_config() -> Config:
    return Config(log2_n=tuple(range(4, 11)), repeats=10, n_population=4096, n_test=4096,
                  quick=True)


def fit_slope(ns: np.ndarray, mean: np.ndarray, floor: np.ndarray) -> dict:
    ok = (mean > 0) & (mean >= FLOOR_MULTIPLE * floor)
    out = {"n_used": ns[ok].tolist(), "slope": None}
    if ok.sum() >= MIN_FIT_POINTS:
        out["slope"] = loglog_slope(ns[ok], mean[ok])
        out["intercept"] = float(np.polyfit(np.log(ns[ok]), np.log(mean[ok]), 1)[1])
    return out


def run_regime(name: str, cfg: Config, stage1) -> tuple[dict, list[dict]]:
    regime = REGIMES[name]
    ns = 2 ** np.array(cfg.log2_n)
    n_max = int(ns.max())
    pop = pool_arrays(regime, stage1, POOLS["population"], cfg.n_population, "meta")
    test = pool_arrays(regime, stage1, POOLS["test"], cfg.n_test, "meta")
    h_star = fit_threshold(pop["peak"], pop["label"])
    ev = ThresholdEvaluator(test["peak"], test["label"])
    err_star_test = float(ev.error(h_star.theta))
    err_star_pop = h_star.error(pop["peak"], pop["label"])

    train = pool_arrays(regime, stage1, POOLS["train"], cfg.repeats * n_max, "meta")
    thetas = np.empty((cfg.repeats, len(ns)))
    train_err = np.empty((cfg.repeats, len(ns)))
    for r in range(cfg.repeats):
        x = train["peak"][r * n_max : (r + 1) * n_max]
        y = train["label"][r * n_max : (r + 1) * n_max]
        for j, n in enumerate(ns):
            h = fit_threshold(x[:n], y[:n])
            thetas[r, j], train_err[r, j] = h.theta, h.error(x[:n], y[:n])
    err = ev.error(thetas)  # (repeats, len(ns))
    dis = ev.disagreement(thetas, h_star.theta)
    measured = err if name == "separable" else err - err_star_test
    mean = measured.mean(axis=0)
    floor = np.array([noise_floor(d, ev.n) for d in dis.mean(axis=0)])
    bound = realizable_bound(ns, cfg.delta) if name == "separable" \
        else agnostic_bound(ns, cfg.delta)
    fit = fit_slope(ns, mean, floor)
    lo, hi = SLOPE_RANGES[name]
    summary = {
        "measured": "test error" if name == "separable" else "excess test error",
        "bound": "realizable VC bound" if name == "separable"
                 else "agnostic VC bound",
        "n": ns.tolist(),
        "theta_star": h_star.theta,
        "L_star_population": {"value": err_star_pop,
                              "hoeffding_ci": hoeffding_interval(err_star_pop, len(pop["peak"]),
                                                                 cfg.delta)},
        "L_star_test": {"value": err_star_test,
                        "hoeffding_ci": hoeffding_interval(err_star_test, ev.n, cfg.delta)},
        "n_test": ev.n,
        "runs_with_training_error": int((train_err > 0).sum()),
        "runs_total": int(train_err.size),
        "runs_with_training_error_per_n": (train_err > 0).sum(axis=0).tolist(),
        "max_train_nuisance_peak": float(train["peak"][train["label"] < 0].max()),
        "min_train_defect_peak": float(train["peak"][train["label"] > 0].min()),
        "mean": mean.tolist(),
        "std": measured.std(axis=0, ddof=1).tolist(),
        "q10": np.quantile(measured, 0.1, axis=0).tolist(),
        "q90": np.quantile(measured, 0.9, axis=0).tolist(),
        "mean_test_error": err.mean(axis=0).tolist(),
        "mean_disagreement": dis.mean(axis=0).tolist(),
        "noise_floor": floor.tolist(),
        "bound_values": bound.tolist(),
        "fraction_runs_within_bound": (measured <= bound).mean(axis=0).tolist(),
        "fit": fit,
        "checks": {
            "slope_range": [lo, hi],
            "slope_in_range": fit["slope"] is not None and lo <= fit["slope"] <= hi,
            "bound_above_mean_every_n": bool(np.all(bound > mean)),
        },
    }
    rows = [{"regime": name, "n": int(n), "repeat": r, "theta_hat": thetas[r, j],
             "test_error": float(err[r, j]), "train_error": float(train_err[r, j]),
             "measured": float(measured[r, j]),
             "disagreement": float(dis[r, j])}
            for r in range(cfg.repeats) for j, n in enumerate(ns)]
    return summary, rows


def run(cfg: Config, tag: str) -> None:
    stage1, s1_info = tune_stage1(REGIMES[cfg.stage1_regime], cfg.stage1, cfg.target_recall)
    summary = {"stage1": s1_info, "train_protocol": "fresh dies per repeat: repeat r uses "
               "train-pool items [r * n_max, (r + 1) * n_max), nested prefixes within a repeat",
               "regimes": {}}
    rows: list[dict] = []
    for name in cfg.regimes:
        s, rr = run_regime(name, cfg, stage1)
        summary["regimes"][name] = s
        rows += rr
        c = s["checks"]
        sl = s["fit"]["slope"]
        print(f"{name}: slope {'n/a' if sl is None else f'{sl:.3f}'} over n = "
              f"{s['fit']['n_used']} (range {c['slope_range']}: {c['slope_in_range']}); "
              f"bound above mean at every n: {c['bound_above_mean_every_n']}", flush=True)
    write_results(tag, cfg, summary, rows, "cpu")
    plot(tag)


def plot(tag: str) -> None:
    apply_style()
    s = read_results(tag)["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for ax, (name, rs) in zip(axes, s["regimes"].items(), strict=True):
        ns = np.array(rs["n"], float)
        mean, floor = np.array(rs["mean"]), np.array(rs["noise_floor"])
        q10, q90 = np.array(rs["q10"]), np.array(rs["q90"])
        pos = mean > 0
        ax.plot(ns, rs["bound_values"], color=REFERENCE, ls="--", lw=1.6,
                label=f"bound: {rs['bound']}")
        ax.fill_between(ns, np.where(q10 > 0, q10, np.nan), np.where(q90 > 0, q90, np.nan),
                        color=SERIES[0], alpha=0.15, lw=0, label="10-90% of repeats")
        ax.plot(ns[pos], mean[pos], "o-", color=SERIES[0], label=f"mean {rs['measured']}")
        ax.plot(ns, FLOOR_MULTIPLE * floor, ":", color=TEXT_SECONDARY, lw=1.2,
                label=f"{FLOOR_MULTIPLE:g} × pool noise floor")
        fit = rs["fit"]
        if fit["slope"] is not None:
            nu = np.array(fit["n_used"], float)
            ax.plot(nu, np.exp(fit["intercept"]) * nu ** fit["slope"], color=SERIES[1], lw=2.2,
                    alpha=0.8, label=f"fitted slope {fit['slope']:.2f}")
        lo, hi = rs["checks"]["slope_range"]
        ax.set(xscale="log", yscale="log", xlabel="training candidates n",
               ylabel=rs["measured"],
               title=f"{name} (pass range for the slope: [{lo}, {hi}])")
        log_ticks(ax, ns[::2])
        ax.legend(fontsize=8, loc="lower left")
    out = "e1_thresholds.png" if tag == "e1_thresholds" else f"{tag}.png"
    fig.savefig(ROOT / "figures" / out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    tag = "e1_thresholds_quick" if args.quick else "e1_thresholds"
    if args.plot_only:
        plot(tag)
        return
    run(quick_config() if args.quick else Config(), tag)


if __name__ == "__main__":
    main()
