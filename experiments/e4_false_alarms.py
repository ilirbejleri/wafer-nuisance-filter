"""E4 (Result 4): certifying the die-level false-alarm rate with Hoeffding + a union bound.

    uv run python experiments/e4_false_alarms.py              # full run
    uv run python experiments/e4_false_alarms.py --quick      # small pools, *_quick outputs
    uv run python experiments/e4_false_alarms.py --plot-only  # redraw the figure from results/

Outputs: results/e4_false_alarms.json, results/e4_false_alarms_raw.csv (one row per m, repeat),
figures/e4_false_alarms.png.

Protocol (overlapping regime). Everything below is fixed before any certification die is drawn.

* Scorer: norm-constrained logistic regression on the hand-crafted features (||x|| <= 1), trained
  once on train-pool items 0, ..., 4095, with B chosen from the E2 grid by validation 0-1 error;
  then frozen. A candidate's score is w.x; a die is flagged at threshold t iff some candidate's
  score exceeds t (a die without candidates is never flagged).
* Score grid (T = 50): the quantiles of the validation pool's nuisance-candidate scores at levels
  0.9 + 0.1 (j - 0.5) / T, j = 1..T, i.e. 0.901, ..., 0.999 (a defect-free die has ~2.7
  candidates, so this spans die-level false-alarm rates from roughly 25% down to a few 0.1%).
* Repeat r (200 repeats) uses certification dies r * 8000, ..., (r + 1) * 8000 - 1 (fresh
  defect-free dies for every repeat); size m uses the first m. p_hat_m(t_j) = fraction flagged;
  the selected threshold is the lowest t_j with p_hat_m(t_j) + sqrt(ln(T/delta)/(2m)) <= alpha
  (``bounds.result_4``), or none.
* True FAR p(t_j): flag frequency on the far-test pool (100,000 defect-free dies, disjoint), with
  its Hoeffding interval. A repeat violates if the true FAR of its selected threshold exceeds
  alpha (a repeat with no certifiable threshold cannot violate).
* For the detection side, the flag rate on defective dies (stage1_check pool, 20,000 dies
  with one defect each) at every grid threshold.

Check (fixed before running): violation fraction <= delta; report the smallest m giving a
certifiable threshold. The write-up (theory_tables.py) reports a 1 - delta upper bound on the
violation rate from the v violations in the R repeats (``bounds.violation_rate_bound``:
ln(1/delta)/R if v = 0, v/R + sqrt(ln(1/delta)/(2R)) otherwise); conditional on the scorer, the
grid and the far-test pool, the repeats' violation indicators are i.i.d.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import matplotlib
import numpy as np
from _common import ROOT, log_ticks, read_results, write_results

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wafer_nuisance.bounds import (  # noqa: E402
    certification_margin,
    hoeffding_halfwidth,
    select_threshold,
)
from wafer_nuisance.data.config import (  # noqa: E402
    POOLS,
    REGIMES,
    STAGE1_REGIME,
    STAGE1_TARGET_RECALL,
)
from wafer_nuisance.detect.stage1 import Stage1Config  # noqa: E402
from wafer_nuisance.eval.plotting import REFERENCE, SERIES, TEXT_SECONDARY, apply_style  # noqa: E402
from wafer_nuisance.features.cache import (  # noqa: E402
    feature_maps,
    pool_arrays,
    pool_features,
    tune_stage1,
)
from wafer_nuisance.learn import fit_logistic_ball, zero_one_error  # noqa: E402


@dataclass
class Config:
    regime: str = "overlapping"
    feature_set: str = "hand"
    n_train: int = 4096
    B_grid: tuple[float, ...] = tuple(2.0**k for k in range(-1, 9))
    tol: float = 1e-8
    T: int = 50
    grid_levels: tuple[float, float] = (0.9, 1.0)
    m_values: tuple[int, ...] = (500, 1000, 2000, 4000, 8000)
    repeats: int = 200
    alpha: float = 0.05
    delta: float = 0.05
    n_fit: int = POOLS["feature_fit"].size
    n_validation: int = POOLS["validation"].size
    n_far: int = POOLS["far_test"].size
    n_defective: int = POOLS["stage1_check"].size
    target_recall: float = STAGE1_TARGET_RECALL
    stage1_regime: str = STAGE1_REGIME
    stage1: Stage1Config = field(default_factory=Stage1Config)
    quick: bool = False


def quick_config() -> Config:
    return Config(n_train=1024, m_values=(500, 1000, 2000), repeats=10, n_fit=2048,
                  n_validation=2048, n_far=4096, n_defective=1024, quick=True)


def train_scorer(cfg: Config, regime, stage1) -> tuple:
    maps, _ = feature_maps(regime, stage1, cfg.n_fit)
    fm = maps[cfg.feature_set]
    tr = pool_features(regime, stage1, POOLS["train"], cfg.n_train)
    val = pool_features(regime, stage1, POOLS["validation"], cfg.n_validation)
    X, y = fm(tr["hand"]), tr["label"].astype(float)
    Xv, yv = fm(val["hand"]), val["label"].astype(float)
    fits, w0 = [], None
    for B in cfg.B_grid:
        r = fit_logistic_ball(X, y, B, tol=cfg.tol, w0=w0)
        w0 = r.w
        fits.append((zero_one_error(r.w, Xv, yv), B, r))
    j = min(range(len(fits)), key=lambda i: (fits[i][0], i))
    val_err, B, res = fits[j]
    scores_nuis = Xv[yv < 0] @ res.w
    T = cfg.T
    lo, hi = cfg.grid_levels
    levels = lo + (hi - lo) * (np.arange(1, T + 1) - 0.5) / T
    grid = np.quantile(scores_nuis, levels)
    info = {"B": B, "validation_error": val_err, "converged": res.converged,
            "grad_map": res.grad_norm, "w": res.w.tolist(), "grid_levels": levels.tolist(),
            "grid": grid.tolist(), "n_validation_nuisance": int((yv < 0).sum())}
    return (fm, res.w), grid, info


def run(cfg: Config, tag: str) -> None:
    stage1, s1_info = tune_stage1(REGIMES[cfg.stage1_regime], cfg.stage1, cfg.target_recall)
    regime = REGIMES[cfg.regime]
    scorer, grid, sc_info = train_scorer(cfg, regime, stage1)
    print(f"scorer: B = {sc_info['B']}, validation error {sc_info['validation_error']:.4f}; "
          f"grid {grid[0]:.3f} .. {grid[-1]:.3f}", flush=True)

    far = pool_arrays(regime, stage1, POOLS["far_test"], cfg.n_far, "die_score",
                      payload=scorer)
    p_true = (far["max_score"][:, None] > grid[None, :]).mean(axis=0)
    hw_far = hoeffding_halfwidth(cfg.n_far, cfg.delta)
    dfc = pool_arrays(regime, stage1, POOLS["stage1_check"], cfg.n_defective, "die_score",
                      payload=scorer)
    flag_defective = (dfc["max_score"][:, None] > grid[None, :]).mean(axis=0)

    m_max = max(cfg.m_values)
    cert = pool_arrays(regime, stage1, POOLS["certification"], cfg.repeats * m_max,
                       "die_score", payload=scorer)
    top = cert["max_score"].reshape(cfg.repeats, m_max)
    rows, per_m = [], []
    for m in cfg.m_values:
        flags = top[:, :m, None] > grid[None, None, :]  # (repeats, m, T)
        p_hat = flags.mean(axis=1)
        margin = certification_margin(m, cfg.T, cfg.delta)
        sel = [select_threshold(grid, p_hat[r], m, cfg.alpha, cfg.delta)
               for r in range(cfg.repeats)]
        viol = [j is not None and p_true[j] > cfg.alpha for j in sel]
        for r, j in enumerate(sel):
            rows.append({"m": m, "repeat": r, "selected": -1 if j is None else j,
                         "threshold": float("nan") if j is None else float(grid[j]),
                         "certified": float("nan") if j is None else float(p_hat[r, j] + margin),
                         "p_hat": float("nan") if j is None else float(p_hat[r, j]),
                         "true_far": float("nan") if j is None else float(p_true[j]),
                         "violation": bool(viol[r])})
        got = [j for j in sel if j is not None]
        v = float(np.mean(viol))
        per_m.append({
            "m": m, "margin": margin, "fraction_certifiable": len(got) / cfg.repeats,
            "violation_fraction": v, "violations": int(np.sum(viol)),
            "mean_certified": float(np.mean([p_hat[r, j] + margin for r, j in enumerate(sel)
                                             if j is not None])) if got else None,
            "mean_true_far": float(np.mean(p_true[got])) if got else None,
            "q10_true_far": float(np.quantile(p_true[got], 0.1)) if got else None,
            "q90_true_far": float(np.quantile(p_true[got], 0.9)) if got else None,
            "mean_flag_rate_defective": float(np.mean(flag_defective[got])) if got else None,
            "selected_index_counts": np.bincount(got, minlength=cfg.T).tolist() if got else [],
        })
        print(f"m = {m}: certifiable in {len(got)}/{cfg.repeats}, violation fraction {v:.3f}, "
              f"mean true FAR {per_m[-1]['mean_true_far']}", flush=True)

    certifiable = [q["m"] for q in per_m if q["fraction_certifiable"] > 0]
    always = [q["m"] for q in per_m if q["fraction_certifiable"] == 1.0]
    summary = {
        "stage1": s1_info, "regime": cfg.regime, "scorer": sc_info,
        "true_far": p_true.tolist(), "true_far_hoeffding_halfwidth": hw_far,
        "flag_rate_defective": flag_defective.tolist(),
        "far_test_zero_candidate_fraction": float(np.mean(np.isneginf(far["max_score"]))),
        "far_test_mean_candidates": float(far["n_candidates"].mean()),
        "per_m": per_m,
        "smallest_m_certifiable_any_repeat": certifiable[0] if certifiable else None,
        "smallest_m_certifiable_every_repeat": always[0] if always else None,
        "certification_protocol": "fresh dies per repeat: repeat r uses certification dies "
                                  f"[r * {m_max}, (r + 1) * {m_max}), nested prefixes",
        "checks": {"violation_fraction_le_delta_every_m":
                   bool(all(q["violation_fraction"] <= cfg.delta for q in per_m))},
    }
    write_results(tag, cfg, summary, rows, "cpu")
    plot(tag)


def plot(tag: str) -> None:
    apply_style()
    r = read_results(tag)
    s, cfg = r["summary"], r["config"]
    alpha = cfg["alpha"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    q = [x for x in s["per_m"] if x["mean_certified"] is not None]
    ms = np.array([x["m"] for x in q], float)
    a1.plot(ms, [x["mean_certified"] for x in q], "o--", color=SERIES[1],
            label="certified bound (mean over repeats)")
    a1.fill_between(ms, [x["q10_true_far"] for x in q], [x["q90_true_far"] for x in q],
                    color=SERIES[0], alpha=0.15, lw=0)
    a1.plot(ms, [x["mean_true_far"] for x in q], "o-", color=SERIES[0],
            label="true FAR of the selected threshold (mean, 10-90%)")
    a1.axhline(alpha, color=REFERENCE, ls=":", lw=1.4, label=f"α = {alpha}")
    for x in s["per_m"]:
        a1.annotate(f"cert. {x['fraction_certifiable']:.0%}\nviol. {x['violation_fraction']:.1%}",
                    (x["m"], 0), textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=7, color=TEXT_SECONDARY)
    a1.set(xscale="log", xlabel="certification dies m", ylabel="die-level false-alarm rate",
           title="Result 4: certified vs true false-alarm rate", ylim=(0, 1.5 * alpha))
    log_ticks(a1, [x["m"] for x in s["per_m"]], y_log=False)
    a1.legend(fontsize=8, loc="upper right")

    far, det = np.array(s["true_far"]), np.array(s["flag_rate_defective"])
    a2.plot(far, det, "o-", ms=3, color=SERIES[2])
    a2.axvline(alpha, color=REFERENCE, ls=":", lw=1.4, label=f"α = {alpha}")
    a2.set(xscale="log", xlabel="true die-level FAR (far-test pool)",
           ylabel="flag rate on defective dies",
           title="die-level trade-off over the 50 grid thresholds")
    a2.legend(fontsize=8)
    out = "e4_false_alarms.png" if tag == "e4_false_alarms" else f"{tag}.png"
    fig.savefig(ROOT / "figures" / out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    tag = "e4_false_alarms_quick" if args.quick else "e4_false_alarms"
    if args.plot_only:
        plot(tag)
        return
    run(quick_config() if args.quick else Config(), tag)


if __name__ == "__main__":
    main()
