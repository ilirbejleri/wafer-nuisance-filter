"""Headline figure for the README: one panel per result, drawn from results/*.json only.

    uv run python experiments/headline.py      # writes figures/headline.png

Panels: Result 1 (threshold rate), Result 2 (bounds for the linear filter on the hand-crafted
features), Result 3 (asymptotic learning curve), Result 4 (certified die-level false-alarm rate).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
from _common import ROOT, log_ticks, read_results

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wafer_nuisance.eval.plotting import (  # noqa: E402
    REFERENCE,
    SERIES,
    TEXT_SECONDARY,
    apply_style,
)


def panel_b(ax) -> None:
    rs = read_results("e1_thresholds")["summary"]["regimes"]["overlapping"]
    ns, mean = np.array(rs["n"], float), np.array(rs["mean"])
    ax.plot(ns, rs["bound_values"], "--", color=REFERENCE, lw=1.6, label="agnostic VC bound")
    ax.plot(ns[mean > 0], mean[mean > 0], "o-", color=SERIES[0], label="mean excess error")
    fit = rs["fit"]
    nu = np.array(fit["n_used"], float)
    ax.plot(nu, np.exp(fit["intercept"]) * nu ** fit["slope"], color=SERIES[1], lw=2.2,
            alpha=0.8, label=f"fitted slope {fit['slope']:.2f}")
    ax.set(xscale="log", yscale="log", xlabel="training candidates n", ylabel="excess 0-1 error",
           title="Result 1: threshold on peak contrast")
    log_ticks(ax, ns[::2])
    ax.legend(fontsize=8, loc="lower left")


def panel_c(ax) -> None:
    v = read_results("e2_linear_bounds")["summary"]["sets"]["hand"]
    ns = np.array(v["n"], float)
    ax.plot(ns, v["mean_vc_bound"], "--", color=SERIES[1], label="VC bound")
    ax.plot(ns, v["mean_rad_bound"], "--", color=SERIES[2],
            label="Rademacher, clipped (validated B)")
    ax.plot(ns, v["mean_rad_cert_bound"], ":", color=SERIES[2], lw=1.6,
            label="Rademacher, clipped (B minimizing it)")
    ax.plot(ns, v["mean_test_error"], "o-", color=SERIES[0], label="test error")
    pi = read_results("stage1")["summary"]["regimes"]["overlapping"]["positive_rate"]["value"]
    ax.axhline(min(pi, 1 - pi), color=TEXT_SECONDARY, lw=1.0, ls="-.",
               label="trivial classifier min(π, 1 − π)")
    ax.set(xscale="log", yscale="log", xlabel="training candidates n", ylabel="0-1 error / bound",
           title="Result 2: linear filter, hand-crafted features")
    log_ticks(ax, ns[::2])
    ax.legend(fontsize=8, loc="lower left")


def panel_d(ax) -> None:
    sets = read_results("e3_learning_curve")["summary"]["sets"]
    ax.axhspan(0.9, 1.1, color=REFERENCE, alpha=0.12, lw=0, label="±10% (pass band)")
    ax.axhline(1.0, color=REFERENCE, ls="--", lw=1.4)
    for j, (name, v) in enumerate(sets.items()):
        q = v["per_n"]
        r = np.array([x["n_over_p"] for x in q], float)
        ratio = np.array([x["mean_n_excess"] for x in q]) / v["predicted_n_excess"]
        se = np.array([x["se_n_excess"] for x in q]) / v["predicted_n_excess"]
        ax.errorbar(r, ratio, yerr=1.96 * se, fmt="o-", color=SERIES[j], capsize=3,
                    label=f"{name} (p = {v['p']})")
    ax.set(xscale="log", xlabel="n / p", ylabel="measured / predicted  n·excess",
           title="Result 3: excess logistic loss vs Tr(H⁻¹G)/2n")
    log_ticks(ax, r, y_log=False)
    ax.legend(fontsize=8)


def panel_a(ax) -> None:
    r = read_results("e4_false_alarms")
    s, alpha = r["summary"], r["config"]["alpha"]
    rules = (("Hoeffding", s["per_m"], SERIES[0]),
             ("Chernoff", [x["chernoff"] | {"m": x["m"]} for x in s["per_m"]], SERIES[3]))
    for name, qs, color in rules:
        q = [x for x in qs if x["mean_certified"] is not None]
        ms = np.array([x["m"] for x in q], float)
        ax.plot(ms, [x["mean_certified"] for x in q], "o--", color=color, alpha=0.6,
                label=f"{name}: certified bound")
        ax.plot(ms, [x["mean_true_far"] for x in q], "o-", color=color,
                label=f"{name}: true FAR of selected threshold")
    ax.axhline(alpha, color=REFERENCE, ls=":", lw=1.4, label=f"α = {alpha}")
    ax.set(xscale="log", xlabel="certification dies m", ylabel="die-level false-alarm rate",
           title="Result 4: certified false-alarm rate", ylim=(0, 1.5 * alpha))
    log_ticks(ax, [x["m"] for x in s["per_m"]], y_log=False)
    ax.legend(fontsize=7, loc="upper right")


PANELS: tuple[Callable, ...] = (panel_b, panel_c, panel_d, panel_a)


def main() -> Path | None:
    apply_style()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(5.2 * len(PANELS), 4.6), squeeze=False,
                             layout="constrained")
    for panel, ax in zip(PANELS, axes[0], strict=True):
        panel(ax)
    out = ROOT / "figures" / "headline.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)
    return out


if __name__ == "__main__":
    main()
