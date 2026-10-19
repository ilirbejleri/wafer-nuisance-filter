"""Numbers for the LaTeX write-up (theory/), generated from results/*.json.

    uv run python experiments/theory_tables.py

Writes theory/generated/numbers.tex (one \\newcommand per number) and
results/theory_constants.json (the same values plus their source). The LaTeX never contains a
hand-typed experimental number: every measured value is a macro defined here and read from a
results file; theoretical values are evaluated from the formulas in wafer_nuisance.bounds with
the configuration recorded in those files.
"""

from __future__ import annotations

import csv
import json
import math

from _common import ROOT

from wafer_nuisance.bounds import vc_bound, violation_rate_bound


def fmt(x, nd: int = 3) -> str:
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and math.isinf(x)):
        return r"\infty"
    if isinstance(x, int) or (isinstance(x, float) and x.is_integer()):
        return f"{int(x):,}".replace(",", "{,}")
    return f"{x:.{nd}f}"


def collect() -> dict[str, tuple[str, str]]:
    """macro name -> (formatted value, source file)."""
    out: dict[str, tuple[str, str]] = {}

    def put(name: str, value, src: str, nd: int = 3) -> None:
        assert name.isalpha(), name
        out[name] = (fmt(value, nd), src)

    def load(tag: str) -> dict | None:
        path = ROOT / "results" / f"{tag}.json"
        return json.loads(path.read_text())["summary"] if path.exists() else None

    def load_full(tag: str) -> dict:
        return json.loads((ROOT / "results" / f"{tag}.json").read_text())

    reg = {"separable": "Sep", "overlapping": "Ovl"}

    if s := load("stage1"):
        src = "stage1.json"
        put("StageOneTau", s["stage1"]["tau"], src, 4)
        put("StageOneTuningRecall", s["stage1"]["tuning_recall"], src, 3)
        put("StageOneNTune", s["stage1"]["n_tune_dies"], src)
        put("StageOneNReference", s["stage1"]["n_reference_dies"], src)
        put("StageOneNRecall", s["regimes"]["overlapping"]["recall"]["overall"]["n"], src)
        for name, p in reg.items():
            r = s["regimes"][name]
            put(f"{p}PositiveRate", r["positive_rate"]["value"], src)
            put(f"{p}CandPerDie", r["candidates_per_die"]["mean"], src, 2)
            put(f"{p}ZeroCandRate", r["redraws"]["zero_candidate_die_rate"]["value"], src)
            put(f"{p}Recall", r["recall"]["overall"]["value"], src, 4)
            put(f"{p}BestErrors", r["threshold_check"]["best_errors"], src)
            put(f"{p}BestErrorRate", r["threshold_check"]["best_error_rate"]["value"], src, 4)
            put(f"{p}NPopulation", r["n_candidates_sampled"], src)
        # Error of the trivial classifier (always predict the majority label) under Q on the
        # main regime's population pool: a 0-1 bound says something only below this level.
        pi = s["regimes"]["overlapping"]["positive_rate"]["value"]
        put("TrivialError", min(pi, 1 - pi), src)

    if s := load("features"):
        src = "features.json"
        names = {"hand": "Hand", "pca4": "PcaFour", "pca16": "PcaSixteen",
                 "pca64": "PcaSixtyFour", "cnn384": "CnnFull"}
        for k, p in names.items():
            f = s["sets"][k]
            put(f"{p}C", f["C"], src, 2)
            put(f"{p}LinValError", f["linear_validation_error"]["value"], src)
            put(f"{p}MeanSqNorm", f["validation_mean_sq_norm"], src, 2)
        put("PeakValError", s["threshold_peak_validation_error"]["value"], src)

    if s := load("e1_thresholds"):
        src = "e1_thresholds.json"
        for name, p in reg.items():
            r = s["regimes"][name]
            slope = r["fit"]["slope"]
            put(f"EOne{p}Slope", "\\text{n/a}" if slope is None else slope, src, 2)
            put(f"EOne{p}SlopePass", "yes" if r["checks"]["slope_in_range"] else "no", src)
            used = r["fit"]["n_used"]  # the n the slope was fitted on (above 5 x noise floor)
            put(f"EOne{p}FitNMin", min(used) if used else "\\text{--}", src)
            put(f"EOne{p}FitNMax", max(used) if used else "\\text{--}", src)
            put(f"EOne{p}FitPoints", len(used), src)
            put(f"EOne{p}BoundPass", "yes" if r["checks"]["bound_above_mean_every_n"] else "no",
                src)
            put(f"EOne{p}LStarTest", r["L_star_test"]["value"], src, 4)
            put(f"EOne{p}BoundRatioLast",
                r["bound_values"][-1] / r["mean"][-1] if r["mean"][-1] > 0 else None, src, 0)
        # Realizability check for Cor 5 (separable regime): theta* was chosen on the population
        # pool, so its errors on the independent test pool bound L(h*); and the realizable bound
        # applies to the runs whose ERM is consistent with its training sample.
        sep = s["regimes"]["separable"]
        k_star = round(sep["L_star_test"]["value"] * sep["n_test"])
        put("EOneSepNTest", sep["n_test"], src)
        put("EOneSepTestErrors", k_star, src)
        put("EOneSepLStarBound", violation_rate_bound(k_star, sep["n_test"], 0.05), src, 5)
        put("EOneSepInconsistentRuns", sep["runs_with_training_error"], src)
        put("EOneSepRunsTotal", sep["runs_total"], src)
        put("EOneSepMaxTrainNuisancePeak", sep["max_train_nuisance_peak"], src)
        put("EOneNLast", s["regimes"]["overlapping"]["n"][-1], src)
        # How sharply the nuisance peak-contrast density ends at theta* (overlapping regime,
        # population-pool histogram in stage1.json): P(defect | bin) one bin below and one bin
        # above the bin that contains theta*.
        if st := load("stage1"):
            theta = s["regimes"]["overlapping"]["theta_star"]
            h = st["regimes"]["overlapping"]["peak_hist"]
            edges, nu, de = h["bins"], h["nuisance"], h["defect"]
            j = next(i for i in range(len(nu)) if edges[i] <= theta < edges[i + 1])
            put("EOneOvlThetaStar", theta, src)
            put("OvlBinWidth", edges[j + 1] - edges[j], "stage1.json", 2)
            for name, i in (("Below", j - 1), ("Above", j + 1)):
                put(f"OvlBin{name}Lo", edges[i], "stage1.json", 2)
                put(f"OvlBin{name}Hi", edges[i + 1], "stage1.json", 2)
                put(f"OvlNuis{name}", nu[i], "stage1.json")
                put(f"OvlCond{name}", de[i] / (de[i] + nu[i]), "stage1.json", 2)
        put("EOneMinFracWithinBound", pct(min(min(r["fraction_runs_within_bound"])
                                            for r in s["regimes"].values())), src)
        put("EOneRepeats", load_full("e1_thresholds")["config"]["repeats"], src)
        if st := load("stage1"):
            tc = st["regimes"]["separable"]["threshold_check"]
            put("SepMaxNuisancePeak", tc["max_nuisance_peak"], "stage1.json")
            put("SepMinDefectPeak", tc["min_defect_peak"], "stage1.json")
        raw = ROOT / "results" / "e1_thresholds_raw.csv"
        if raw.exists():
            zero: dict[int, list[bool]] = {}
            with open(raw) as f:
                for row in csv.DictReader(f):
                    if row["regime"] == "separable":
                        zero.setdefault(int(row["n"]), []).append(float(row["test_error"]) == 0)
            for n, name in ((16, "Sixteen"), (1024, "OneK"), (4096, "FourK")):
                put(f"EOneSepZeroFrac{name}", pct(sum(zero[n]) / len(zero[n])),
                    "e1_thresholds_raw.csv")

    sets = {"hand": "Hand", "pca4": "PcaFour", "pca16": "PcaSixteen", "pca64": "PcaSixtyFour",
            "cnn384": "CnnFull"}
    set_label = {"hand": "hand-crafted", "pca4": "PCA-4", "pca16": "PCA-16", "pca64": "PCA-64",
                 "cnn384": "CNN-384"}
    if s := load("e2_linear_bounds"):
        src = "e2_linear_bounds.json"
        pi = load("stage1")["regimes"]["overlapping"]["positive_rate"]["value"]
        trivial = min(pi, 1 - pi)

        def first_below(ns, curve):  # first n whose mean bound beats the trivial classifier
            n = next((n for n, b in zip(ns, curve, strict=True) if b < trivial), None)
            return n or "\\text{none in range}"

        for k, v in s["sets"].items():
            p = "ETwo" + sets[k]
            put(f"{p}FracVC", pct(v["fraction_vc_above"]), src)
            put(f"{p}FracRad", pct(v["fraction_rad_above"]), src)
            put(f"{p}Pass", "yes" if all(v["checks"].values()) else "no", src)
            put(f"{p}FirstNVC", first_below(v["n"], v["mean_vc_bound"]), src)
            put(f"{p}FirstNRad", first_below(v["n"], v["mean_rad_bound"]), src)
            put(f"{p}TestLast", v["mean_test_error"][-1], src)
            put(f"{p}VCLast", v["mean_vc_bound"][-1], src, 2)
            put(f"{p}RadLast", v["mean_rad_bound"][-1], src, 2)
            put(f"{p}RadCertLast", v["mean_rad_cert_bound"][-1], src, 2)
            put(f"{p}RadCertTestLast", v["mean_rad_cert_test_error"][-1], src)
            put(f"{p}LStar", v["L_star"]["L_star_test"]["value"], src)
            put(f"{p}EstLast", v["estimation_error"][-1], src, 4)
            put(f"{p}EstFirst", v["estimation_error"][0], src)
            counts = {float(b): c[-1] for b, c in v["B_selected_counts"].items()}
            put(f"{p}BModeLast", max(counts, key=counts.get), src)
        put("ETwoNLast", s["sets"]["hand"]["n"][-1], src)
        put("ETwoNFirst", s["sets"]["hand"]["n"][0], src)
        put("ETwoRepeats", load_full("e2_linear_bounds")["config"]["repeats"], src)
        put("ETwoMinFracBoth", pct(min(min(v["fraction_vc_above"], v["fraction_rad_above"])
                                     for v in s["sets"].values())), src)
        # Why the Rademacher bound stays above the trivial error (theory/looseness.tex). With
        # C = 1 and G grid values, a run's (clipped-loss) bound at its validated B has three
        # terms: the training psi loss / ln 2; the complexity term 2 (B/n) sqrt(sum ||x||^2) / ln 2
        # = 2 B r / (sqrt(n) ln 2), r^2 = mean ||x||^2 on its training sample; and the deviation
        # term 3 sqrt(ln(2G/delta) / (2n)), which does not depend on B. The unclipped bound has the
        # same complexity term (contraction), the training logistic loss / ln 2 and the deviation
        # term 3 B sqrt(ln(2G/delta) / (2n)) / ln 2. Every column of the table is the mean over the
        # runs at the largest n of that run's term, each at the run's own validated B, so the
        # columns add up to the mean bounds; B and B r are averaged over runs the same way.
        ec = load_full("e2_linear_bounds")
        delta, G = ec["config"]["delta"], len(ec["config"]["B_grid"])
        n_last = s["sets"]["hand"]["n"][-1]
        dev = math.sqrt(math.log(2 * G / delta) / 2)  # clipped deviation term = 3 dev / sqrt(n)
        dev_term = 3 * dev / math.sqrt(n_last)
        put("ETwoDevTermLast", dev_term, src, 3)
        ln2 = math.log(2)
        kdim = {"hand": 10, "pca4": 4, "pca16": 16, "pca64": 64, "cnn384": 384}
        with open(ROOT / "results" / "e2_linear_bounds_raw.csv") as f:
            last_rows = [r for r in csv.DictReader(f) if int(r["n"]) == n_last]
        spread = []
        for k, v in s["sets"].items():
            p = "ETwo" + sets[k]
            ls = v["L_star"]
            R = [r for r in last_rows if r["set"] == k]
            Bs = [float(r["B"]) for r in R]
            Br = [float(r["B"]) * math.sqrt(float(r["mean_sq_norm"])) for r in R]
            comp = [2 * b / math.sqrt(n_last) / ln2 for b in Br]
            psi = [float(r["train_psi"]) / ln2 for r in R]
            log = [float(r["train_log"]) / ln2 for r in R]
            dev_u = [3 * b * dev / math.sqrt(n_last) / ln2 for b in Bs]
            for r, c, a, b, d in zip(R, comp, psi, log, dev_u, strict=True):
                assert math.isclose(a + c + dev_term, float(r["rad_bound"]), rel_tol=1e-9)
                assert math.isclose(b + c + d, float(r["rad_bound_unclipped"]), rel_tol=1e-9)
            mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
            counts = {b: Bs.count(b) for b in sorted(set(Bs))}
            if len(counts) > 1:
                spread.append(f"{set_label[k]}: " + ", ".join(
                    f"$B={b:g}$ in {c} runs" for b, c in counts.items()))
            r2 = mean([float(r["mean_sq_norm"]) for r in R])
            psi_floor = ls["population_psi_loss"] / ln2
            put(f"{p}WStar", ls["norm_w"], src, 0)
            put(f"{p}LogFloor", ls["population_log_loss"] / ln2, src, 2)
            put(f"{p}PsiFloor", psi_floor, src, 2)
            put(f"{p}BMeanLast", mean(Bs), src, 1)
            put(f"{p}RSq", r2, src, 2)
            put(f"{p}TrainPsiTermLast", mean(psi), src, 2)
            put(f"{p}CompTermLast", mean(comp), src, 2)
            put(f"{p}RadUnclippedLast", mean([float(r["rad_bound_unclipped"]) for r in R]),
                src, 2)
            put(f"{p}RadUnclippedCertLast", v["mean_rad_unclipped_cert_bound"][-1], src, 2)
            put(f"{p}TrainLogTermLast", mean(log), src, 2)
            put(f"{p}DevTermUnclippedLast", mean(dev_u), src, 2)
            # Up to constants the Rademacher complexity term is B r / sqrt(n) and the VC term
            # sqrt((k+1) ln n / n): compare B r with sqrt((k+1) ln n) at the largest n.
            put(f"{p}Br", mean(Br), src, 1)
            put(f"{p}VCRoot", math.sqrt((kdim[k] + 1) * math.log(n_last)), src, 1)
            # n at which psi floor + (2 B r / ln 2 + 3 dev) / sqrt(n) = min(pi, 1 - pi): what the
            # bound with the runs' mean B r needs to beat the trivial classifier if the training
            # psi loss were already at the population optimum's; never if the floor is above it.
            need = ((2 * mean(Br) / ln2 + 3 * dev) / (trivial - psi_floor)) ** 2
            put(f"{p}RadNeedN", float(f"{need:.2g}") if psi_floor < trivial else "\\text{never}",
                src)
        put("ETwoBSpread", "every run except for " + "; ".join(spread) if spread else "every run",
            "e2_linear_bounds_raw.csv")
        # B chosen by the filter that minimizes its own (clipped) bound, most common at n_last
        cert_B = [max(((float(b), c[-1]) for b, c in v["rad_cert_B_counts"].items()),
                      key=lambda t: t[1])[0] for v in s["sets"].values()]
        put("ETwoCertBMin", min(cert_B), src)
        put("ETwoCertBMax", max(cert_B), src)
        rad_last = [v["mean_rad_bound"][-1] for v in s["sets"].values()]
        put("ETwoRadLastMin", min(rad_last), src, 2)
        put("ETwoRadLastMax", max(rad_last), src, 2)
        ci = s["sets"]["hand"]["L_star"]["L_star_test"]["hoeffding_ci"]
        put("ETwoLStarHalfwidth", (ci[1] - ci[0]) / 2, src, 4)
        put("ETwoHandVCRatioLast", s["sets"]["hand"]["mean_vc_bound"][-1]
            / s["sets"]["hand"]["mean_test_error"][-1], src, 1)
        lstar = s["sets"]["cnn384"]["L_star"]["L_star_test"]["value"]
        lo, hi = 385, 2**40  # smallest n with VC bound <= min(pi, 1 - pi) at error = L(h*)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if vc_bound(lstar, mid, 384, delta) > trivial else (lo, mid)
        put("ETwoCnnFullVCNeedN", float(f"{hi:.2g}"), src)
        for pair, n in s["crossover_n"].items():
            a, b = pair.split("_vs_")
            put(f"ETwoCross{sets[a]}{sets[b]}", n or "\\text{none}", src)

    if s := load("e3_learning_curve"):
        src = "e3_learning_curve.json"
        for k, v in s["sets"].items():
            p = "EThree" + sets[k]
            last = v["per_n"][-1]
            put(f"{p}P", v["p"], src)
            put(f"{p}Pred", v["predicted_n_excess"], src, 2)
            put(f"{p}PHalf", v["well_specified_n_excess"], src, 1)
            put(f"{p}Miss", v["misspecification_ratio"], src)
            put(f"{p}HalfA", v["trace_HinvG_test_halves"][0] / 2, src, 2)
            put(f"{p}HalfB", v["trace_HinvG_test_halves"][1] / 2, src, 2)
            put(f"{p}MeasLast", last["mean_n_excess"], src, 2)
            put(f"{p}SeLast", last["se_n_excess"], src, 2)
            put(f"{p}FloorLast", last["mean_n_floor"], src, 2)
            put(f"{p}RatioLast", last["ratio"], src)
            put(f"{p}Pass", "yes" if v["checks"]["pass"] else "no", src)
            put(f"{p}FromNP", v["checks"]["smallest_n_over_p_in_range_from_then_on"]
                or "\\text{none}", src)
            put(f"{p}Coverage", last["wald_coverage"], src)
            put(f"{p}CoverageFirst", v["per_n"][0]["wald_coverage"], src)
            put(f"{p}CovTraceLast", last["n_cov_trace_ratio"], src)
            put(f"{p}Separable", sum(q["separable_runs"] for q in v["per_n"]), src)
            put(f"{p}RatioFirst", v["per_n"][0]["ratio"], src)

        ed = load_full("e3_learning_curve")["config"]
        n_max = max(v["per_n"][-1]["n"] for v in s["sets"].values())
        put("EThreeNPop", ed["n_population"], src)
        put("EThreeNMax", n_max, src)
        put("EThreePopBias", pct(n_max / ed["n_population"]), src)
        put("EThreeRepeats", ed["repeats"], src)

    if s := load("e3b_plugin_curve"):
        src = "e3b_plugin_curve.json"
        eb3 = load_full("e3b_plugin_curve")["config"]
        put("EThreeBRepeats", eb3["repeats"], src)
        put("EThreeBFactor", eb3["factor"], src)
        put("EThreeBNList", ", ".join(str(m) for m in eb3["n_over_p"]), src)
        put("EThreeBNMaxOverP", max(eb3["n_over_p"]), src)
        put("EThreeBNMinOverP", min(eb3["n_over_p"]), src)
        rng = next(iter(s["sets"].values()))["checks"]["ratio_range"]
        put("EThreeBRangeLo", rng[0], src, 2)
        put("EThreeBRangeHi", rng[1], src, 2)
        n_top = max(v["per_n"][-1]["n"] for v in s["sets"].values())
        put("PoolTrainEThreeB", eb3["repeats"] * eb3["factor"] * n_top, src)
        per = [q for v in s["sets"].values() for q in v["per_n"]]
        firsts = [v["per_n"][0]["ratio_plugin"] for v in s["sets"].values()]
        put("EThreeBRatioFirstMin", min(firsts), src, 2)
        put("EThreeBRatioFirstMax", max(firsts), src, 2)
        # Largest gap between the plug-in and population-H,G ratios over every (set, n).
        put("EThreeBMaxRatioGap", max(abs(q["ratio_plugin"] - q["ratio_population"])
                                      for q in per), src, 2)
        lastq = [v["per_n"][-1] for v in s["sets"].values()]
        sd = [q["sd_plugin_prediction"] / q["mean_improvement"] for q in lastq]
        put("EThreeBPredSdMin", min(sd), src, 2)
        put("EThreeBPredSdMax", max(sd), src, 2)
        for k, v in s["sets"].items():
            p = "EThreeB" + sets[k]
            first, last = v["per_n"][0], v["per_n"][-1]
            n = last["n"]
            # n x improvement, on the scale of Tr(H^-1 G) (1 - 1/factor) / 2.
            put(f"{p}MeasLast", n * last["mean_improvement"], src, 2)
            put(f"{p}SeLast", n * last["se_improvement"], src, 2)
            put(f"{p}PluginLast", n * last["mean_plugin_prediction"], src, 2)
            put(f"{p}PopLast", n * last["population_prediction"], src, 2)
            put(f"{p}PLast", n * last["well_specified_prediction"], src, 2)
            put(f"{p}RatioLast", last["ratio_plugin"], src, 2)
            put(f"{p}RatioFirst", first["ratio_plugin"], src, 2)
            put(f"{p}RatioPopLast", last["ratio_population"], src, 2)
            put(f"{p}RatioPLast", last["ratio_well_specified"], src, 2)
            put(f"{p}Pass", "yes" if v["checks"]["pass"] else "no", src)
            # Per-run error (prediction - measurement) / mean measurement, at the largest n,
            # next to the spread of the measurement and of the prediction alone.
            put(f"{p}ErrSd", last["relative_error_sd"], src, 2)
            put(f"{p}ErrLo", last["relative_error_q10_q50_q90"][0], src, 2)
            put(f"{p}ErrHi", last["relative_error_q10_q50_q90"][2], src, 2)
            put(f"{p}MeasSd", last["sd_improvement"] / last["mean_improvement"], src, 2)
            put(f"{p}PredSd", last["sd_plugin_prediction"] / last["mean_improvement"], src, 2)
            put(f"{p}FracNeg", pct(last["fraction_negative_improvement"]), src)
            put(f"{p}TraceRatioFirst", first["plugin_trace_ratio"], src, 2)
            put(f"{p}TraceRatioLast", last["plugin_trace_ratio"], src, 2)
            put(f"{p}ZeroOneLast", last["mean_improvement_01"], src, 4)
            put(f"{p}ZeroOneSeLast", last["se_improvement_01"], src, 4)
            put(f"{p}ZeroOneFirst", first["mean_improvement_01"], src, 4)
            put(f"{p}Separable", sum(q["separable_runs"] for q in v["per_n"]), src)
            # Test-pool noise floor of one run's measured improvement, relative to the mean.
            put(f"{p}FloorRel", pct(last["mean_floor"] / last["mean_improvement"]), src)

    if s := load("e4_false_alarms"):
        src = "e4_false_alarms.json"
        put("EFourScorerB", s["scorer"]["B"], src)
        put("EFourScorerValError", s["scorer"]["validation_error"], src)
        put("EFourFarHalfwidth", s["true_far_hoeffding_halfwidth"], src, 4)
        put("EFourPass", "yes" if s["checks"]["violation_fraction_le_delta_every_m"] else "no", src)
        put("EFourSmallestM", s["smallest_m_certifiable_any_repeat"] or "\\text{none}", src)
        put("EFourSmallestMAll", s["smallest_m_certifiable_every_repeat"] or "\\text{none}", src)
        # Oracle reference: the most sensitive grid threshold whose true FAR (far-test pool) is
        # <= alpha, i.e. what one could pick knowing p(t); certification pays the difference.
        alpha = load_full("e4_false_alarms")["config"]["alpha"]
        ok = [j for j, f in enumerate(s["true_far"]) if f <= alpha]
        j = max(ok, key=lambda i: s["true_far"][i]) if ok else None
        put("EFourOracleFar", s["true_far"][j] if ok else "\\text{--}", src, 4)
        put("EFourOracleDetect", s["flag_rate_defective"][j] if ok else "\\text{--}", src)
        put("EFourAlpha", alpha, src, 2)
        # Certification needs at least the margin below alpha: sqrt(ln(T/delta)/(2m)) < alpha.
        ea_cfg = load_full("e4_false_alarms")["config"]
        put("EFourMinMPossible",
            math.floor(math.log(ea_cfg["T"] / ea_cfg["delta"]) / (2 * alpha**2)) + 1, src)
        put("EFourT", ea_cfg["T"], src)
        put("EFourRepeats", ea_cfg["repeats"], src)
        put("EFourNTrain", ea_cfg["n_train"], src)
        put("EFourNFar", ea_cfg["n_far"], src)
        # Widening of the margin from protecting all T thresholds instead of one.
        put("EFourViolBoundZero", pct(violation_rate_bound(0, ea_cfg["repeats"], ea_cfg["delta"])),
            src)
        put("EFourUnionFactor", math.sqrt(math.log(ea_cfg["T"] / ea_cfg["delta"])
                                       / math.log(1 / ea_cfg["delta"])), src, 2)
        with open(ROOT / "results" / "e4_false_alarms_raw.csv") as f:
            far = [float(r["true_far"]) for r in csv.DictReader(f) if int(r["selected"]) >= 0]
        put("EFourMaxSelectedFar", max(far), "e4_false_alarms_raw.csv", 4)
        # Multiplicative-Chernoff rule: certifiable only if eta = sqrt(2 ln(T/delta)/(alpha m)) < 1,
        # i.e. m > 2 ln(T/delta) / alpha.
        put("EFourChMinMPossible",
            math.floor(2 * math.log(ea_cfg["T"] / ea_cfg["delta"]) / alpha) + 1, src)
        put("EFourChPass", "yes" if s["checks"]["chernoff_violation_fraction_le_delta_every_m"]
            else "no", src)
        put("EFourChSmallestM", s["chernoff_smallest_m_certifiable_any_repeat"]
            or "\\text{none}", src)
        put("EFourChSmallestMAll", s["chernoff_smallest_m_certifiable_every_repeat"]
            or "\\text{none}", src)
        with open(ROOT / "results" / "e4_false_alarms_raw.csv") as f:
            far = [float(r["true_far_chernoff"]) for r in csv.DictReader(f)
                   if int(r["selected_chernoff"]) >= 0]
        put("EFourChMaxSelectedFar", max(far), "e4_false_alarms_raw.csv", 4)
        names = {300: "ThreeHundred", 500: "FiveHundred", 1000: "OneK", 2000: "TwoK",
                 4000: "FourK", 8000: "EightK"}
        for q in s["per_m"]:
            p = "EFour" + names[q["m"]]
            put(f"{p}Cert", pct(q["fraction_certifiable"]), src)
            put(f"{p}Viol", pct(q["violation_fraction"]), src)
            # 1 - delta upper bound on the violation rate from v violations in R repeats.
            v = round(q["violation_fraction"] * ea_cfg["repeats"])
            put(f"{p}ViolHi", pct(violation_rate_bound(v, ea_cfg["repeats"], ea_cfg["delta"])),
                src)
            put(f"{p}Margin", q["margin"], src, 4)
            put(f"{p}TrueFar", q["mean_true_far"] if q["mean_true_far"] is not None
                else "\\text{--}", src, 4)
            put(f"{p}CertBound", q["mean_certified"] if q["mean_certified"] is not None
                else "\\text{--}", src, 4)
            put(f"{p}Detect", q["mean_flag_rate_defective"]
                if q["mean_flag_rate_defective"] is not None else "\\text{--}", src)
            c, pc = q["chernoff"], "EFourCh" + names[q["m"]]
            put(f"{pc}Eta", c["eta"], src, 2)
            put(f"{pc}Cert", pct(c["fraction_certifiable"]), src)
            put(f"{pc}Viol", pct(c["violation_fraction"]), src)
            put(f"{pc}ViolHi", pct(violation_rate_bound(c["violations"], ea_cfg["repeats"],
                                                         ea_cfg["delta"])), src)
            for key, field in (("TrueFar", "mean_true_far"), ("CertBound", "mean_certified")):
                put(f"{pc}{key}", c[field] if c[field] is not None else "\\text{--}", src, 4)
            put(f"{pc}Detect", c["mean_flag_rate_defective"]
                if c["mean_flag_rate_defective"] is not None else "\\text{--}", src)

    # Pool sizes actually used (README pool table), each read from the results file that used it.
    tags = ("e1_thresholds", "e2_linear_bounds", "e3_learning_curve", "e4_false_alarms")
    cfgs = {tag: load_full(tag)["config"] for tag in tags
            if (ROOT / "results" / f"{tag}.json").exists()}
    if len(cfgs) == 4:
        eb, ec, ed, ea = (cfgs[k] for k in tags)
        put("PoolFeatureFit", ec["n_fit"], "e2_linear_bounds.json")
        put("PoolValidation", ec["n_validation"], "e2_linear_bounds.json")
        put("PoolTest", ec["n_test"], "e2_linear_bounds.json")
        put("PoolPopulationETwo", ec["n_population"], "e2_linear_bounds.json")
        put("PoolPopulationEThree", ed["n_population"], "e3_learning_curve.json")
        put("PoolTrainEOne", eb["repeats"] * 2 ** max(eb["log2_n"]), "e1_thresholds.json")
        put("PoolTrainETwo", ec["repeats"] * 2 ** max(ec["log2_n"]), "e2_linear_bounds.json")
        ed_n = max(v["per_n"][-1]["n"] for v in load("e3_learning_curve")["sets"].values())
        put("PoolTrainEThree", ed["repeats"] * ed_n, "e3_learning_curve.json")
        put("PoolCertification", ea["repeats"] * max(ea["m_values"]), "e4_false_alarms.json")
        put("PoolFarTest", ea["n_far"], "e4_false_alarms.json")
        put("PoolDefectiveDies", ea["n_defective"], "e4_false_alarms.json")
    return out


def pct(x: float) -> str:
    """Fraction -> percent string with one decimal ("12.5\\%")."""
    return f"{100 * x:.1f}\\%"


def main() -> None:
    out = collect()
    gen = ROOT / "theory" / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated by experiments/theory_tables.py from results/*.json. Do not edit.\n"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}  % {src}\n" for k, (v, src) in sorted(out.items())]
    (gen / "numbers.tex").write_text("".join(lines))
    (ROOT / "results" / "theory_constants.json").write_text(json.dumps(
        {k: {"value": v, "source": src} for k, (v, src) in sorted(out.items())}, indent=2))
    print(f"wrote {len(out)} macros to theory/generated/numbers.tex")


if __name__ == "__main__":
    main()
