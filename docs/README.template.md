# Wafer nuisance filter: how many labels does it need?

A two-stage die-to-die inspection pipeline on synthetic wafer dies. Stage 1 is plain computer
vision: align each die to a reference, threshold the difference image and take connected
components as defect candidates. Stage 2 is a learned nuisance filter that separates real defects
from the nuisance candidates. The question is **how many labeled candidates the filter needs, and
how well standard learning theory predicts that number**.

![headline](figures/headline.png)

| Result | What it says | Measured |
|---|---|---|
| **1** Threshold on peak contrast | VC dimension 1: error Õ(1/n) if separable, excess error Õ(1/√n) otherwise | Bounds hold in {{EOneMinFracWithinBound}} of runs, but the agnostic bound is {{EOneOvlBoundRatioLast}}× the measured excess at n = {{EOneNLast}}. Fitted slopes: {{EOneOvlSlope}} (overlapping), {{EOneSepSlope}} (separable) |
| **2** Linear filter, VC and Rademacher bounds | Both bound the 0-1 error of every filter in the class | Both above the test error in {{ETwoMinFracBoth}} of runs. The VC bound (hand-crafted features) beats the trivial error {{TrivialError}} from n = {{ETwoHandFirstNVC}} and is {{ETwoHandVCRatioLast}}× the test error at n = {{ETwoNLast}}. The Rademacher bound never beats it ({{ETwoRadLastMin}}–{{ETwoRadLastMax}} at n = {{ETwoNLast}}) |
| **3** Learning curve of logistic regression | Excess logistic loss ≈ Tr(H⁻¹G)/2n | measured / predicted at n = 300p: {{EThreePcaFourRatioLast}}, {{EThreePcaSixteenRatioLast}}, {{EThreePcaSixtyFourRatioLast}} (p = {{EThreePcaFourP}}, {{EThreePcaSixteenP}}, {{EThreePcaSixtyFourP}}); Wald coverage {{EThreePcaFourCoverage}}, {{EThreePcaSixteenCoverage}}, {{EThreePcaSixtyFourCoverage}} |
| **3b** What {{EThreeBFactor}}× more labels will buy | The same formula, with H and G estimated from the labels already in hand | plug-in / measured gain at n = {{EThreeBNMaxOverP}}p: {{EThreeBPcaFourRatioLast}}, {{EThreeBPcaSixteenRatioLast}}, {{EThreeBPcaSixtyFourRatioLast}}; at n = {{EThreeBNMinOverP}}p only {{EThreeBRatioFirstMin}}–{{EThreeBRatioFirstMax}} |
| **4** Certified die-level false-alarm rate | Hoeffding + a union bound over T = {{EFourT}} thresholds certifies FAR ≤ α = {{EFourAlpha}} | No violations in {{EFourRepeats}} repeats at any m (violation rate ≤ {{EFourViolBoundZero}} at 95%). Certifiable from m = {{EFourSmallestM}} clean dies, where the selected threshold's true FAR is {{EFourTwoKTrueFar}} |

## What I found

- **The bounds hold, but none of them is a usable label budget.** The tightest one (VC,
  hand-crafted features) is still {{ETwoHandVCRatioLast}}× the measured error at n = {{ETwoNLast}}.
- **The Rademacher bound is looser than VC here**, for every feature set at the validated norm
  bound B. It is dimension-free, but the overlapping classes force a large ‖w‖, and its
  complexity term grows with B. Only for the 384-dimensional CNN features does the filter that
  minimizes its own Rademacher bound ({{ETwoCnnFullRadCertLast}}) beat VC ({{ETwoCnnFullVCLast}}).
- **The separable-regime slope check failed.** I expected a 1/n rate and fixed the range
  [−1.15, −0.85] before running; the fit gave {{EOneSepSlope}} on only {{EOneSepFitPoints}} points. The
  separable pool has a margin between nuisance and defects, so once the learned threshold lands
  in the gap the error is exactly zero, and the mean falls faster than 1/n. The bound itself
  holds; it is the worst case over distributions, and this distribution is easier.
- **The asymptotic learning curve is the useful prediction.** Tr(H⁻¹G)/2n is within 10% of the
  measured excess once n ≥ {{EThreePcaFourFromNP}}–{{EThreePcaSixteenFromNP}}·p, and below that it
  underestimates it. Estimating H and G from the training labels (3b) costs almost nothing. It
  predicts the *average* gain from more labels, though; one batch's gain differs from it with a
  standard deviation of {{EThreeBPcaSixtyFourErrSd}}–{{EThreeBPcaFourErrSd}} times the mean gain.
- **Ten hand-crafted features match a 384-dimensional CNN embedding** in best-in-class error
  ({{ETwoHandLStar}} vs. {{ETwoCnnFullLStar}}) and are at least as good at every training size tested.

Statements, complete proofs, and the measured-vs-theory comparison for each result are in
[`theory/main.pdf`](theory/main.pdf). Every number in this README and the PDF is generated from
a file in [`results/`](results/).

## Run it

```bash
uv sync
uv run pytest                                  # unit tests (bounds vs. hand values, gradients, cache keys)
uv run python experiments/run_all.py --quick   # smoke run with small pools (*_quick outputs)
uv run python experiments/run_all.py           # everything: experiments, numbers, figures, README, PDF
uv run python experiments/<name>.py --plot-only  # redraw one figure from results/
```

Run the full pipeline from a clean git tree: each results file records the commit it came from.
Most of the full run is CNN feature extraction, cached under `cache/` and reused across
experiments. Rebuilding the PDF needs `latexmk`.

```
src/wafer_nuisance/   data (die generator, pools, sampler), detect (stage 1), features
                      (hand-crafted, CNN, normalization, cache), learn, bounds (Results 1-4), eval
experiments/          one script per experiment (e1 ... e4) + run_all, theory_tables, headline,
                      build_readme
theory/               main.tex + one file per section, generated/numbers.tex, main.pdf
results/, figures/    generated outputs
docs/                 README.template.md (this README is generated from it)
```

## Limitations

- **Synthetic data.** The die generator is a stand-in for real inspection images. Its nuisance
  sources (local warp, linewidth variation, intensity blobs, line-edge roughness) are plausible
  but not calibrated to a real tool.
- **Independent dies.** Every result assumes i.i.d. dies. Real dies share wafer- and lot-level
  variation, so this is the first assumption to break on real data.
- **Population quantities.** L(h\*) and Result 3's θ\*, H and G come from a large labeled
  population pool. Result 3b removes that need for H and G, but only for the expected gain, and
  only once n is large enough for the asymptotics.
- **Elementary tools.** I kept to elementary tools (Hoeffding rather than variance-adaptive or
  exact binomial intervals, no cross-validation), which is part of why Result 4 is conservative.
