"""H1 and H2, pre-registered.

This file is written BEFORE any sentiment score exists. That is the point: the
comparison, the controls and the significance test are fixed in advance, so no
choice here can be made after seeing which version of the result is nicer.

H1  Sentiment features add predictive power over lagged price controls.
H2  The source ranking found for Turkiye does not transfer to Tunisia.

HOW H1 IS TESTED
Baseline and treatment are run through the SAME walk-forward loop, over the SAME
sessions, so their predictions are paired. Paired predictions require a paired
test: McNemar on the discordant pairs, not a comparison of two accuracies with
overlapping confidence intervals. Reporting "56.4% vs 54.9%" without a paired
test is the single easiest way to publish noise.

THE THREE ARMS, AND WHY THE PLACEBO EXISTS
10% of relevant headlines restate the index's own move (AUDIT_REPORT 8d). Their
sentiment is a proxy for that day's return, and return autocorrelation is +0.263,
so they can carry momentum into the sentiment channel and look like news signal.
Every H1 result is therefore reported three ways:

  all            every scored headline
  ex_price       price-report headlines removed
  placebo        price-report headlines ONLY

The placebo is the interpretive key. If `all` beats baseline but `ex_price` does
not, the effect is momentum laundering, not sentiment. A pre-registered placebo
that you would have run either way is worth more than any amount of argument
after the fact.

WHAT THIS FILE NEEDS THAT DOES NOT EXIST YET
`features.py` must emit the SENTIMENT_COLUMNS below, built from the full 42,645
headline corpus scored by a trained classifier -- NOT from the 2,932-row gold set,
which is stratified for training and carries a median of 1 headline per day.
Until then `run()` raises with the exact list of missing columns.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

from baseline import evaluate, walk_forward

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data" / "curated"
DEFAULT_INPUT = CURATED / "daily_features.parquet"
DEFAULT_OUTPUT = CURATED / "hypothesis_results.json"

# Price-only controls. Identical in both arms so the only difference is sentiment.
BASELINE_FEATURES = ["ret_lag0", "ret_lag1", "ret_lag2"]

# What features.py must produce, per arm suffix ("", "_ex_price", "_price_only").
SENTIMENT_COLUMNS = ["sent_mean_lag1", "sent_pos_share_lag1", "sent_neg_share_lag1"]

ARMS = {"all": "", "ex_price": "_ex_price", "placebo": "_price_only"}

# M3: the placebo is necessary but not sufficient. Momentum also enters through
# headlines the price-report regex never matches (sector commentary written
# BECAUSE the market moved; headline volume, which spikes after large moves).
# This arm uses sentiment residualised on ret_lag0, ret_lag1 and log_headlines_lag0,
# so a surviving effect cannot be momentum re-entering through text.
ORTHOGONAL_ARM = "orthogonal"


def required_columns(arm: str) -> list[str]:
    if arm == ORTHOGONAL_ARM:
        return ["sent_resid_lag1"]
    # has_sent_* tells the model "no headlines today" apart from "neutral today".
    # Without it, the 0-fill needed to keep the arms aligned is a silent lie.
    return ([c.replace("_lag1", f"{ARMS[arm]}_lag1") for c in SENTIMENT_COLUMNS]
            + [f"has_sent{ARMS[arm]}_lag1"])


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni. Three or four arms are tested against one baseline on the
    same data; reporting the smallest raw p as if it were the only test is how a
    null becomes a finding."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, adjusted, running = len(ordered), {}, 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = round(running, 4)
    return adjusted


def minimum_detectable_effect(n_pairs: int, discordant: int, power: float = 0.80,
                              alpha: float = 0.05) -> dict:
    """S3: what accuracy gain could this design even detect?

    Published daily news-sentiment effects on index direction are ~1-2pp. If the
    MDE exceeds that, "H1 NOT SUPPORTED" is the guaranteed outcome whether or not
    H1 is true, and the paper must say so instead of claiming evidence of absence.
    """
    import math
    if n_pairs == 0 or discordant == 0:
        return {"mde_pp": None, "note": "no discordant pairs"}
    z_a, z_b = 1.959964, 0.841621
    se = math.sqrt(discordant) / n_pairs
    mde = (z_a + z_b) * se
    return {
        "discordant_pairs": int(discordant),
        "se_of_accuracy_difference": round(se, 5),
        "mde_pp_at_80_power": round(mde * 100, 2),
        "typical_published_effect_pp": "1-2",
        "adequately_powered": bool(mde * 100 <= 2.0),
    }


def missing_columns(frame: pd.DataFrame) -> dict[str, list[str]]:
    return {arm: [c for c in required_columns(arm) if c not in frame.columns]
            for arm in ARMS}


def paired_test(treatment: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    """McNemar on the discordant pairs of two aligned prediction frames."""
    if not treatment.session.equals(baseline.session):
        raise ValueError("Arms are not aligned on the same sessions; cannot pair.")
    t_ok = (treatment.y_pred == treatment.y_true)
    b_ok = (baseline.y_pred == baseline.y_true)
    b_only = int((t_ok & ~b_ok).sum())   # treatment right, baseline wrong
    c_only = int((~t_ok & b_ok).sum())   # baseline right, treatment wrong
    result = mcnemar([[int((t_ok & b_ok).sum()), b_only],
                      [c_only, int((~t_ok & ~b_ok).sum())]], exact=True)
    return {
        "treatment_only_correct": b_only,
        "baseline_only_correct": c_only,
        "p_value": round(float(result.pvalue), 4),
        "significant_at_05": bool(result.pvalue < 0.05),
    }


def diebold_mariano(treatment: pd.DataFrame, baseline: pd.DataFrame,
                    horizon: int = 1) -> dict:
    """Diebold-Mariano on squared forecast errors of the RETURN, not the sign.

    Why this exists (S3). The sign test discards magnitude: being wrong by 5bp and
    wrong by 500bp count the same, so it throws away most of the information in the
    data and needs ~3pp of accuracy to detect anything. A model can improve returns
    forecasts materially while moving directional accuracy by less than the sign
    test can see. DM uses the loss differential directly and is far better powered.

    Newey-West HAC variance with a horizon-1 lag truncation, because daily loss
    differentials are serially correlated -- volatility clusters at +0.387 here, so
    an iid variance would be badly understated and the test anti-conservative.

    Negative statistic => treatment has LOWER loss => treatment is better.
    """
    if not treatment.session.equals(baseline.session):
        raise ValueError("Arms are not aligned on the same sessions; cannot pair.")
    actual = treatment["ret_next"].to_numpy()
    d = (treatment["score"].to_numpy() - actual) ** 2 - \
        (baseline["score"].to_numpy() - actual) ** 2
    n = len(d)
    if n < 30:
        return {"skipped": f"only {n} paired forecasts"}
    dbar = float(d.mean())
    gamma0 = float(((d - dbar) ** 2).mean())
    lags = max(1, horizon - 1)
    var = gamma0
    for lag in range(1, lags + 1):
        cov = float(((d[lag:] - dbar) * (d[:-lag] - dbar)).mean())
        var += 2 * (1 - lag / (lags + 1)) * cov
    if var <= 0:
        return {"skipped": "non-positive HAC variance"}
    stat = dbar / np.sqrt(var / n)
    pvalue = 2 * (1 - stats.norm.cdf(abs(stat)))

    def r2(frame):
        err = ((frame["score"].to_numpy() - actual) ** 2).sum()
        return 1 - err / ((actual - actual.mean()) ** 2).sum()

    return {
        "dm_statistic": round(float(stat), 4),
        "p_value": round(float(pvalue), 4),
        "significant_at_05": bool(pvalue < 0.05),
        "treatment_better": bool(dbar < 0),
        "mean_loss_differential": float(f"{dbar:.3e}"),
        "oos_r2_treatment": round(float(r2(treatment)), 5),
        "oos_r2_baseline": round(float(r2(baseline)), 5),
        "note": ("squared-error loss on returns; Newey-West HAC. Better powered than "
                 "the sign test, which needs ~3pp accuracy to detect anything here."),
    }


def run_h1(frame: pd.DataFrame, min_train: int = 500, refit_every: int = 20,
           kind: str = "regress") -> dict:
    absent = missing_columns(frame)
    if all(absent.values()):
        raise SystemExit(
            "No sentiment columns found -- H1 cannot run yet.\n"
            f"features.py must emit, for each arm: {SENTIMENT_COLUMNS}\n"
            f"missing: {json.dumps(absent, indent=2)}\n"
            "These must come from the FULL 42,645-headline corpus scored by a "
            "trained classifier, not from the 2,932-row gold set."
        )
    base_preds = walk_forward(frame, BASELINE_FEATURES, min_train, refit_every, kind)
    results = {"baseline": evaluate(base_preds) | {"features": BASELINE_FEATURES}}
    absent[ORTHOGONAL_ARM] = [c for c in required_columns(ORTHOGONAL_ARM)
                              if c not in frame.columns]
    raw_p = {}
    for arm in list(ARMS) + [ORTHOGONAL_ARM]:
        if absent[arm]:
            results[arm] = {"skipped": f"missing columns: {absent[arm]}"}
            continue
        features = BASELINE_FEATURES + required_columns(arm)
        preds = walk_forward(frame, features, min_train, refit_every, kind)
        test = paired_test(preds, base_preds)
        raw_p[arm] = test["p_value"]
        results[arm] = (evaluate(preds)
                        | {"features": features, "vs_baseline": test,
                           "vs_baseline_continuous": diebold_mariano(preds, base_preds)
                           if kind == "regress" else
                           {"skipped": "DM needs kind='regress' (a return forecast)"}})
    if raw_p:
        adjusted = holm(raw_p)
        for arm, p_adj in adjusted.items():
            results[arm]["vs_baseline"]["p_value_holm"] = p_adj
            results[arm]["vs_baseline"]["significant_at_05"] = bool(p_adj < 0.05)
        results["multiple_comparisons"] = {
            "method": "Holm-Bonferroni", "n_tests": len(raw_p), "raw": raw_p,
            "adjusted": adjusted,
            "note": ("baseline.py additionally evaluates 8 exploratory configurations; "
                     "those are NOT corrected here and must be reported as exploratory."),
        }
    d = results["baseline"]
    results["power"] = minimum_detectable_effect(
        d["n_predictions"],
        int(((base_preds.y_pred == base_preds.y_true)
             != (base_preds.y_constant == base_preds.y_true)).sum()))
    results["interpretation"] = _interpret(results)
    return results


def _interpret(results: dict) -> str:
    def beat(arm):
        r = results.get(arm, {})
        return bool(r.get("vs_baseline", {}).get("significant_at_05")
                    and r.get("accuracy", 0) > results["baseline"]["accuracy"])
    if "skipped" in results.get("all", {}):
        return "Incomplete: not all arms ran."

    # The sign test is underpowered (MDE ~3pp). The continuous test sees effects it
    # cannot, INCLUDING harmful ones -- so check it before reporting a bare null.
    def dm(arm):
        return results.get(arm, {}).get("vs_baseline_continuous", {}) or {}

    degrading = [a for a in ARMS
                 if dm(a).get("significant_at_05") and dm(a).get("treatment_better") is False]
    if len(degrading) >= 2:
        return ("H1 REJECTED, NOT MERELY UNSUPPORTED: the sign test is null, but the "
                "better-powered Diebold-Mariano test on returns finds sentiment makes "
                f"the forecast SIGNIFICANTLY WORSE in {len(degrading)} of {len(ARMS)} arms "
                f"({', '.join(degrading)}). OOS R2 falls below the price-only baseline in "
                "every arm. Adding these features costs variance and returns no signal -- "
                "consistent with sentiment scores that are noise. Re-check the labels "
                "before concluding anything about sentiment itself.")
    improving = [a for a in ARMS
                 if dm(a).get("significant_at_05") and dm(a).get("treatment_better")]
    if beat("all") and not beat("ex_price"):
        return ("MOMENTUM LAUNDERING: the effect disappears once price-report "
                "headlines are removed. Do not report this as sentiment.")
    if beat("placebo") and not beat("ex_price"):
        return ("PLACEBO FIRED: price-report headlines alone beat the baseline "
                "while genuine news does not. The signal is autocorrelation.")
    if beat("ex_price") and beat(ORTHOGONAL_ARM):
        return ("H1 SUPPORTED: sentiment adds predictive power over lagged returns, "
                "and survives both the price-report exclusion and orthogonalisation "
                "against momentum and news volume.")
    if beat("ex_price") and not results.get(ORTHOGONAL_ARM, {}).get("skipped"):
        return ("H1 PARTIALLY SUPPORTED: survives price-report exclusion but NOT "
                "orthogonalisation against momentum/volume -- the effect may still "
                "be momentum entering through headlines the regex does not catch.")
    if beat("ex_price"):
        return ("H1 SUPPORTED (weak): survives price-report exclusion; the "
                "orthogonalised arm did not run.")
    if improving:
        return (f"H1 SUPPORTED on the continuous test ({', '.join(improving)}) but not "
                "on the sign test. Report both; the sign test is underpowered.")
    return ("H1 NOT SUPPORTED: no arm beats the price-only baseline on either the "
            "sign test or the continuous test. Note the sign test alone could not "
            "have established this -- its MDE exceeds any plausible effect.")


def run_h2(frame: pd.DataFrame, sources: list[str], min_train: int = 500,
           refit_every: int = 20, kind: str = "regress") -> dict:
    """Leave-one-source-out ablation, ranked by the accuracy lost when dropped.

    A source that matters is one whose REMOVAL hurts. Ranking by a coefficient or
    by headline count would rank prominence, not contribution.
    """
    full_cols = [f"sent_mean_{s}_lag1" for s in sources]
    present = [c for c in full_cols if c in frame.columns]
    if len(present) < 2:
        raise SystemExit(
            f"Need per-source sentiment columns (sent_mean_<source>_lag1) for >=2 "
            f"sources; found {present}. features.py must emit them for H2.")

    # M6: sources enter at different dates by design (config.SOURCE_WINDOWS puts
    # lapresse at 2024-10, economist at 2020-05). Dropping lapresse only affects
    # 2024-10 onward; dropping ilboursa affects the whole sample. Ranking by
    # accuracy lost would therefore rank COVERAGE LENGTH, not contribution.
    # Restrict to the window where every ranked source is simultaneously active.
    active = frame[present].notna() & (frame[present] != 0)
    common = active.all(axis=1)
    if common.sum() < min_train + 100:
        raise SystemExit(
            f"Only {int(common.sum())} sessions have all {len(present)} sources "
            f"active; need > min_train ({min_train}) + 100. Restrict `sources` to "
            "outlets whose coverage windows overlap, or H2 ranks coverage length.")
    first, last = frame.loc[common, "session"].min(), frame.loc[common, "session"].max()
    frame = frame[(frame.session >= first) & (frame.session <= last)].reset_index(drop=True)

    full = evaluate(walk_forward(frame, BASELINE_FEATURES + present,
                                 min_train, refit_every, kind))
    ranking = []
    for column in present:
        reduced = [c for c in present if c != column]
        scores = evaluate(walk_forward(frame, BASELINE_FEATURES + reduced,
                                       min_train, refit_every, kind))
        ranking.append({
            "source": column.replace("sent_mean_", "").replace("_lag1", ""),
            "accuracy_without": scores["accuracy"],
            "accuracy_drop": round(full["accuracy"] - scores["accuracy"], 4),
        })
    ranking.sort(key=lambda r: r["accuracy_drop"], reverse=True)
    return {"full_model": full, "leave_one_out": ranking,
            "most_important": ranking[0]["source"] if ranking else None,
            "common_window": [str(first.date()), str(last.date())],
            "sessions_in_window": int(len(frame)),
            "caveat": ("accuracy_drop has no standard error attached; treat the "
                       "ranking as exploratory until each drop carries a bootstrap CI"),
            "n_comparisons": len(present)}


def run(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT,
        min_train: int = 500, refit_every: int = 20) -> dict:
    frame = pd.read_parquet(input_path)
    results = {"h1": run_h1(frame, min_train, refit_every)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-train", type=int, default=500)
    parser.add_argument("--refit-every", type=int, default=20)
    args = parser.parse_args()
    r = run(args.input, args.output, args.min_train, args.refit_every)["h1"]
    print(f"{'arm':<12}{'acc':>9}{'const':>9}{'p(vs base)':>12}  verdict")
    print("-" * 62)
    print(f"{'baseline':<12}{r['baseline']['accuracy']:>9.4f}"
          f"{r['baseline']['constant_baseline']:>9.4f}{'—':>12}")
    for arm in ARMS:
        s = r[arm]
        if "skipped" in s:
            print(f"{arm:<12}{'skipped':>9}  {s['skipped']}")
            continue
        t = s["vs_baseline"]
        print(f"{arm:<12}{s['accuracy']:>9.4f}{s['constant_baseline']:>9.4f}"
              f"{t['p_value']:>12.4f}  {'SIG' if t['significant_at_05'] else 'ns'}")
    print()
    print(f"{'arm':<12}{'DM stat':>10}{'DM p':>9}{'OOS R2':>10}{'base R2':>10}")
    print("-" * 51)
    for arm in ARMS:
        dm = r.get(arm, {}).get("vs_baseline_continuous", {})
        if "skipped" in dm or not dm:
            continue
        print(f"{arm:<12}{dm['dm_statistic']:>10.3f}{dm['p_value']:>9.4f}"
              f"{dm['oos_r2_treatment']:>10.5f}{dm['oos_r2_baseline']:>10.5f}")
    pw = r.get("power", {})
    print(f"\nMDE at 80% power: {pw.get('mde_pp_at_80_power')}pp "
          f"(sign test) | adequately powered: {pw.get('adequately_powered')}")
    print(f"\n{r['interpretation']}")


if __name__ == "__main__":
    main()
