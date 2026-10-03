"""Which outlet's CamemBERT sentiment carries weight for the next-session Tunindex?

    python3 preprocessing/outlet_ranking.py
    -> data/curated/outlet_ranking_results.json

Per-outlet H2 analysis, run through the pre-registered harness (baseline.walk_forward:
expanding window, min_train 500, refit every 20, 5-session embargo) so every
leakage guard carries over. Three views per outlet, because they answer different
questions:

  add_one        baseline + this outlet vs the price-only baseline, out of sample.
                 THE test: does the outlet add anything a trader could use?
                 Diebold-Mariano on returns (primary, Holm across outlets) and
                 McNemar on signs.
  leave_one_out  all outlets vs all-but-this-one, out of sample: the outlet's
                 UNIQUE contribution given the others (the H2 ablation design).
  weight         the outlet's standardised coefficient in one in-sample OLS with
                 every outlet and the price controls, Newey-West t-stat. What the
                 word "weight" means, but in-sample: descriptive, never a result.

WINDOW. 2019 onward, sent_source == "yearly_refit" rows only. Before 2019 the
sentiment is either the saved model's fill (trained on labels up to 2026 -- look-
ahead) or from under-trained yearly models. Using those rows would leak.

COVERAGE. An outlet enters as two columns: its daily mean score (0 when it
published nothing) and a has_<outlet> indicator, so "silent today" never reads as
"neutral today". An outlet is tested only if it has >= MIN_SESSIONS sessions of
news AND >= MIN_NON_NEUTRAL of its headlines scored non-neutral. Otherwise it is
reported with the reason and not tested: it cannot move a ~1,400-prediction test,
and testing it would only dilute the Holm correction for the others.

The second rule exists because CamemBERT is blind on English: in 2019+ it scores
Guardian 162/162 neutral, NYT 207/210, Economist 95/97. Their "sentiment" is a
constant, so a test on it would measure only whether the outlet published.

PRICE REPORTS restate the index's own move (config.PRICE_REPORT_PATTERN) and
launder momentum into sentiment. Every analysis runs twice, with and without them.
An outlet that only wins with them in is carrying momentum, not news.

English-outlet scores come from CamemBERT, a French model trained on 2,287 French
and 52 English gold rows: treat any English-outlet result as instrument-limited.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

import config
from baseline import evaluate, walk_forward
from hypothesis_tests import BASELINE_FEATURES, diebold_mariano, holm, paired_test

CURATED = Path(__file__).resolve().parent.parent / "data" / "curated"
SERIES = CURATED / "tunindex_headline_series.csv"
FRAME = CURATED / "tunindex_timeseries.csv"
HEADLINES = CURATED / "03_dedup.parquet"
OUTPUT = CURATED / "outlet_ranking_results.json"
START = "2019-01-01"      # first year every yearly CamemBERT model had >= 724 gold rows
MIN_SESSIONS = 100
MIN_NON_NEUTRAL = 0.10
VARIANTS = ("all_headlines", "ex_price_reports")


def panel(ex_price: bool) -> tuple[pd.DataFrame, dict[str, int], dict[str, float]]:
    """One row per session from START: price controls, target, and per-outlet
    sentiment + has_ indicator. Headline rows are already aligned to the session
    whose ret_next they may predict (build_headline_series.py): no shift."""
    news = pd.read_csv(SERIES, parse_dates=["session"])
    news = news[(news["sent_source"] == "yearly_refit") & (news["session"] >= START)]
    if ex_price:
        french = pd.read_parquet(HEADLINES, columns=["row_id", "headline_clean"])
        news = news.merge(french, on="row_id", validate="one_to_one")
        news = news[~news["headline_clean"].astype(str).str.contains(
            config.PRICE_REPORT_PATTERN, case=False, regex=True, na=False)]
    wide = news.pivot_table(index="session", columns="source", values="sent_score", aggfunc="mean")
    non_neutral = news.groupby("source")["sent_label"].apply(
        lambda s: round(float((s != "neutral").mean()), 3)).to_dict()

    frame = pd.read_csv(FRAME, parse_dates=["session"])
    frame = frame.loc[frame["session"] >= START, ["session", "ret_next"] + BASELINE_FEATURES]
    frame = frame.merge(wide.add_prefix("sent_").reset_index(), on="session", how="left")
    coverage = {}
    for source in wide.columns:
        frame[f"has_{source}"] = frame[f"sent_{source}"].notna().astype(int)
        frame[f"sent_{source}"] = frame[f"sent_{source}"].fillna(0.0)
        coverage[source] = int(frame[f"has_{source}"].sum())
    assert frame[BASELINE_FEATURES].notna().all().all()
    return frame.reset_index(drop=True), coverage, non_neutral


def columns(sources: list[str]) -> list[str]:
    return [c for s in sources for c in (f"sent_{s}", f"has_{s}")]


def versus(frame: pd.DataFrame, features: list[str], base: pd.DataFrame,
           base_auc: float, kind: str) -> dict:
    preds = walk_forward(frame, BASELINE_FEATURES + features, kind=kind)
    scores = evaluate(preds)
    return {"roc_auc": scores["roc_auc"], "delta_auc": round(scores["roc_auc"] - base_auc, 4),
            "accuracy": scores["accuracy"],
            "sign_test": paired_test(preds, base),
            "dm_test": diebold_mariano(preds, base)}


def weights(frame: pd.DataFrame, sources: list[str]) -> dict:
    cols = BASELINE_FEATURES + columns(sources)
    data = frame.dropna(subset=["ret_next"])
    X = (data[cols] - data[cols].mean()) / data[cols].std()
    fit = sm.OLS(data["ret_next"] / data["ret_next"].std(), sm.add_constant(X)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5})
    return {s: {"beta_std": round(float(fit.params[f"sent_{s}"]), 4),
                "t_hac": round(float(fit.tvalues[f"sent_{s}"]), 2),
                "p_hac": round(float(fit.pvalues[f"sent_{s}"]), 4)} for s in sources}


def run_variant(ex_price: bool, kinds: tuple[str, ...]) -> dict:
    frame, coverage, non_neutral = panel(ex_price)
    skipped = {s: (f"{coverage[s]} sessions < {MIN_SESSIONS}" if coverage[s] < MIN_SESSIONS else
                   f"{non_neutral[s]:.1%} non-neutral < {MIN_NON_NEUTRAL:.0%} (instrument blind)")
               for s in coverage
               if coverage[s] < MIN_SESSIONS or non_neutral[s] < MIN_NON_NEUTRAL}
    testable = sorted((s for s in coverage if s not in skipped), key=lambda s: -coverage[s])
    result = {"sessions": int(len(frame)), "coverage_sessions": coverage,
              "non_neutral_share": non_neutral, "not_tested": skipped,
              "in_sample_weight": weights(frame, testable), "models": {}}
    for kind in kinds:
        base = walk_forward(frame, BASELINE_FEATURES, kind=kind)
        base_scores = evaluate(base)
        auc = base_scores["roc_auc"]
        add_one = {s: versus(frame, columns([s]), base, auc, kind) for s in testable}
        adjusted = holm({s: r["dm_test"]["p_value"] for s, r in add_one.items()
                         if "p_value" in r["dm_test"]})
        for s, p in adjusted.items():
            add_one[s]["dm_test"]["p_value_holm"] = p

        full = walk_forward(frame, BASELINE_FEATURES + columns(testable), kind=kind)
        full_auc = evaluate(full)["roc_auc"]
        leave_one_out = {}
        for s in testable:
            reduced = walk_forward(frame, BASELINE_FEATURES + columns([o for o in testable if o != s]),
                                   kind=kind)
            leave_one_out[s] = {"auc_drop_when_removed": round(full_auc - evaluate(reduced)["roc_auc"], 4),
                                "dm_full_vs_without": diebold_mariano(full, reduced)}
        result["models"][kind] = {
            "baseline": base_scores,
            "all_outlets": {"roc_auc": full_auc, "delta_auc": round(full_auc - auc, 4),
                            "dm_test": diebold_mariano(full, base)},
            "add_one": add_one, "leave_one_out": leave_one_out,
            "ranking_by_delta_auc": sorted(testable, key=lambda s: -add_one[s]["delta_auc"])}
        print(f"  {'ex-price' if ex_price else 'all'} / {kind}: done", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--models", nargs="+", default=["regress", "gbm"])
    args = parser.parse_args()
    results = {"window_start": START, "min_sessions_to_test": MIN_SESSIONS,
               "harness": "baseline.walk_forward min_train=500 refit_every=20 embargo=5",
               "sentiment": "CamemBERT yearly-refit (sent_source == yearly_refit) only",
               "variants": {v: run_variant(v == "ex_price_reports", tuple(args.models))
                            for v in VARIANTS}}
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for variant, r in results["variants"].items():
        print(f"\n=== {variant}  ({r['sessions']:,} sessions from {START}) ===")
        for kind, m in r["models"].items():
            b = m["baseline"]
            print(f"\n[{kind}] price-only baseline AUC {b['roc_auc']:.4f} "
                  f"(n={b['n_predictions']:,})   all outlets AUC {m['all_outlets']['roc_auc']:.4f} "
                  f"(DM p={m['all_outlets']['dm_test'].get('p_value')})")
            print(f"{'outlet':<22}{'sessions':>9}{'weight':>8}{'t':>7}{'dAUC':>8}"
                  f"{'DM p':>8}{'Holm':>8}{'McN p':>8}{'LOO dAUC':>10}")
            for s in m["ranking_by_delta_auc"]:
                a, w, l = m["add_one"][s], r["in_sample_weight"][s], m["leave_one_out"][s]
                print(f"{s:<22}{r['coverage_sessions'][s]:>9,}{w['beta_std']:>8.3f}{w['t_hac']:>7.2f}"
                      f"{a['delta_auc']:>+8.4f}{a['dm_test'].get('p_value', float('nan')):>8.4f}"
                      f"{a['dm_test'].get('p_value_holm', float('nan')):>8.4f}"
                      f"{a['sign_test']['p_value']:>8.4f}{l['auc_drop_when_removed']:>+10.4f}")
        for s, why in r["not_tested"].items():
            print(f"not tested: {s:<22} {why}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
