"""
Validation for the relevance filter (audit finding S6): the keyword lists in
config.py are self-declared "FIRST-PASS ... not yet validated against hand
labels", and they drop ~45% of the cleaned corpus. This script does the two
things that were missing:

  report     -- recompute the tag for every row of 02_relevance.parquet with
                the CURRENT code and compare against the tag stored in the
                parquet (which was produced before the country-negation fix).
                Prints the measured wrong-country rate before/after, how many
                rows change tag, and the residual that negation cannot fix.
                If hand labels exist, also prints precision/recall per tag.
  worksheet  -- emit a stratified hand-labelling sheet (CSV) so precision and
                recall can actually be computed instead of asserted.

Usage:
    python3 preprocessing/relevance_validation.py report
    python3 preprocessing/relevance_validation.py worksheet [-n 300] [-o PATH]

Hand labels: fill the `true_tag_fill_by_hand` column of the worksheet with one
of tunisia_econ / global_linked / other, save, then re-run `report --labels
<path>`. audit/language_relevance_sample.csv is read too if its
`about_tunisian_economy` / `about_global_economy` columns are filled in.
"""
import argparse
from pathlib import Path

import pandas as pd

from relevance import tag_row
from relevance_geo import is_foreign_context, mentions_foreign, mentions_tunisia

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data" / "curated"
AUDIT = ROOT / "audit"
TAGS = ["tunisia_econ", "global_linked", "other"]


def load() -> pd.DataFrame:
    df = pd.read_parquet(CURATED / "02_relevance.parquet")
    df["new_tag"] = [tag_row(h, l)[0] for h, l in zip(df["headline_clean"], df["lang"])]
    return df


def _hand_labels(path: Path | None) -> pd.DataFrame:
    """row_id -> true_tag, from the worksheet and/or the audit sample."""
    frames = []
    if path and path.exists():
        w = pd.read_csv(path)
        col = "true_tag_fill_by_hand"
        if col in w:
            w = w[w[col].isin(TAGS)]
            frames.append(w[["row_id", col]].rename(columns={col: "true_tag"}))
    a = AUDIT / "language_relevance_sample.csv"
    if a.exists():
        s = pd.read_csv(a)
        tun = [c for c in s.columns if c.startswith("about_tunisian_economy")]
        glo = [c for c in s.columns if c.startswith("about_global_economy")]
        if tun and glo:
            t = s[tun[0]].astype(str).str.strip().str.upper()
            g = s[glo[0]].astype(str).str.strip().str.upper()
            lab = pd.DataFrame({
                "row_id": s["row_id"],
                "true_tag": ["tunisia_econ" if a_ == "Y" else "global_linked" if b == "Y" else "other"
                             for a_, b in zip(t, g)],
                "filled": (t.isin(["Y", "N"]) & g.isin(["Y", "N"])),
            })
            frames.append(lab[lab["filled"]][["row_id", "true_tag"]])
    if not frames:
        return pd.DataFrame(columns=["row_id", "true_tag"])
    return pd.concat(frames).drop_duplicates("row_id")


def _prf(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for t in TAGS:
        tp = int(((df[pred_col] == t) & (df["true_tag"] == t)).sum())
        fp = int(((df[pred_col] == t) & (df["true_tag"] != t)).sum())
        fn = int(((df[pred_col] != t) & (df["true_tag"] == t)).sum())
        rows.append({
            "tag": t, "n_true": tp + fn, "n_pred": tp + fp,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        })
    return pd.DataFrame(rows)


def report(labels: Path | None) -> None:
    df = load()
    print(f"02_relevance.parquet: {len(df)} rows\n")
    print("tag counts, stored (pre-fix) vs recomputed (post-fix):")
    print(pd.DataFrame({
        "before": df["relevance_tag"].value_counts(),
        "after": df["new_tag"].value_counts(),
    }).fillna(0).astype(int), "\n")
    changed = df[df["relevance_tag"] != df["new_tag"]]
    print(f"rows changing tag: {len(changed)}")
    print(changed.groupby(["relevance_tag", "new_tag"]).size().to_string(), "\n")

    # Measured wrong-country rate: headline names a foreign country/city and
    # no Tunisian entity, yet is tagged tunisia_econ.
    for col, name in [("relevance_tag", "before"), ("new_tag", "after")]:
        te = df[df[col] == "tunisia_econ"]
        bad = te.apply(lambda r: is_foreign_context(r["headline_clean"], r["lang"]), axis=1)
        print(f"wrong-country rows inside tunisia_econ ({name}): "
              f"{int(bad.sum())} / {len(te)} = {100 * bad.mean():.2f}%")
    print()

    # Residual that country-negation cannot see: matched on a generic keyword
    # and names NO place at all ("L'inflation ralentit en octobre").
    te = df[df["new_tag"] == "tunisia_econ"].copy()
    kw_generic = ~te.apply(lambda r: mentions_tunisia(str(r["matched_keyword"]), r["lang"]), axis=1)
    no_place = ~te.apply(lambda r: mentions_tunisia(r["headline_clean"], r["lang"])
                         or mentions_foreign(r["headline_clean"], r["lang"]), axis=1)
    resid = te[kw_generic & no_place]
    print(f"UNVALIDATED residual: {len(resid)} / {len(te)} = {100 * len(resid) / len(te):.2f}% "
          "of tunisia_econ matched a generic keyword and name no country at all.")
    print("   (geography cannot decide these -- they need hand labels)")
    print(resid["headline_clean"].head(5).to_string(), "\n")

    lab = _hand_labels(labels)
    ev = df.merge(lab, on="row_id")
    if ev.empty:
        print("hand labels found: 0 -- precision/recall NOT computable. "
              "Run `worksheet`, fill true_tag_fill_by_hand, re-run with --labels.")
        return
    print(f"hand labels found: {len(ev)}")
    print("BEFORE:\n", _prf(ev, "relevance_tag").to_string(index=False))
    print("AFTER:\n", _prf(ev, "new_tag").to_string(index=False))


def worksheet(n: int, out: Path) -> None:
    df = load()
    df["changed"] = df["relevance_tag"] != df["new_tag"]
    # Stratify on (lang, stored tag) plus a deliberate over-sample of the rows
    # the fix changes -- those are where the disagreement, and the risk of a
    # newly-introduced false negative, actually live.
    n_changed = min(n // 3, int(df["changed"].sum()))
    part = [df[df["changed"]].sample(n_changed, random_state=0)]
    rest = df[~df["changed"]]
    groups = rest.groupby(["lang", "relevance_tag"])
    per = max(1, (n - n_changed) // max(1, groups.ngroups))
    part += [g.sample(min(per, len(g)), random_state=0) for _, g in groups]
    s = pd.concat(part).drop_duplicates("row_id")
    s = s[["row_id", "source", "lang", "headline_clean", "matched_keyword",
           "relevance_tag", "new_tag"]].rename(
        columns={"relevance_tag": "tag_before_fix", "new_tag": "tag_after_fix"})
    s["true_tag_fill_by_hand"] = ""   # tunisia_econ | global_linked | other
    s["notes_fill_by_hand"] = ""
    out.parent.mkdir(parents=True, exist_ok=True)
    s.sample(frac=1, random_state=1).to_csv(out, index=False)  # shuffle: don't show the label order
    print(f"Wrote {out} ({len(s)} rows, {n_changed} of them rows the fix changes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--labels", type=Path, default=AUDIT / "relevance_validation_worksheet.csv")
    w = sub.add_parser("worksheet")
    w.add_argument("-n", type=int, default=300)
    w.add_argument("-o", type=Path, default=AUDIT / "relevance_validation_worksheet.csv")
    a = ap.parse_args()
    if a.cmd == "report":
        report(a.labels)
    else:
        worksheet(a.n, a.o)
