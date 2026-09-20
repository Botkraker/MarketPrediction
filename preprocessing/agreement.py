"""Inter-annotator agreement across the annotator_N_label columns.

This is the statistic that makes an LLM-annotated gold set defensible: without
it, "adjudicated_label" is one model's opinion with no reliability estimate
attached. Run it after two or more annotators have labelled the gold set.

Labels are ORDINAL (very_negative < ... < very_positive), so pairwise agreement
is reported with quadratic-weighted Cohen's kappa -- confusing very_negative
with positive is a worse error than confusing negative with neutral, and an
unweighted statistic cannot see that. Fleiss' kappa is also reported because it
is the conventional multi-rater number, but it is nominal: it treats every
disagreement as equally bad. Report both and say which is which.

Krippendorff's alpha is not computed -- it would need a dependency the project
does not have. Weighted Cohen + Fleiss covers the same ground for a fixed set
of raters labelling every item.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

from gold import LABELS

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data" / "curated"
DEFAULT_INPUT = CURATED / "sentiment_gold_annotation1.csv"
DEFAULT_OUTPUT = CURATED / "agreement.json"

# How a kappa value is conventionally described (Landis & Koch 1977). Reported
# as a label only -- the number is what goes in the paper.
def interpret(kappa: float) -> str:
    for threshold, word in ((0.81, "almost perfect"), (0.61, "substantial"),
                            (0.41, "moderate"), (0.21, "fair"), (0.0, "slight")):
        if kappa >= threshold:
            return word
    return "poor (worse than chance)"


def annotator_label_columns(frame: pd.DataFrame) -> list[str]:
    """Every annotator_N_label column that is actually populated, in N order."""
    found = []
    for column in frame.columns:
        if column.startswith("annotator_") and column.endswith("_label"):
            if frame[column].astype(str).str.strip().ne("").any():
                found.append(column)
    return sorted(found, key=lambda c: int(c.split("_")[1]))


def majority_label(row: pd.Series, columns: list[str]) -> tuple[str, bool]:
    """(winning label, tied) by plurality vote. A tie returns the empty string
    so it can never be silently resolved -- ties are for a human or a tiebreak
    model to settle, not for argmax ordering to decide."""
    votes = [str(row[c]).strip() for c in columns if str(row[c]).strip()]
    if not votes:
        return "", False
    counts = pd.Series(votes).value_counts()
    top = counts.max()
    winners = counts[counts == top].index.tolist()
    if len(winners) > 1:
        return "", True
    return winners[0], False


def compute(input_path: Path = DEFAULT_INPUT,
            output_path: Path = DEFAULT_OUTPUT) -> dict:
    frame = pd.read_csv(input_path, keep_default_na=False)
    if "in_study" in frame.columns:
        keep = frame["in_study"].astype(str).str.lower().isin(("true", "1"))
        frame = frame[keep].reset_index(drop=True)

    columns = annotator_label_columns(frame)
    if len(columns) < 2:
        raise SystemExit(
            f"Need >=2 populated annotator_N_label columns, found {len(columns)}: "
            f"{columns or 'none'}.\nRun a second annotator first, e.g.\n"
            "  python preprocessing/llm_annotate.py --annotator 2 --model <other-model>"
        )

    # Only rows every annotator actually labelled can enter an agreement stat.
    labelled = frame[columns].apply(lambda s: s.astype(str).str.strip().ne(""))
    complete = frame[labelled.all(axis=1)].reset_index(drop=True)
    if complete.empty:
        raise SystemExit("No row has a label from every annotator.")

    pairwise = []
    for a, b in combinations(columns, 2):
        linear = cohen_kappa_score(complete[a], complete[b], labels=LABELS,
                                   weights="linear")
        quadratic = cohen_kappa_score(complete[a], complete[b], labels=LABELS,
                                      weights="quadratic")
        nominal = cohen_kappa_score(complete[a], complete[b], labels=LABELS)
        pairwise.append({
            "pair": f"{a} vs {b}",
            "raw_agreement": round(float((complete[a] == complete[b]).mean()), 4),
            "cohen_kappa_nominal": round(float(nominal), 4),
            "cohen_kappa_linear": round(float(linear), 4),
            "cohen_kappa_quadratic": round(float(quadratic), 4),
            "interpretation_quadratic": interpret(float(quadratic)),
        })

    table, _ = aggregate_raters(complete[columns].to_numpy())
    fleiss = float(fleiss_kappa(table, method="fleiss"))

    majority = complete.apply(lambda r: majority_label(r, columns), axis=1)
    unanimous = int(complete[columns].nunique(axis=1).eq(1).sum())

    result = {
        "input": str(input_path.name),
        "annotators": columns,
        "models": {c: sorted(set(frame.get(c.replace("_label", "_model"), pd.Series(dtype=str))) - {""})
                   for c in columns},
        "rows_total": int(len(frame)),
        "rows_all_annotators_labelled": int(len(complete)),
        "fleiss_kappa_nominal": round(fleiss, 4),
        "fleiss_interpretation": interpret(fleiss),
        "unanimous_rows": unanimous,
        "unanimous_share": round(unanimous / len(complete), 4) if len(complete) else None,
        "tied_rows": int(sum(t for _, t in majority)),
        "pairwise": pairwise,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = compute(args.input, args.output)
    print(f"annotators: {', '.join(result['annotators'])}")
    print(f"rows compared: {result['rows_all_annotators_labelled']}")
    print(f"Fleiss kappa (nominal): {result['fleiss_kappa_nominal']} "
          f"({result['fleiss_interpretation']})")
    for pair in result["pairwise"]:
        print(f"  {pair['pair']}: raw={pair['raw_agreement']} "
              f"quadratic kappa={pair['cohen_kappa_quadratic']} "
              f"({pair['interpretation_quadratic']})")
    print(f"unanimous: {result['unanimous_share']} | ties: {result['tied_rows']}")


if __name__ == "__main__":
    main()
