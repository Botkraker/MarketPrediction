"""Sensitivity of the H1 result to the walk-forward's free parameters.

WHY (audit finding M1). `min_train=500` and `refit_every=20` were never justified.
They are not innocuous: min_train sets where the test window starts and therefore n,
and refit_every sets how stale the model is allowed to get between fits. A result
that only holds at one arbitrary setting is not a result, and a reviewer who tries
another setting and gets a different answer will assume the worst.

This grids both and reports whether the verdict is stable. It answers "does the
conclusion depend on a knob nobody chose deliberately", not "which setting is best"
-- picking the best would be the same outcome-selection problem that M1 flags for
`kind="regress"`.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from baseline import evaluate, walk_forward
from hypothesis_tests import BASELINE_FEATURES, diebold_mariano, paired_test

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data" / "curated"
DEFAULT_INPUT = CURATED / "daily_features.parquet"
DEFAULT_OUTPUT = CURATED / "sensitivity_results.json"

MIN_TRAIN_GRID = (300, 500, 750, 1000)
REFIT_GRID = (1, 5, 20, 60)


def run(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT,
        kind: str = "regress") -> dict:
    frame = pd.read_parquet(input_path)
    rows = []
    for min_train in MIN_TRAIN_GRID:
        for refit_every in REFIT_GRID:
            preds = walk_forward(frame, BASELINE_FEATURES, min_train, refit_every, kind)
            scores = evaluate(preds)
            const = pd.DataFrame({
                "session": preds.session, "score": 0.0,
                "ret_next": preds.ret_next,
                "y_pred": preds.y_constant, "y_true": preds.y_true})
            rows.append({
                "min_train": min_train,
                "refit_every": refit_every,
                "n": scores["n_predictions"],
                "accuracy": scores["accuracy"],
                "constant": scores["constant_baseline"],
                "lift": scores["lift_over_constant"],
                "mcnemar_p": paired_test(preds, const)["p_value"],
                "oos_r2": diebold_mariano(preds, const).get("oos_r2_treatment"),
            })
    table = pd.DataFrame(rows)
    verdict = {
        "beats_constant_everywhere": bool((table.lift > 0).all()),
        "significant_at_05_count": int((table.mcnemar_p < 0.05).sum()),
        "of_configurations": int(len(table)),
        "accuracy_range": [float(table.accuracy.min()), float(table.accuracy.max())],
        "oos_r2_range": [float(table.oos_r2.min()), float(table.oos_r2.max())],
        "stable": bool((table.lift > 0).all()
                       and (table.mcnemar_p < 0.05).mean() >= 0.75),
    }
    result = {"grid": rows, "verdict": verdict,
              "note": ("Exploratory. These configurations are NOT Holm-corrected and "
                       "must not be mined for the best setting -- that would repeat the "
                       "outcome-selection problem M1 raises for kind='regress'.")}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    r = run(args.input, args.output)
    print(f"{'min_train':>10}{'refit':>7}{'n':>7}{'acc':>9}{'const':>9}"
          f"{'lift':>9}{'McNemar':>10}{'OOS R2':>10}")
    print("-" * 71)
    for row in r["grid"]:
        flag = "*" if row["mcnemar_p"] < 0.05 else " "
        print(f"{row['min_train']:>10}{row['refit_every']:>7}{row['n']:>7}"
              f"{row['accuracy']:>9.4f}{row['constant']:>9.4f}{row['lift']:>+9.4f}"
              f"{row['mcnemar_p']:>9.4f}{flag}{row['oos_r2']:>10.5f}")
    v = r["verdict"]
    print(f"\nbeats constant in every configuration : {v['beats_constant_everywhere']}")
    print(f"significant in {v['significant_at_05_count']}/{v['of_configurations']} "
          f"configurations")
    print(f"accuracy range {v['accuracy_range'][0]:.4f}-{v['accuracy_range'][1]:.4f}"
          f"  |  OOS R2 range {v['oos_r2_range'][0]:.5f}-{v['oos_r2_range'][1]:.5f}")
    print(f"VERDICT STABLE: {v['stable']}")


if __name__ == "__main__":
    main()
