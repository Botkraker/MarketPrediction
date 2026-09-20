"""Price-only walk-forward baseline: the bar that H1's sentiment model must clear.

WHY WALK-FORWARD AND NOT CROSS-VALIDATION
-----------------------------------------
k-fold CV shuffles time. A model trained on 2025 and tested on 2015 has seen the
future, and on a series with 0.263 first-order autocorrelation that leak is
enormous. Every fit here uses ONLY sessions strictly before the one it predicts.

WHY THE CONSTANT BASELINE IS NOT "50%"
--------------------------------------
Tunindex rises on 54.2% of sessions (+12.5%/yr drift). "Always predict up" is
therefore already 54.2% accurate while knowing nothing. Any accuracy below that
is worse than a model that cannot read. The constant baseline here is refit
walk-forward too -- it predicts the majority class OF THE TRAINING WINDOW, not
of the whole sample, since using the full-sample majority would itself leak.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data" / "curated"
DEFAULT_INPUT = CURATED / "daily_features.parquet"
DEFAULT_OUTPUT = CURATED / "baseline_results.json"

FEATURE_SETS = {
    # ret_lag0 is the last CLOSED session's return -- available at prediction
    # time and the strongest single predictor. Omitting it (an earlier bug)
    # costs 1.7pp of accuracy and turns a significant result into a null.
    "momentum":        ["ret_lag0"],
    "momentum3":       ["ret_lag0", "ret_lag1", "ret_lag2"],
    "momentum3_vol":   ["ret_lag0", "ret_lag1", "ret_lag2",
                        "abs_ret_lag0", "abs_ret_lag1", "abs_ret_lag2"],
    # news VOLUME only -- still price-only in spirit (no sentiment), included to
    # show whether counting headlines adds anything before we ever score them.
    "momentum3_news":  ["ret_lag0", "ret_lag1", "ret_lag2", "log_headlines_lag0"],
}


def walk_forward(frame: pd.DataFrame, features: list[str],
                 min_train: int = 500, refit_every: int = 20,
                 kind: str = "classify") -> pd.DataFrame:
    """Expanding-window walk-forward. Refits every `refit_every` sessions; between
    refits the last fitted model predicts, which is what a deployed model does."""
    data = frame.dropna(subset=features + ["ret_next"]).reset_index(drop=True)
    y = (data["ret_next"] > 0).astype(int).to_numpy()
    target = data["ret_next"].to_numpy()
    X = data[features].to_numpy()

    rows, model, train_majority = [], None, None
    for i in range(min_train, len(data)):
        if model is None or (i - min_train) % refit_every == 0:
            # strictly-before slice: nothing at or after i is visible
            if kind == "classify":
                model = make_pipeline(StandardScaler(),
                                      LogisticRegression(max_iter=1000, C=1.0))
                model.fit(X[:i], y[:i])
            else:
                # Regress the RETURN and take the sign. Classifying the direction
                # throws away magnitude, and the binary label is dominated by the
                # upward drift, so the classifier collapses toward "always up".
                model = make_pipeline(StandardScaler(), LinearRegression())
                model.fit(X[:i], target[:i])
            train_majority = int(y[:i].mean() >= 0.5)
        raw = float(model.predict(X[i:i + 1])[0])
        rows.append({
            "session": data["session"].iloc[i],
            "y_true": int(y[i]),
            "y_pred": int(raw > 0.5) if kind == "classify" else int(raw > 0),
            "score": raw,
            "y_constant": train_majority,
            "ret_next": float(data["ret_next"].iloc[i]),
        })
    return pd.DataFrame(rows)


def evaluate(preds: pd.DataFrame) -> dict:
    acc = float((preds.y_pred == preds.y_true).mean())
    const = float((preds.y_constant == preds.y_true).mean())
    # Binomial SE of the model's accuracy -- an honest +/- on the headline number
    se = float(np.sqrt(acc * (1 - acc) / len(preds)))
    return {
        "n_predictions": int(len(preds)),
        "accuracy": round(acc, 4),
        "accuracy_se": round(se, 4),
        "constant_baseline": round(const, 4),
        "lift_over_constant": round(acc - const, 4),
        "beats_constant": bool(acc > const),
        "share_predicted_up": round(float(preds.y_pred.mean()), 4),
    }


def run(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT,
        min_train: int = 500, refit_every: int = 20) -> dict:
    frame = pd.read_parquet(input_path)
    results = {}
    for name, features in FEATURE_SETS.items():
        for kind in ("classify", "regress"):
          preds = walk_forward(frame, features, min_train, refit_every, kind)
          scores = evaluate(preds)
          scores["features"] = features
          scores["kind"] = kind
          scores["by_year"] = {
            str(year): round(float((g.y_pred == g.y_true).mean()), 4)
            for year, g in preds.groupby(preds.session.dt.year)}
          results[f"{name}:{kind}"] = scores
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-train", type=int, default=500)
    parser.add_argument("--refit-every", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    results = run(args.output.parent / "daily_features.parquet" if False else DEFAULT_INPUT,
                  args.output, args.min_train, args.refit_every)
    print(f"{'feature set':<18}{'n':>6}{'acc':>9}{'+/-':>7}{'const':>9}{'lift':>8}")
    print("-" * 57)
    for name, s in results.items():
        print(f"{name:<18}{s['n_predictions']:>6}{s['accuracy']:>9.4f}"
              f"{s['accuracy_se']:>7.4f}{s['constant_baseline']:>9.4f}"
              f"{s['lift_over_constant']:>+8.4f}")


if __name__ == "__main__":
    main()
