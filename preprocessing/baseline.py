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
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss,
                             matthews_corrcoef, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data" / "curated"
DEFAULT_INPUT = CURATED / "daily_features.parquet"
DEFAULT_OUTPUT = CURATED / "baseline_results.json"

# Blueprint 5.2 F0, completed by amendment prereg-h1-v1-a1: lagged returns plus
# volume change and the target session's day of week. H1's baseline arm.
F0_FEATURES = ["ret_lag0", "ret_lag1", "ret_lag2", "vol_chg_lag0",
               "dow_next_mon", "dow_next_tue", "dow_next_wed", "dow_next_thu"]

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
    "F0":              F0_FEATURES,
}


# Blueprint section 6.3: "5-session embargo between train and test". With return
# autocorrelation at +0.263 and |ret| at +0.387, the sessions immediately before a
# prediction are the ones most correlated with it, so training on them is the
# closest thing to leakage this design can still contain.
EMBARGO_SESSIONS = 5


def make_model(kind: str):
    if kind == "classify":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    # Regress the RETURN and take the sign. Classifying the direction throws away
    # magnitude, and the binary label is dominated by the upward drift, so the
    # classifier collapses toward "always up".
    return make_pipeline(StandardScaler(), LinearRegression())


def walk_forward(frame: pd.DataFrame, features: list[str],
                 min_train: int = 500, refit_every: int = 20,
                 kind: str = "classify",
                 embargo: int = EMBARGO_SESSIONS) -> pd.DataFrame:
    """Expanding-window walk-forward with an embargo.

    Refits every `refit_every` sessions; between refits the last fitted model
    predicts, which is what a deployed model does.

    `embargo` drops the last `embargo` sessions before the prediction point from
    the training slice (blueprint 6.3). Without it the model trains on the very
    sessions whose returns are most correlated with the target. Set embargo=0
    only to quantify what the embargo costs."""
    data = frame.dropna(subset=features + ["ret_next"]).reset_index(drop=True)
    y = (data["ret_next"] > 0).astype(int).to_numpy()
    target = data["ret_next"].to_numpy()
    X = data[features].to_numpy()

    rows, model, train_majority = [], None, None
    for i in range(min_train, len(data)):
        cut = max(1, i - embargo)          # strictly-before, minus the embargo
        if model is None or (i - min_train) % refit_every == 0:
            # strictly-before slice: nothing at or after i is visible
            model = make_model(kind)
            model.fit(X[:cut], y[:cut] if kind == "classify" else target[:cut])
            train_majority = int(y[:cut].mean() >= 0.5)
        if kind == "classify":
            # predict_proba, not predict: the class label carries no ranking
            # information, so ROC-AUC computed on it would be nonsense.
            score = float(model.predict_proba(X[i:i + 1])[0, 1])
            pred = int(score > 0.5)
        else:
            score = float(model.predict(X[i:i + 1])[0])
            pred = int(score > 0)
        rows.append({
            "session": data["session"].iloc[i],
            "y_true": int(y[i]),
            "y_pred": pred,
            "score": score,
            "y_constant": train_majority,
            "ret_next": float(data["ret_next"].iloc[i]),
        })
    return pd.DataFrame(rows)


def evaluate(preds: pd.DataFrame) -> dict:
    """Blueprint section 6.2: ROC-AUC is primary; balanced accuracy and MCC REPLACE
    raw accuracy as the headline, because at a 54.9% base rate accuracy is nearly
    blind. Raw accuracy is still reported, for comparability with the reference
    study and because the constant-baseline comparison is stated in those units."""
    acc = float((preds.y_pred == preds.y_true).mean())
    const = float((preds.y_constant == preds.y_true).mean())
    # Binomial SE of the model's accuracy -- an honest +/- on the headline number
    se = float(np.sqrt(acc * (1 - acc) / len(preds)))
    y, s = preds.y_true.to_numpy(), preds.score.to_numpy()
    auc = float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")
    # Brier needs a probability. A regression score is a return, not one, so rank-
    # normalise to [0,1] rather than pretending the raw value is calibrated.
    prob = (pd.Series(s).rank(pct=True).to_numpy() if s.min() < 0 or s.max() > 1
            else s)
    return {
        "n_predictions": int(len(preds)),
        "roc_auc": round(auc, 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, preds.y_pred)), 4),
        "mcc": round(float(matthews_corrcoef(y, preds.y_pred)), 4),
        "brier": round(float(brier_score_loss(y, prob)), 4),
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
    print(f"{'feature set':<24}{'n':>6}{'AUC':>8}{'balAcc':>8}{'MCC':>8}"
          f"{'Brier':>8}{'acc':>8}{'const':>8}")
    print("-" * 78)
    for name, s in results.items():
        print(f"{name:<24}{s['n_predictions']:>6}{s['roc_auc']:>8.4f}"
              f"{s['balanced_accuracy']:>8.4f}{s['mcc']:>8.4f}{s['brier']:>8.4f}"
              f"{s['accuracy']:>8.4f}{s['constant_baseline']:>8.4f}")
    print("\nBlueprint 6.2: AUC is primary; balanced accuracy and MCC replace raw "
          "accuracy as headline.")


if __name__ == "__main__":
    main()
