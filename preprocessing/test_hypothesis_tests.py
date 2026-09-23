"""The harness must detect planted signal AND planted confounds.

A test suite that only checks "it runs" is worthless for an inference tool. Each
test below plants a known ground truth and asserts the harness reaches the right
verdict -- including the wrong-for-the-right-reason case (momentum laundering),
which is the whole purpose of the placebo arm.
"""
import numpy as np
import pandas as pd
import pytest

from hypothesis_tests import (ARMS, missing_columns, paired_test, required_columns,
                              run_h1, run_h2)


def _frame(n=900, seed=0, signal="none"):
    rng = np.random.default_rng(seed)
    # One underlying return series; row i sees r[i] (ret_lag0) and predicts
    # r[i+1] (ret_next). Lags count back from the TARGET, matching features.py.
    # Do NOT make ret_lag0 and ret_next the same array -- that hands the baseline
    # the answer and every arm scores 1.0.
    r = rng.normal(0.0005, 0.0045, n)
    f = pd.DataFrame({
        "session": pd.bdate_range("2014-01-01", periods=n),
        "ret_next": np.roll(r, -1),
    })
    f["ret_lag0"] = r
    for lag in (1, 2, 3):
        f[f"ret_lag{lag}"] = np.roll(r, lag)
    ret = f["ret_next"].to_numpy()
    f["vol_chg_lag0"] = rng.normal(0, 0.7, n)
    dow = f["session"].shift(-1).dt.dayofweek
    for day, name in enumerate(("mon", "tue", "wed", "thu")):
        f[f"dow_next_{name}"] = (dow == day).astype(float)

    noise = lambda: rng.normal(0, 1, n)
    # default: pure noise sentiment in every arm
    cols = {}
    for arm, suffix in ARMS.items():
        for base in ("sent_mean", "sent_pos_share", "sent_neg_share"):
            cols[f"{base}{suffix}_lag1"] = noise()
        cols[f"has_sent{suffix}_lag1"] = np.ones(n)
    cols["sent_resid_lag1"] = noise()
    cols["log_headlines_lag0"] = rng.normal(2.5, 0.4, n)

    if signal == "real":
        # genuine news signal, present in the non-price arms
        edge = np.sign(ret) * rng.uniform(0.4, 1.0, n)
        cols["sent_mean_lag1"] = edge + rng.normal(0, 0.4, n)
        cols["sent_mean_ex_price_lag1"] = edge + rng.normal(0, 0.4, n)
        cols["sent_resid_lag1"] = edge + rng.normal(0, 0.4, n)
    elif signal == "momentum":
        # the confound: sentiment is just yesterday's return, and ONLY the
        # price-report arms carry it. ex_price stays pure noise.
        # The REAL confound path: yesterday's price report describes ret_lag0,
        # which IS in the control set now. Plant it there so the test exercises
        # the actual channel rather than an already-controlled lag.
        laundered = f["ret_lag0"].to_numpy() * 100
        cols["sent_mean_lag1"] = laundered + rng.normal(0, 0.3, n)
        cols["sent_mean_price_only_lag1"] = laundered + rng.normal(0, 0.3, n)

    for k, v in cols.items():
        f[k] = v
    return f.iloc[3:-1].reset_index(drop=True)


def test_refuses_to_run_without_sentiment_columns():
    bare = _frame().drop(columns=[c for arm in ARMS for c in required_columns(arm)])
    with pytest.raises(SystemExit, match="No sentiment columns found"):
        run_h1(bare)


def test_missing_columns_names_every_arm():
    absent = missing_columns(_frame().drop(columns=["sent_mean_ex_price_lag1"]))
    assert absent["ex_price"] == ["sent_mean_ex_price_lag1"]
    assert absent["all"] == [] and absent["placebo"] == []


def test_holm_correction_is_applied_and_never_lowers_a_p_value():
    from hypothesis_tests import holm
    raw = {"a": 0.01, "b": 0.04, "c": 0.20}
    adj = holm(raw)
    assert all(adj[k] >= raw[k] for k in raw)
    assert adj["a"] == 0.03                     # 3 * 0.01
    assert adj["b"] >= adj["a"]                 # monotone non-decreasing


def test_power_flags_an_underpowered_design():
    from hypothesis_tests import minimum_detectable_effect
    weak = minimum_detectable_effect(n_pairs=2678, discordant=853)
    assert weak["mde_pp_at_80_power"] > 2.0
    assert weak["adequately_powered"] is False
    strong = minimum_detectable_effect(n_pairs=50000, discordant=900)
    assert strong["adequately_powered"] is True


def test_has_sent_indicator_is_required_so_zero_fill_is_not_a_silent_lie():
    from hypothesis_tests import required_columns
    for arm in ("all", "ex_price", "placebo"):
        assert any(c.startswith("has_sent") for c in required_columns(arm))


def test_detects_a_genuinely_planted_signal():
    r = run_h1(_frame(signal="real", seed=1), min_train=400, refit_every=25)
    assert r["all"]["accuracy"] > r["baseline"]["accuracy"]
    assert r["ex_price"]["vs_baseline"]["significant_at_05"]
    assert "H1 SUPPORTED" in r["interpretation"]


def test_diagnoses_momentum_laundering_rather_than_calling_it_sentiment():
    """The load-bearing test. Sentiment is lagged returns, carried only by the
    price-report arm. `all` improves; `ex_price` must not -- and the harness must
    say so instead of reporting a sentiment effect."""
    r = run_h1(_frame(signal="momentum", seed=2), min_train=400, refit_every=25)
    assert not r["ex_price"]["vs_baseline"]["significant_at_05"]
    assert "H1 SUPPORTED" not in r["interpretation"]
    assert ("MOMENTUM LAUNDERING" in r["interpretation"]
            or "PLACEBO FIRED" in r["interpretation"]
            or "NOT SUPPORTED" in r["interpretation"])


def test_reports_no_support_when_sentiment_is_noise():
    r = run_h1(_frame(signal="none", seed=3), min_train=400, refit_every=25)
    assert "NOT SUPPORTED" in r["interpretation"]


def test_paired_test_refuses_unaligned_arms():
    a = pd.DataFrame({"session": pd.bdate_range("2024-01-01", periods=5),
                      "y_pred": [1]*5, "y_true": [1]*5})
    b = a.assign(session=pd.bdate_range("2024-02-01", periods=5))
    with pytest.raises(ValueError, match="not aligned"):
        paired_test(a, b)


def test_paired_test_counts_discordant_pairs_only():
    common = pd.bdate_range("2024-01-01", periods=4)
    t = pd.DataFrame({"session": common, "y_true": [1, 1, 0, 0], "y_pred": [1, 1, 1, 0]})
    b = pd.DataFrame({"session": common, "y_true": [1, 1, 0, 0], "y_pred": [0, 1, 0, 0]})
    r = paired_test(t, b)
    assert r["treatment_only_correct"] == 1     # session 0
    assert r["baseline_only_correct"] == 1      # session 2


def test_h2_ranks_sources_by_damage_when_removed():
    f = _frame(signal="real", seed=4)
    rng = np.random.default_rng(9)
    useful = np.sign(f.ret_next) * rng.uniform(0.5, 1.0, len(f))
    f["sent_mean_ilboursa_lag1"] = useful + rng.normal(0, 0.3, len(f))
    f["sent_mean_kapitalis_lag1"] = rng.normal(0, 1, len(f))   # pure noise
    r = run_h2(f, ["ilboursa", "kapitalis"], min_train=400, refit_every=25)
    assert r["most_important"] == "ilboursa"
    assert r["leave_one_out"][0]["accuracy_drop"] >= r["leave_one_out"][-1]["accuracy_drop"]


def test_h2_refuses_with_fewer_than_two_sources():
    with pytest.raises(SystemExit, match="Need per-source"):
        run_h2(_frame(), ["ilboursa"])


def test_h2_restricts_to_the_common_coverage_window():
    """M6: sources with different coverage windows must not be ranked against each
    other on accuracy lost -- that ranks coverage length, not contribution."""
    f = _frame(signal="real", seed=7)
    rng = np.random.default_rng(11)
    f["sent_mean_ilboursa_lag1"] = rng.normal(0, 1, len(f))
    f["sent_mean_lapresse_lag1"] = rng.normal(0, 1, len(f))
    f.loc[: len(f) - 400, "sent_mean_lapresse_lag1"] = 0.0   # late-entering source
    r = run_h2(f, ["ilboursa", "lapresse"], min_train=100, refit_every=25)
    assert "common_window" in r
    # the window must start where the late source becomes active, not at the top
    assert r["sessions_in_window"] < len(f)
    assert r["caveat"].startswith("accuracy_drop has no standard error")


def test_h2_refuses_when_coverage_windows_barely_overlap():
    """Better to refuse than to silently rank coverage length."""
    f = _frame(signal="real", seed=8)
    rng = np.random.default_rng(12)
    f["sent_mean_ilboursa_lag1"] = rng.normal(0, 1, len(f))
    f["sent_mean_lapresse_lag1"] = rng.normal(0, 1, len(f))
    f.loc[: len(f) - 50, "sent_mean_lapresse_lag1"] = 0.0    # almost no overlap
    with pytest.raises(SystemExit, match="coverage windows overlap"):
        run_h2(f, ["ilboursa", "lapresse"], min_train=100, refit_every=25)


def _paired_forecasts(n=400, treat_better=True, seed=0):
    rng = np.random.default_rng(seed)
    actual = rng.normal(0, 0.005, n)
    good = actual * 0.5 + rng.normal(0, 0.002, n)
    bad = actual * 0.5 + rng.normal(0, 0.006, n)
    sess = pd.bdate_range("2014-01-01", periods=n)
    t_score, b_score = (good, bad) if treat_better else (bad, good)
    return (pd.DataFrame({"session": sess, "score": t_score, "ret_next": actual}),
            pd.DataFrame({"session": sess, "score": b_score, "ret_next": actual}))


def test_dm_detects_a_better_forecast():
    from hypothesis_tests import diebold_mariano
    t, b = _paired_forecasts(treat_better=True)
    r = diebold_mariano(t, b)
    assert r["significant_at_05"] and r["treatment_better"]
    assert r["dm_statistic"] < 0                    # lower loss => negative stat
    assert r["oos_r2_treatment"] > r["oos_r2_baseline"]


def test_dm_detects_a_worse_forecast_the_sign_test_would_miss():
    """The whole reason DM is here: the sign test needs ~3pp of accuracy to see
    anything, so it reports 'ns' while the forecast is measurably degrading."""
    from hypothesis_tests import diebold_mariano
    t, b = _paired_forecasts(treat_better=False)
    r = diebold_mariano(t, b)
    assert r["significant_at_05"] and r["treatment_better"] is False
    assert r["dm_statistic"] > 0
    assert r["oos_r2_treatment"] < r["oos_r2_baseline"]


def test_dm_refuses_unaligned_arms_and_tiny_samples():
    from hypothesis_tests import diebold_mariano
    t, b = _paired_forecasts(n=20)
    assert "skipped" in diebold_mariano(t, b)
    t, b = _paired_forecasts()
    b = b.assign(session=pd.bdate_range("2020-01-01", periods=len(b)))
    with pytest.raises(ValueError, match="not aligned"):
        diebold_mariano(t, b)


def test_interpretation_reports_degradation_not_a_bare_null():
    """A significant DM in the harmful direction must not be summarised as
    'H1 NOT SUPPORTED' -- that hides a real finding behind an underpowered null."""
    from hypothesis_tests import ARMS, _interpret
    worse = {"significant_at_05": True, "treatment_better": False}
    results = {"baseline": {"accuracy": 0.57, "n_predictions": 2678}}
    for arm in ARMS:
        results[arm] = {"accuracy": 0.57, "vs_baseline": {"significant_at_05": False},
                        "vs_baseline_continuous": worse}
    verdict = _interpret(results)
    assert "REJECTED" in verdict and "WORSE" in verdict
