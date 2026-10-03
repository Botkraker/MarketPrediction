"""One row per headline: English text, outlet, CamemBERT sentiment, Tunindex values.

    python3 preprocessing/build_headline_series.py
    -> data/curated/tunindex_headline_series.csv

Long format: a session with 12 headlines appears on 12 rows, each repeating that
session's close / ret / ret_next. Every piece is joined on row_id:

  headline    English. French headlines are machine-translated by
              translate_headlines.py; `lang` keeps the ORIGINAL language.
  sent_*      CamemBERT, from score_corpus.py --model camembert --mode expanding.
              Scored on the French original, not on the translation. Each year
              is scored by a model fitted only on gold labels dated before that
              year (sent_model_train_end), so those rows carry no look-ahead.
  sent_source WHICH model produced sent_label on that row:
                yearly_refit  the leakage-free expanding run above.
                saved_model   the FILL. The expanding run cannot score 2014-15
                              (too few prior labels) and its 2016-18 models are
                              under-trained (298-582 rows; the 2016 one labels
                              every headline neutral). Rows whose yearly model
                              predates --fill-before (default 2019) take the
                              saved model's label instead
                              (camembert_clf.py --score-corpus). That model saw
                              gold labels from every year up to 2026, so these
                              rows DO carry look-ahead.
              Any backtest or H1 claim must use sent_source == "yearly_refit"
              rows only. `--fill-before none` builds the file without the fill.

ALIGNMENT is the pipeline's, not a new one. A headline dated D belongs to the
first session STRICTLY after D (features.map_to_next_session, side="right"), so a
headline sits on the same row as ret_next of its session, exactly like
n_headlines and sent_*_lag1 in tunindex_timeseries.csv. Never shift by row: rows
are headlines, not sessions, and the alignment is already done.

Prices are taken from tunindex_timeseries.csv rather than recomputed, and build()
asserts that the number of headlines per session equals that frame's
n_headlines, so the two files cannot drift apart silently.

LANGUAGE is the corpus `lang` column, which is assigned per OUTLET
(config.SOURCE_LANG), not detected per headline. That is deliberate: the audit's
langdetect pass agrees with the outlet language on 93-100% of headlines, and in a
3,000-row check every "en" hit inside a French outlet was a French headline with
an English brand name (Samsung FastTrack, Temenos, Hult Prize). Per-headline
detection would only add errors. There is no Arabic (0 rows -- AUDIT_REPORT S5).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import features

CURATED = features.CURATED
DEFAULT_FRAME = CURATED / "tunindex_timeseries.csv"
DEFAULT_TRANSLATED = CURATED / "headlines_en.parquet"
DEFAULT_SCORED = CURATED / "04_scored_v2_camembert_repro.parquet"
DEFAULT_STATIC = CURATED / "04_scored_v2_camembert_static.parquet"
DEFAULT_OUTPUT = CURATED / "tunindex_headline_series.csv"


def build(frame_path: Path = DEFAULT_FRAME,
          headlines_path: Path = features.DEFAULT_HEADLINES,
          calendar_path: Path = features.DEFAULT_CALENDAR,
          translated_path: Path = DEFAULT_TRANSLATED,
          scored_path: Path = DEFAULT_SCORED,
          static_path: Path = DEFAULT_STATIC,
          fill_before: str | None = "2019-01-01",
          output_path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    frame = pd.read_csv(frame_path, parse_dates=["session"])

    news = pd.read_parquet(headlines_path, columns=[
        "row_id", "source", "lang", "published_date",
        "is_canonical", "relevance_tag", "date_parse_ok"])
    # Same filter as features.build, so the counts match n_headlines.
    news = news[news["is_canonical"] & (news["relevance_tag"] != "other")
                & news["date_parse_ok"]].copy()
    news["day"] = pd.to_datetime(news["published_date"], errors="coerce").dt.normalize()
    news = news.dropna(subset=["day"])
    sessions = pd.DatetimeIndex(
        pd.read_csv(calendar_path, parse_dates=["session_date"])["session_date"]).sort_values()
    news["session"] = features.map_to_next_session(news["day"], sessions)

    english = pd.read_parquet(translated_path, columns=["row_id", "headline_en"])
    scored = pd.read_parquet(scored_path, columns=[
        "row_id", "sent_label", "sent_score", "sent_model_train_end"])
    news = (news.merge(english.rename(columns={"headline_en": "headline"}),
                       on="row_id", how="left", validate="one_to_one")
                .merge(scored, on="row_id", how="left", validate="one_to_one"))

    news[["sent_label", "sent_model_train_end"]] = news[
        ["sent_label", "sent_model_train_end"]].replace("", pd.NA)
    news["sent_source"] = pd.Series(pd.NA, index=news.index, dtype="object").mask(
        news["sent_label"].notna(), "yearly_refit")
    if fill_before:
        static = pd.read_parquet(static_path, columns=["row_id", "sent_label", "sent_score"])
        news = news.merge(static, on="row_id", how="left", validate="one_to_one",
                          suffixes=("", "_static"))
        assert news["sent_label_static"].notna().all(), "static scores do not cover the corpus"
        fill = ~(pd.to_datetime(news["sent_model_train_end"]) >= pd.Timestamp(fill_before))
        news.loc[fill, "sent_label"] = news.loc[fill, "sent_label_static"]
        news.loc[fill, "sent_score"] = news.loc[fill, "sent_score_static"]
        news.loc[fill, "sent_source"] = "saved_model"
        news.loc[fill, "sent_model_train_end"] = pd.NA

    out = news.merge(frame[["session", "close", "ret", "ret_next"]], on="session", how="inner")
    out = out[["session", "published_date", "row_id", "source", "lang", "headline",
               "sent_label", "sent_score", "sent_source", "sent_model_train_end",
               "close", "ret", "ret_next"]
              ].sort_values(["session", "published_date", "source", "row_id"]).reset_index(drop=True)

    per_session = out.groupby("session").size().reindex(frame["session"], fill_value=0)
    assert (per_session.to_numpy() == frame["n_headlines"].to_numpy()).all(), (
        f"{int((per_session.to_numpy() != frame['n_headlines'].to_numpy()).sum())} sessions "
        f"disagree with {frame_path.name} n_headlines -- alignment or filter drifted")
    assert out["headline"].notna().all(), (
        f"{int(out['headline'].isna().sum())} headlines have no English text -- "
        "run translate_headlines.py first")
    has = out["sent_label"].notna()
    assert out.loc[has, "sent_score"].notna().all() and out.loc[has, "sent_source"].notna().all()
    # No look-ahead on the refit rows: the model that scored a headline was fitted before it.
    refit = out["sent_source"] == "yearly_refit"
    assert (pd.to_datetime(out.loc[refit, "sent_model_train_end"])
            <= pd.to_datetime(out.loc[refit, "published_date"])).all()

    out.to_csv(output_path, index=False, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scored", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--fill-before", default="2019-01-01",
                        help="rows scored by a yearly model older than this take the saved "
                             "model's label (look-ahead); 'none' disables the fill")
    args = parser.parse_args()
    out = build(scored_path=args.scored, output_path=args.out,
                fill_before=None if args.fill_before.lower() == "none" else args.fill_before)
    print(f"{len(out):,} headline rows over {out.session.nunique():,} sessions "
          f"{out.session.min().date()}..{out.session.max().date()} -> {args.out}")
    by_year = out.groupby(out.session.dt.year).agg(
        headlines=("row_id", "size"), scored=("sent_label", "count"),
        saved_model=("sent_source", lambda s: (s == "saved_model").sum()),
        negative=("sent_label", lambda s: (s == "negative").sum()),
        neutral=("sent_label", lambda s: (s == "neutral").sum()),
        positive=("sent_label", lambda s: (s == "positive").sum()))
    print(by_year.to_string())


if __name__ == "__main__":
    main()
