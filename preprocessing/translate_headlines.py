"""Translate every relevant headline to English, locally and incrementally.

    python3 preprocessing/translate_headlines.py               # translate what is new
    python3 preprocessing/translate_headlines.py --check-gold  # reproduce the committed translations

-> data/curated/headlines_en.parquet (row_id, lang, headline_clean, headline_en)

No API, no key: Helsinki-NLP/opus-mt-fr-en run on the local GPU, pinned to the
revision recorded in finbert_head_results.json (`translator_revision`). That is
the model behind the off-repo gold translations
(embeddings/sentiment_gold_v2_full__finbert_translated.translations.csv), so this
script is also that orphan's producing script; --check-gold measures how closely
it reproduces them.

INCREMENTAL, which is what makes it a pipeline stage: the output is its own
cache. A rerun translates only headline texts that are not already in the output
under the same model revision, and saves after every chunk, so an interrupted
run resumes and a re-scrape costs only the new headlines.

English rows pass through unchanged. A language with no entry in MODELS stops the
run rather than being silently passed through.

REPAIR PASS. On ~0.04% of headlines the model loops ("Tunisia-Tunisia-Tunisia...")
or drops everything after a "|" separator. Those are detected by `suspicious`,
retranslated segment by segment with a repetition guard, and any that still look
wrong are kept but marked translation_suspect = True -- never silently trusted.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer

import features

CURATED = features.CURATED
DEFAULT_OUTPUT = CURATED / "headlines_en.parquet"
GOLD_TRANSLATIONS = CURATED / "embeddings" / "sentiment_gold_v2_full__finbert_translated.translations.csv"
MODELS = {"fr": ["Helsinki-NLP/opus-mt-fr-en", "c4aed37b318c763fd177aa449b44e3b783cc6c02"]}
BATCH, MAX_LEN, CHUNK = 32, 128, 3200


class Translator:
    def __init__(self, lang: str):
        name, revision = MODELS[lang]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = MarianTokenizer.from_pretrained(name, revision=revision)
        self.model = MarianMTModel.from_pretrained(name, revision=revision).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()

    @torch.no_grad()
    def __call__(self, texts: list[str], **generate) -> list[str]:
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # similar lengths per batch
        out = [""] * len(texts)
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            enc = self.tok([texts[j] for j in idx], return_tensors="pt", padding=True,
                           truncation=True, max_length=MAX_LEN).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=MAX_LEN, **generate)
            for j, text in zip(idx, self.tok.batch_decode(gen, skip_special_tokens=True)):
                out[j] = text
        return out


def numbers_preserved(source: pd.Series, target: pd.Series) -> float:
    """Share of rows whose digits all survive translation.

    Compared as a multiset of digit characters, so 7000 -> 7,000 and 25,3 -> 25.3
    count as preserved; French ordinals (1er, 3e) are dropped first because they
    correctly become words.
    """
    ordinal = re.compile(r"\b\d+(?:er|ère|e|ème|éme|eme|nd|nde)\b")
    kept = [not (Counter(re.findall(r"\d", ordinal.sub("", s))) - Counter(re.findall(r"\d", t)))
            for s, t in zip(source.astype(str), target.astype(str))]
    return round(sum(kept) / max(len(kept), 1), 4)


PIPE = re.compile(r"\s*[|│]\s*")


def suspicious(source: str, target: str) -> bool:
    """A looped or truncated translation: a 3-gram said 3+ times, or an absurd length."""
    words = re.findall(r"\w+", target)
    looped = max(Counter(zip(words, words[1:], words[2:])).values(), default=0) >= 3
    ratio = len(target) / max(len(source), 1)
    return looped or ratio > 2 or (PIPE.search(source) is not None and ratio < 0.6)


def repair(translate: "Translator", texts: list[str]) -> list[str]:
    """Retranslate segment by segment (split on |), junk stripped, repetition guarded."""
    parts = [[p for p in PIPE.split(t.replace("\ufffc", "")) if p.strip()] for t in texts]
    flat = translate([p for ps in parts for p in ps], no_repeat_ngram_size=3)
    out, i = [], 0
    for ps in parts:
        out.append(" | ".join(flat[i:i + len(ps)]))
        i += len(ps)
    return out


def run(headlines_path: Path = features.DEFAULT_HEADLINES,
        output_path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    news = pd.read_parquet(headlines_path, columns=[
        "row_id", "lang", "headline_clean", "is_canonical", "relevance_tag", "date_parse_ok"])
    # Same filter as features.build / score_corpus: the rows the pipeline actually uses.
    news = news[news["is_canonical"] & (news["relevance_tag"] != "other")
                & news["date_parse_ok"]][["row_id", "lang", "headline_clean"]].reset_index(drop=True)
    news["headline_clean"] = news["headline_clean"].astype(str)
    unknown = set(news["lang"]) - set(MODELS) - {"en"}
    if unknown:
        raise SystemExit(f"No translation model configured for {sorted(unknown)} -- add it to MODELS.")

    # The previous output is the cache, but only if the same model revision made it.
    meta_path = output_path.with_name(output_path.stem + "_metadata.json")
    cache: dict[tuple[str, str], str] = {}
    if output_path.exists() and meta_path.exists() and \
            json.loads(meta_path.read_text(encoding="utf-8")).get("models") == MODELS:
        old = pd.read_parquet(output_path).dropna(subset=["headline_en"])
        cache = dict(zip(zip(old["lang"], old["headline_clean"]), old["headline_en"]))
    news["headline_en"] = [t if l == "en" else cache.get((l, t))
                           for l, t in zip(news["lang"], news["headline_clean"])]
    meta = {"models": MODELS, "input": headlines_path.name, "batch_size": BATCH, "max_length": MAX_LEN}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")   # lets a resume trust the cache

    start, translated, repaired = time.time(), 0, 0
    for lang in sorted(set(news["lang"]) - {"en"}):
        todo = news.loc[(news["lang"] == lang) & news["headline_en"].isna(),
                        "headline_clean"].unique().tolist()
        if not todo:
            continue
        translate = Translator(lang)
        for i in range(0, len(todo), CHUNK):
            chunk = todo[i:i + CHUNK]
            mapping = dict(zip(chunk, translate(chunk)))
            hit = (news["lang"] == lang) & news["headline_clean"].isin(mapping)
            news.loc[hit, "headline_en"] = news.loc[hit, "headline_clean"].map(mapping)
            news.to_parquet(output_path, index=False)                   # resume point
            translated += len(chunk)
            print(f"  {lang}: {i + len(chunk):,}/{len(todo):,} new headlines", flush=True)

    def flag() -> pd.Series:
        return pd.Series([l != "en" and suspicious(s, t) for l, s, t in
                          zip(news["lang"], news["headline_clean"], news["headline_en"])], index=news.index)

    bad = flag()
    for lang in sorted(set(news.loc[bad, "lang"])):
        rows = news.index[bad & (news["lang"] == lang)]
        news.loc[rows, "headline_en"] = repair(Translator(lang), news.loc[rows, "headline_clean"].tolist())
        repaired += len(rows)
    news["translation_suspect"] = flag()

    assert news["headline_en"].notna().all() and (news["headline_en"].str.strip() != "").all()
    news.to_parquet(output_path, index=False)
    fr = news[news["lang"] != "en"]
    meta.update({"rows": int(len(news)), "passthrough_en": int((news["lang"] == "en").sum()),
                 "translated_this_run": translated, "repair_attempts_this_run": repaired,
                 "still_suspect": int(news["translation_suspect"].sum()), "minutes": round((time.time() - start) / 60, 1),
                 "numbers_preserved_share": numbers_preserved(fr["headline_clean"], fr["headline_en"])})
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return news


def check_gold() -> None:
    """How closely does this script reproduce the committed off-repo translations?"""
    gold = pd.read_csv(GOLD_TRANSLATIONS, keep_default_na=False)
    gold = gold[gold["original"] != gold["translated"]]          # the rows that were translated
    mine = Translator("fr")(gold["original"].tolist())
    same = [a.strip() == b.strip() for a, b in zip(mine, gold["translated"])]
    print(f"exact match with committed translations: {sum(same):,}/{len(same):,} "
          f"= {sum(same) / len(same):.4f}")
    diffs = [(o, t, m) for o, t, m, s in zip(gold["original"], gold["translated"], mine, same) if not s]
    for original, theirs, ours in diffs[:5]:
        print(f"  FR  {original}\n  old {theirs}\n  new {ours}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-gold", action="store_true")
    args = parser.parse_args()
    check_gold() if args.check_gold else run(output_path=args.out)


if __name__ == "__main__":
    main()
