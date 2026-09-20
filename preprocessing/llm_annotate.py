"""Annotate the sentiment gold CSV with a local LM Studio model."""

import argparse
import json
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from gold import DEFAULT_OUTPUT, LABELS

DEFAULT_ENDPOINT = "http://localhost:1234/api/v1/chat"
DEFAULT_MODEL = "qwen2.5-7b-instruct-1m"
ANNOTATION_ATTEMPTS = 3


def _response_text(response: dict) -> str:
    try:
        return response["output"][0]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("LM Studio response did not contain message content") from error


def _parse_pseudo_annotation(content: str) -> dict[str, str] | None:
    malformed = content.replace('\\"', '"').strip('"')
    match = re.search(
        r'(?:label|value)\s*:\s*"(?P<label>[^"]+)"\s*,\s*'
        # Trailing quote optional: json.loads() may already have unwrapped the
        # payload, after which .strip('"') above eats the final field's real
        # closing quote. Non-greedy + anchored so a present quote is not
        # swallowed into the captured reason.
        r'reason\s*:\s*"(?P<reason>.+?)"?\s*$',
        malformed,
        flags=re.DOTALL,
    )
    return match.groupdict() if match else None


def _parse_annotation(content: str) -> tuple[str, str]:
    content = content.strip()
    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        object_start = content.find("{")
        if object_start < 0:
            result = _parse_pseudo_annotation(content)
            if result is None:
                raise ValueError(f"Model returned invalid JSON: {content!r}") from error
            object_start = None
        if object_start is not None:
            try:
                result, _ = json.JSONDecoder().raw_decode(content[object_start:])
            except json.JSONDecodeError as nested_error:
                result = _parse_pseudo_annotation(content)
                if result is None:
                    raise ValueError(f"Model returned invalid JSON: {content!r}") from nested_error

    if isinstance(result, str):
        result = _parse_pseudo_annotation(result) or result

    if not isinstance(result, dict):
        raise ValueError("Model JSON must be an object")
    label = result.get("label")
    reason = result.get("reason")
    if label not in LABELS:
        raise ValueError(f"Model returned invalid label: {label!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Model returned an empty or invalid reason")
    return label, reason.strip()


def annotate_headline(
    headline: str,
    lang: str,
    relevance_tag: str,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    timeout: int = 120,
) -> tuple[str, str]:
    """Ask LM Studio for one sentiment label and a concise explanation."""
    system_prompt = (
        "You annotate news headline sentiment for the Tunisian economy. "
        "Return ONLY one complete valid JSON object with exactly two fields: label and reason. "
        f"label must be one of {LABELS}. reason must be one short sentence of at most 15 words. "
        "Do not use Markdown, prefixes, suffixes, or extra quotes."
    )
    user_prompt = json.dumps(
        {
            "headline": headline,
            "language": lang,
            "relevance_tag": relevance_tag,
        },
        ensure_ascii=False,
    )
    last_error: ValueError | None = None
    for _ in range(ANNOTATION_ATTEMPTS):
        payload = {
            "model": model,
            "system_prompt": system_prompt,
            "input": user_prompt,
        }
        request = Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as error:
            raise RuntimeError(f"Could not reach LM Studio at {endpoint}: {error}") from error
        try:
            return _parse_annotation(_response_text(result))
        except ValueError as error:
            last_error = error

    raise ValueError(
        f"LM Studio returned invalid annotation after {ANNOTATION_ATTEMPTS} attempts"
    ) from last_error


def annotator_columns(annotator: int) -> tuple[str, str, str]:
    """(label, notes, model) column names for annotator N.

    Annotator 1 keeps the original `annotation_notes` name so the existing
    3,000-row file needs no migration; annotators 2+ get their own notes column.
    """
    if annotator < 1:
        raise ValueError("annotator must be >= 1")
    notes = "annotation_notes" if annotator == 1 else f"annotator_{annotator}_notes"
    return f"annotator_{annotator}_label", notes, f"annotator_{annotator}_model"


def annotate_file(
    input_path: Path = DEFAULT_OUTPUT,
    output_path: Path | None = None,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    timeout: int = 120,
    annotator: int = 1,
) -> pd.DataFrame:
    """Annotate blank rows and persist after every successful row.

    `annotator` selects which annotator_N_* columns are written, so a second or
    third model can label the same gold set independently for an agreement
    statistic. Rows already labelled by THIS annotator are skipped, so a run is
    resumable and never overwrites another annotator's work.
    """
    output_path = output_path or input_path
    label_col, notes_col, model_col = annotator_columns(annotator)
    df = pd.read_csv(input_path, keep_default_na=False)
    required = {"headline_clean", "lang", "relevance_tag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    for column in (label_col, notes_col, model_col, "annotation_status"):
        if column not in df.columns:
            df[column] = ""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for index, row in df.iterrows():
        if row[label_col] and row[notes_col]:
            continue
        label, reason = annotate_headline(
            row["headline_clean"], row["lang"], row["relevance_tag"],
            endpoint=endpoint, model=model, timeout=timeout,
        )
        df.at[index, label_col] = label
        df.at[index, notes_col] = reason
        # Which model produced this label travels with the data -- an agreement
        # statistic is meaningless without knowing what was compared.
        df.at[index, model_col] = model
        df.at[index, "annotation_status"] = "llm_annotated"
        df.to_csv(output_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--annotator", type=int, default=1,
                        help="which annotator_N_* columns to write (default 1)")
    args = parser.parse_args()
    result = annotate_file(args.input, args.output, args.endpoint, args.model,
                           args.timeout, args.annotator)
    label_col, _, _ = annotator_columns(args.annotator)
    print(f"Annotated {result[label_col].ne('').sum()} of {len(result)} rows "
          f"as {label_col} using {args.model}")


if __name__ == "__main__":
    main()