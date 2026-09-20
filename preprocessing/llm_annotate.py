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



# --- Prompt versions ---------------------------------------------------------
# v1 produced the original 3,000 labels. It is KEPT VERBATIM so those labels stay
# reproducible and attributable; do not edit it. Its failure mode is documented in
# AUDIT_REPORT.md section 8d: four of its five lines are JSON formatting rules and it
# never defines the task, so labels tracked growth-flavoured vocabulary rather than
# market impact (58.8% positive, 9.4% neutral, "weather forecast" -> neutral).
PROMPT_V1 = (
    "You annotate news headline sentiment for the Tunisian economy. "
    "Return ONLY one complete valid JSON object with exactly two fields: label and reason. "
    f"label must be one of {LABELS}. reason must be one short sentence of at most 15 words. "
    "Do not use Markdown, prefixes, suffixes, or extra quotes."
)

PROMPT_V2 = """You are labelling news headlines for a study of the Tunisian stock market (Tunindex, Bourse de Tunis).

THE QUESTION YOU ARE ANSWERING
For each headline, answer ONE question: would a Tunisian equity investor, reading only this headline, become more or less optimistic about the near-term value of Tunisian listed companies?

You are NOT judging whether the news is pleasant, whether the topic sounds economic, whether it describes progress, or whether the words are upbeat. Judge expected MARKET IMPACT only.

NEUTRAL IS THE DEFAULT AND SHOULD BE YOUR MOST COMMON ANSWER
Most news headlines have no clear directional implication for share prices. Announcements, plans, appointments, meetings, agreements to cooperate, draft laws, studies, conferences, visits, and descriptive statistics are almost always NEUTRAL. Choose a directional label only when you can state which way prices should move and why. If you are unsure, the answer is neutral. A label of neutral is a correct and informative answer, not a failure to decide.

Foreign news is neutral unless it plausibly transmits to Tunisia (oil price, EU demand, Fed or ECB rates, remittances, tourism, grain prices). News about another country's domestic affairs is neutral.

THE LABELS
- very_negative: a large, direct hit to earnings, solvency, or the cost of capital. Sovereign downgrade, banking crisis, default, major devaluation, sharp index fall.
- negative: a clear adverse effect of ordinary size. Rising inflation, a rate hike, falling profits, widening deficit, a strike at a listed firm.
- neutral: no clear directional implication, or the effects plausibly offset. This includes unchanged policy, procedural and announcement news, and off-topic items.
- positive: a clear favourable effect of ordinary size. Falling inflation, a rate cut, rising profits, new financing secured, improved outlook.
- very_positive: a large, direct improvement. Major investment inflow, sovereign upgrade, debt relief, sharp index rise, record results at a large listed company.

CALIBRATION
Reserve very_negative and very_positive for genuinely large or systemic items; they should be rare. Do not use magnitude words in the headline as a proxy for magnitude of impact.

TRAPS
- "increases" / "exceeds" / "hausse" is not automatically positive. Rising inflation, rising debt, rising unemployment and rising money supply are negative or neutral.
- A draft law, a plan, or a project "under preparation" has not happened yet: neutral.
- A headline that merely concerns the economy is not positive. Topic is not sentiment.
- "maintains" / "unchanged" / "inchangé" means no change: neutral.
- A question ("Will X happen?") states no fact: neutral.
- A headline reporting the index's own past move is labelled by the direction it reports.

EXAMPLES
"La BCT maintient inchangé son taux directeur à 7 % face aux risques d'inflation" -> neutral (policy held; no change to the cost of capital)
"L'inflation repart à la hausse en août, portée par l'alimentation" -> negative (rising inflation pressures margins and invites tightening, despite the word "hausse")
"Fitch maintient la note de la Tunisie à « B- » avec perspective stable" -> neutral (rating affirmed, outlook unchanged)
"Fitch Ratings pointe la dépendance croissante de la Tunisie au financement direct de la BCT" -> negative (flags monetary financing risk)
"Le bénéfice net de SMART Tunisie s'envole de 45 % au premier semestre" -> positive (large earnings improvement at a listed firm)
"La BIAT et la BAD signent une convention de 50 millions de dollars" -> neutral (agreement announced; effect on earnings not yet determinable)
"Tunisie-Environnement : un code de l'environnement est en cours d'élaboration" -> neutral (draft regulation, not yet in force)
"Les billets et monnaies en circulation dépassent la barre des 24 milliards de dinars" -> neutral (descriptive monetary statistic, no clear equity implication)
"L'inflation au Maroc recule de 0,6% en juillet" -> neutral (another country's domestic data)
"Tunisie - Météo : quel temps fera-t-il mardi ?" -> neutral (off topic)

INPUT
You receive JSON with: headline, language (fr, ar or en), and relevance_tag. The tag is "tunisia_econ" when the headline matched Tunisian economic terms and "global_linked" when it matched international terms. Judge a global_linked headline by whether it plausibly transmits to Tunisia. Headlines may be French, Arabic or English; apply identical criteria.

OUTPUT
Return ONLY one valid JSON object with exactly two fields: label and reason.
label must be one of {labels}.
reason must be one short sentence of at most 15 words naming the mechanism, not restating the headline.
No Markdown, no prefixes, no suffixes, no extra quotes.""".replace("{labels}", str(LABELS))

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}
DEFAULT_PROMPT_VERSION = "v2"


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
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> tuple[str, str]:
    """Ask LM Studio for one sentiment label and a concise explanation."""
    if prompt_version not in PROMPTS:
        raise ValueError(f"Unknown prompt_version {prompt_version!r}; have {sorted(PROMPTS)}")
    system_prompt = PROMPTS[prompt_version]
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
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> pd.DataFrame:
    """Annotate blank rows and persist after every successful row.

    `annotator` selects which annotator_N_* columns are written, so a second or
    third model can label the same gold set independently for an agreement
    statistic. Rows already labelled by THIS annotator are skipped, so a run is
    resumable and never overwrites another annotator's work.
    """
    output_path = output_path or input_path
    label_col, notes_col, model_col = annotator_columns(annotator)
    prompt_col = f"annotator_{annotator}_prompt"
    df = pd.read_csv(input_path, keep_default_na=False)
    required = {"headline_clean", "lang", "relevance_tag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    for column in (label_col, notes_col, model_col, prompt_col, "annotation_status"):
        if column not in df.columns:
            df[column] = ""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for index, row in df.iterrows():
        if row[label_col] and row[notes_col]:
            continue
        label, reason = annotate_headline(
            row["headline_clean"], row["lang"], row["relevance_tag"],
            endpoint=endpoint, model=model, timeout=timeout,
            prompt_version=prompt_version,
        )
        df.at[index, label_col] = label
        df.at[index, notes_col] = reason
        # Which model produced this label travels with the data -- an agreement
        # statistic is meaningless without knowing what was compared.
        df.at[index, model_col] = model
        # Prompt version travels with the label: v1 and v2 labels are NOT
        # comparable (AUDIT_REPORT.md section 8d) and must never be pooled silently.
        df.at[index, prompt_col] = prompt_version
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
    parser.add_argument("--prompt-version", choices=sorted(PROMPTS),
                        default=DEFAULT_PROMPT_VERSION,
                        help="v1 produced the original 3,000 labels; v2 defines the task")
    args = parser.parse_args()
    result = annotate_file(args.input, args.output, args.endpoint, args.model,
                           args.timeout, args.annotator, args.prompt_version)
    label_col, _, _ = annotator_columns(args.annotator)
    print(f"Annotated {result[label_col].ne('').sum()} of {len(result)} rows "
          f"as {label_col} using {args.model} (prompt {args.prompt_version})")


if __name__ == "__main__":
    main()