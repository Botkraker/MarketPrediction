"""
Character-level normalization only. No stemming, no lemmatization, no
stopword removal -- those belong to a later lexicon/robustness pass, not here.
"""
import html
import re
import unicodedata

# U+200B-U+200F (zero-width space/marks, LTR/RTL marks), U+202A-U+202E
# (embedding/override), U+2066-U+2069 (isolates), U+061C (Arabic letter mark).
_INVISIBLE_RE = re.compile(
    "[​-‏‪-‮⁦-⁩؜]"
)

_ALEF_RE = re.compile("[آأإٱ]")  # آ أ إ ٱ -> ا
_TATWEEL_RE = re.compile("ـ")


def clean_text(text: str) -> str:
    """Stage-1 normalization: unescape HTML entities, NFKC, strip invisible
    formatting marks. Case, numbers and punctuation (including tatweel) are
    left untouched -- this is normalization, not cleanup."""
    if text is None:
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE_RE.sub("", text)
    return text.strip()


def arabic_normalize_for_matching(text: str) -> str:
    """A second, more aggressive normalization used ONLY for keyword
    matching (relevance tagging) -- never persisted, never shown to
    annotators. Unifies alef forms, collapses ta marbuta/alef maqsura, and
    strips tatweel so keyword substrings match regardless of elongation."""
    if text is None:
        return ""
    text = _ALEF_RE.sub("ا", text)  # -> ا
    text = text.replace("ة", "ه")  # ة -> ه
    text = text.replace("ى", "ي")  # ى -> ي
    text = _TATWEEL_RE.sub("", text)
    return text


_ARABIC_SCRIPT_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def script_counts(text: str) -> tuple[int, int]:
    """(arabic_script_chars, latin_chars) in text."""
    if not text:
        return 0, 0
    return len(_ARABIC_SCRIPT_RE.findall(text)), len(_LATIN_RE.findall(text))


# Arabizi heuristic: a "word" that mixes letters with digits used as Arabic
# letter substitutes (2,3,5,6,7,8,9 are the common ones), excluding French/
# English ordinals like "3e", "2nd", "1er".
_ORDINAL_RE = re.compile(r"^\d+(e|er|ere|ère|nd|rd|th|st)$", re.IGNORECASE)
_MIXED_WORD_RE = re.compile(r"\b[A-Za-z]*\d[A-Za-z\d]*\b")
_HAS_LETTER_RE = re.compile(r"[A-Za-z]")
_ARABIZI_DIGITS = set("2356789")


def count_arabizi_words(text: str) -> int:
    if not text:
        return 0
    n = 0
    for w in _MIXED_WORD_RE.findall(text):
        if _ORDINAL_RE.match(w):
            continue
        if not _HAS_LETTER_RE.search(w):
            continue
        if not any(c in _ARABIZI_DIGITS for c in w):
            continue
        n += 1
    return n
