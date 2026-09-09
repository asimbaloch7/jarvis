"""Interpreting spoken answers to follow-up questions.

Deliberately local and dependency-free. Yes/no and pick-one-of-these are the
overwhelming majority of follow-ups, and resolving them here rather than with
a Gemini round trip keeps conversations fast and keeps them working offline.
"""

from __future__ import annotations

import difflib
import re

YES = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "affirmative",
    "do it", "go ahead", "please do", "correct", "right", "confirm",
    "confirmed", "aye", "absolutely", "definitely", "of course", "y",
}
NO = {
    "no", "nope", "nah", "negative", "don't", "do not", "stop", "cancel",
    "abort", "never mind", "nevermind", "forget it", "n", "no thanks",
}
CANCEL = {
    "cancel", "abort", "stop", "never mind", "nevermind", "forget it",
    "quit", "exit", "leave it", "drop it",
}

ORDINALS = {
    "first": 1, "1st": 1, "one": 1, "number one": 1,
    "second": 2, "2nd": 2, "two": 2, "number two": 2,
    "third": 3, "3rd": 3, "three": 3, "number three": 3,
    "fourth": 4, "4th": 4, "four": 4, "number four": 4,
    "fifth": 5, "5th": 5, "five": 5, "number five": 5,
    "sixth": 6, "6th": 6, "six": 6,
    "seventh": 7, "7th": 7, "seven": 7,
    "eighth": 8, "8th": 8, "eight": 8,
    "ninth": 9, "9th": 9, "nine": 9,
    "tenth": 10, "10th": 10, "ten": 10,
    "last": -1, "last one": -1, "bottom": -1,
}

# Longest phrase first, so "the last one" resolves on "last" rather than
# stumbling into the "one" that follows it.
_ORDINAL_PHRASES = sorted(ORDINALS, key=len, reverse=True)

_FILLER = re.compile(
    r"^\s*(um+|uh+|er+|well|so|like|okay|ok|please|jarvis|hey jarvis)[,\s]+",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    """Strip filler, punctuation, and casing for matching purposes."""
    cleaned = (text or "").strip()
    previous = None
    while cleaned != previous:
        previous = cleaned
        cleaned = _FILLER.sub("", cleaned)
    cleaned = cleaned.strip().strip(".!?,;:")
    return cleaned.lower().strip()


def is_cancel(text: str) -> bool:
    normalized = normalize(text)
    return normalized in CANCEL or any(
        normalized.startswith(word) for word in ("cancel", "abort", "never mind", "forget it")
    )


def resolve_confirm(text: str) -> bool | None:
    """True, False, or None if the answer wasn't a yes or a no."""
    normalized = normalize(text)
    if not normalized:
        return None
    if normalized in YES:
        return True
    if normalized in NO:
        return False
    # Tokenize rather than split, so a trailing comma doesn't hide the "no".
    words = re.findall(r"[a-z']+", normalized)
    if not words:
        return None
    first = words[0]
    if first in {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "y"}:
        return True
    if first in {"no", "nope", "nah", "don't", "n"}:
        return False
    if any(phrase in normalized for phrase in ("go ahead", "do it", "run it")):
        return True
    if any(phrase in normalized for phrase in ("don't", "do not", "cancel", "stop")):
        return False
    return None


def resolve_choice(text: str, options: list[str]) -> str | None:
    """Match a spoken answer to one of `options`, or None if it's ambiguous.

    Handles the literal option text, substrings, ordinals ("the second one"),
    bare numbers, and near-misses from the transcriber.
    """
    if not options:
        return None
    normalized = normalize(text)
    if not normalized:
        return None

    lowered = [opt.lower() for opt in options]

    if normalized in lowered:
        return options[lowered.index(normalized)]

    # A single option contained in what they said, e.g. "the new one" -> "new".
    contained = [opt for opt, low in zip(options, lowered) if low in normalized]
    if len(contained) == 1:
        return contained[0]

    # What they said contained in exactly one option, e.g. "invoice" -> "invoice-parser".
    partial = [opt for opt, low in zip(options, lowered) if normalized in low]
    if len(partial) == 1:
        return partial[0]

    for phrase in _ORDINAL_PHRASES:
        if re.search(rf"\b{re.escape(phrase)}\b", normalized):
            index = ORDINALS[phrase]
            if index == -1:
                return options[-1]
            if 1 <= index <= len(options):
                return options[index - 1]

    digits = re.findall(r"\b(\d{1,2})\b", normalized)
    if digits:
        index = int(digits[0])
        if 1 <= index <= len(options):
            return options[index - 1]

    # Last resort: fuzzy match, but only when it's clearly the best candidate.
    scores = [
        (opt, difflib.SequenceMatcher(None, normalized, low).ratio())
        for opt, low in zip(options, lowered)
    ]
    scores.sort(key=lambda pair: pair[1], reverse=True)
    best, best_score = scores[0]
    runner_up = scores[1][1] if len(scores) > 1 else 0.0
    if best_score >= 0.65 and best_score - runner_up >= 0.12:
        return best

    # Token overlap catches "the parser project" against "invoice-parser".
    said_tokens = set(re.findall(r"[a-z0-9]+", normalized))
    token_scores = []
    for opt, low in zip(options, lowered):
        opt_tokens = set(re.findall(r"[a-z0-9]+", low))
        if opt_tokens:
            token_scores.append((opt, len(said_tokens & opt_tokens) / len(opt_tokens)))
    if token_scores:
        token_scores.sort(key=lambda pair: pair[1], reverse=True)
        if token_scores[0][1] >= 0.5:
            if len(token_scores) == 1 or token_scores[0][1] > token_scores[1][1]:
                return token_scores[0][0]

    return None


def resolve_free(text: str) -> str:
    """Light cleanup for a free-form answer; the skill decides what it means."""
    return normalize(text)


# Phrases people put in front of a name. Longest first so "call it" wins over
# "call". Deliberately conservative: stripping too eagerly mangles real names.
_NAME_PREFIX = re.compile(
    r"^(?:let'?s\s+|i\s+want\s+to\s+|please\s+|you\s+can\s+)*"
    r"(?:call\s+the\s+project|call\s+it|name\s+it|the\s+name\s+is|named|call\s+this)\s+"
)
_SPOKEN_FILLER = re.compile(r"\b(uh+|um+|er+|erm+)\b")


def slugify_spoken_name(text: str) -> str:
    """Turn a spoken project name into a directory name.

    "call it My Invoice Parser" -> "my-invoice-parser"
    "invoice parser dash v2"    -> "invoice-parser-v2"
    """
    cleaned = normalize(text)
    cleaned = cleaned.replace("'", "").replace("\u2019", "")
    # People say the punctuation out loud when spelling a name for a computer.
    cleaned = re.sub(r"\b(dash|hyphen)\b", "-", cleaned)
    cleaned = re.sub(r"\b(underscore|under score)\b", "_", cleaned)
    cleaned = re.sub(r"\b(dot|period|point)\b", ".", cleaned)
    cleaned = re.sub(r"[^\w\s.\-]", " ", cleaned)
    cleaned = _SPOKEN_FILLER.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = _NAME_PREFIX.sub("", cleaned).strip()
    cleaned = re.sub(r"\s*([-_.])\s*", r"\1", cleaned)
    cleaned = re.sub(r"\s+", "-", cleaned.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-_.")
    return cleaned
