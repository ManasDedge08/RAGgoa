"""Script-based language detection, and honesty about what it cannot decide.

Counting Unicode blocks is exact and free, which is why it sits in the timed
path instead of a language-id model. But scripts do not map one-to-one onto
languages: Hindi, Marathi, Nepali and Sanskrit all write in Devanagari, and
Bengali and Assamese share a block. For those, detection narrows the query to a
script and then guesses the most widely spoken candidate.

The guess is reported rather than hidden. ``detect`` returns the alternatives
it could not rule out, the API passes them to the client, and the UI offers a
picker. Spoken queries never rely on this: Sarvam returns the language code.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import LANGUAGE_CODES
from ..languages import (
    REGISTRY,
    SCRIPT_DEFAULT,
    SCRIPT_RANGES,
    by_sarvam,
)


@dataclass
class Detection:
    lang: str
    confidence: float  # share of characters belonging to the winning script
    script: str
    alternatives: list[str]  # other enabled languages using the same script

    @property
    def ambiguous(self) -> bool:
        return bool(self.alternatives)

    def to_dict(self) -> dict:
        return {
            "lang": self.lang,
            "confidence": round(self.confidence, 3),
            "script": self.script,
            "alternatives": self.alternatives,
            "ambiguous": self.ambiguous,
        }


def detect(text: str) -> Detection:
    counts: dict[str, int] = {}
    total = 0
    for ch in text:
        cp = ord(ch)
        for script, ranges in SCRIPT_RANGES.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[script] = counts.get(script, 0) + 1
                total += 1
                break

    if not total:
        return Detection("eng_Latn", 0.0, "Latin", [])

    script = max(counts, key=counts.get)
    enabled_in_script = [
        code for code in LANGUAGE_CODES if REGISTRY[code].script == script
    ]
    if not enabled_in_script:
        return Detection("eng_Latn", 0.0, script, [])

    preferred = SCRIPT_DEFAULT.get(script)
    lang = preferred if preferred in enabled_in_script else enabled_in_script[0]
    alternatives = [c for c in enabled_in_script if c != lang]
    return Detection(lang, counts[script] / total, script, alternatives)


def detect_lang(text: str) -> tuple[str, float]:
    """Backwards-compatible pair form used inside the timed path."""
    result = detect(text)
    return result.lang, result.confidence


def sarvam_code(lang: str) -> str:
    entry = REGISTRY.get(lang)
    return entry.sarvam if entry else "en-IN"


def from_sarvam_code(code: str) -> str:
    return by_sarvam(code)


def display_name(lang: str) -> str:
    entry = REGISTRY.get(lang)
    return entry.name if entry else lang


def has_voice(lang: str) -> bool:
    """Whether a spoken answer can be produced in this language."""
    entry = REGISTRY.get(lang)
    return bool(entry and entry.tts)
