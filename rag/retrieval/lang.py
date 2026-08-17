"""Script-based language detection.

The four demo languages use disjoint Unicode blocks, so counting characters is
both exact and free — far cheaper than loading a language-id model into the
timed path.
"""

from __future__ import annotations

from ..config import LANGUAGES

_RANGES = {
    "hin_Deva": ((0x0900, 0x097F),),
    "ben_Beng": ((0x0980, 0x09FF),),
    "tam_Taml": ((0x0B80, 0x0BFF),),
    "eng_Latn": ((0x0041, 0x005A), (0x0061, 0x007A)),
}


def detect_lang(text: str) -> tuple[str, float]:
    """Return (dataset language code, confidence in 0..1)."""
    counts = {lang: 0 for lang in _RANGES}
    total = 0
    for ch in text:
        cp = ord(ch)
        for lang, ranges in _RANGES.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[lang] += 1
                total += 1
                break
    if total == 0:
        return "eng_Latn", 0.0
    lang = max(counts, key=counts.get)
    return lang, counts[lang] / total


def sarvam_code(lang: str) -> str:
    return LANGUAGES.get(lang, LANGUAGES["eng_Latn"])["sarvam"]


def from_sarvam_code(code: str) -> str:
    for lang, meta in LANGUAGES.items():
        if meta["sarvam"] == code:
            return lang
    return "eng_Latn"


def display_name(lang: str) -> str:
    return LANGUAGES.get(lang, {}).get("name", lang)
