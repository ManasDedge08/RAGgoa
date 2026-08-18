"""Tokenisation shared by the sparse index and the guardrail.

Python's built-in ``re`` does not treat Unicode combining marks as word
characters, so ``\\w+`` splits every Devanagari, Tamil and Bengali word at its
vowel signs and virama — ``कॉर्पोरेशन`` tokenises as
``['क', 'र', 'प', 'र', 'शन']``. That silently guts BM25 for every Indic
language here while leaving English untouched, which is exactly the kind of
bug that never shows up in an English smoke test.

The ``regex`` module supports Unicode property classes, so a token is defined
as a run of letters, combining marks and digits.
"""

from __future__ import annotations

import unicodedata

import regex

# \p{L} letters, \p{M} combining marks (the fix), \p{N} digits.
_TOKEN = regex.compile(r"[\p{L}\p{M}\p{N}]+")


def normalise(text: str) -> str:
    """NFKC-fold so visually identical Indic sequences hash to one token."""
    return unicodedata.normalize("NFKC", text).lower()


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(normalise(text))
