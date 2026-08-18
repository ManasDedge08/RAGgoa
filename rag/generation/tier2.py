"""Tier 2: generative answer, streamed, explicitly outside the 200 ms budget.

Tier 2 exists to sound like a person rather than a search result. It reads the
top passages and speaks one natural answer in the asker's language. It is never
counted against the Tier 1 target, is always logged as its own latency series,
and is only allowed to reach the speaker after ``grounding.check`` passes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

from ..retrieval.lang import display_name
from ..retrieval.retriever import RetrievalResult
from ..voice.sarvam import SarvamClient, SarvamError, describe
from .confidence import Confidence
from .grounding import GroundingVerdict, check
from .templates import prefix_for

MAX_PASSAGES = 3
MAX_PASSAGE_CHARS = 700

SYSTEM_PROMPT = (
    "You answer questions out loud for a voice assistant. Rules:\n"
    "1. Use only the numbered passages provided. Never add facts from memory.\n"
    "2. Answer in {language} — the same language the question was asked in.\n"
    "3. Two or three spoken sentences. No lists, no markdown, no citations read aloud.\n"
    "4. If the passages do not answer the question, say so plainly instead of guessing."
)


@dataclass
class Tier2Answer:
    text: str
    spoken_text: str
    grounding: GroundingVerdict | None
    used_fallback: bool
    latency_ms: float
    first_token_ms: float
    error: str | None = None
    passages_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "spoken_text": self.spoken_text,
            "grounding": self.grounding.to_dict() if self.grounding else None,
            "used_fallback": self.used_fallback,
            "latency_ms": round(self.latency_ms, 2),
            "first_token_ms": round(self.first_token_ms, 2),
            "error": self.error,
            "passages_used": self.passages_used,
        }


def build_prompt(result: RetrievalResult) -> list[dict]:
    lines = []
    for i, cand in enumerate(result.candidates[:MAX_PASSAGES], start=1):
        text = cand.text[:MAX_PASSAGE_CHARS]
        tag = "" if cand.lang == result.lang else f" (source language: {display_name(cand.lang)})"
        lines.append(f"[{i}]{tag} {text}")
    passages = "\n".join(lines)
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(language=display_name(result.lang))},
        {"role": "user", "content": f"Question: {result.query}\n\nPassages:\n{passages}"},
    ]


async def generate(
    result: RetrievalResult,
    confidence: Confidence,
    tier1_span: str,
    client: SarvamClient,
    on_delta: Callable[[str], None] | None = None,
) -> Tier2Answer:
    """Stream a synthesised answer, then gate it on the grounding check."""
    start = time.perf_counter()
    first_token = 0.0
    chunks: list[str] = []
    error: str | None = None

    try:
        async for delta in client.chat_stream(build_prompt(result)):
            if not chunks:
                first_token = (time.perf_counter() - start) * 1000
            chunks.append(delta)
            if on_delta:
                on_delta(delta)
    except SarvamError as exc:
        error = describe(exc)

    text = "".join(chunks).strip()
    elapsed = (time.perf_counter() - start) * 1000

    if error or not text:
        # Tier 2 failing is not a pipeline failure: Tier 1 already answered.
        return Tier2Answer(
            text="",
            spoken_text=tier1_span,
            grounding=None,
            used_fallback=True,
            latency_ms=elapsed,
            first_token_ms=first_token,
            error=error or "empty generation",
            passages_used=[c.passage_id for c in result.candidates[:MAX_PASSAGES]],
        )

    verdict = check(text, result.candidates[:MAX_PASSAGES], tier1_span)
    if verdict.supported:
        spoken = prefix_for(result.lang, confidence.tier) + text
        used_fallback = False
    else:
        # Drifted from the evidence: speak the extractive span instead.
        spoken = prefix_for(result.lang, "low") + tier1_span
        used_fallback = True

    return Tier2Answer(
        text=text,
        spoken_text=spoken,
        grounding=verdict,
        used_fallback=used_fallback,
        latency_ms=(time.perf_counter() - start) * 1000,
        first_token_ms=first_token,
        passages_used=[c.passage_id for c in result.candidates[:MAX_PASSAGES]],
    )
