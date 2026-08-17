"""Tier 1: extractive answer. Grounded by construction, and the timed path.

No model call happens here. The answer is the best-matching sentence span from
the top-ranked passage, wrapped in confidence-conditioned framing. Everything
in this module is pure string work over data the retriever already produced, so
it adds well under a millisecond to the measured Tier 1 latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..retrieval.lang import display_name
from ..retrieval.retriever import RetrievalResult
from .confidence import Confidence
from .templates import cross_lingual_note, off_topic_for, prefix_for


@dataclass
class Tier1Answer:
    text: str  # what gets spoken
    span: str  # the retrieved span, unwrapped
    passage_id: str
    source_lang: str
    answer_lang: str
    tier: str
    cross_lingual: bool
    latency_ms: float

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "span": self.span,
            "passage_id": self.passage_id,
            "source_lang": self.source_lang,
            "answer_lang": self.answer_lang,
            "tier": self.tier,
            "cross_lingual": self.cross_lingual,
            "latency_ms": round(self.latency_ms, 2),
        }


def build(result: RetrievalResult, confidence: Confidence) -> Tier1Answer:
    start = time.perf_counter()
    lang = result.lang

    if confidence.tier == "refuse" or not result.candidates:
        return Tier1Answer(
            text=prefix_for(lang, "refuse") or off_topic_for(lang),
            span="",
            passage_id="",
            source_lang=lang,
            answer_lang=lang,
            tier="refuse",
            cross_lingual=False,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    top = result.candidates[0]
    span = (top.best_sentence or top.text).strip()
    cross = top.lang != lang
    text = prefix_for(lang, confidence.tier) + span
    if cross:
        text += cross_lingual_note(lang, display_name(top.lang))

    return Tier1Answer(
        text=text,
        span=span,
        passage_id=top.passage_id,
        source_lang=top.lang,
        answer_lang=lang,
        tier=confidence.tier,
        cross_lingual=cross,
        latency_ms=(time.perf_counter() - start) * 1000,
    )
