"""An absolute relevance gate, so "I don't know" is actually reachable.

The confidence score in ``confidence.py`` is *relative*: similarity margin,
strategy agreement, how far the top hit sits above the rest. All of that can
look healthy when nothing relevant was retrieved at all, because a list of
uniformly irrelevant passages still has a best member. The result was a system
that hedged — "I found something related, but I'm not fully sure" — instead of
declining, which is the worse failure of the two.

The obvious fix does not work. Bi-encoder cosine cannot separate the cases:
measured over 160 corpus questions and 14 out-of-corpus ones, in-corpus top
similarity sits at p5 0.836 / p50 0.879 and out-of-corpus at p50 0.841 with a
maximum of 0.915. A floor that keeps 94% of real questions catches half the
rest. E5 similarity says "these share vocabulary", not "this answers that".

A cross-encoder scores the pair jointly, which is the judgment actually needed.
Over the same sets, a floor of -2.0 on the top-3 maximum keeps 88% of corpus
questions and declines 64% of out-of-corpus ones, for about 24 ms.

That is an improvement, not a solution, and the number is soft for an honest
reason: "out of corpus" is not cleanly defined against a 1,200-question sample
of MS MARCO. Several probes used as negatives — "what is bitcoin",
"what is photosynthesis" — may legitimately be answerable, which caps the
measurable separation. The floor is therefore set where declining is cheap and
tuned to prefer declining, and it is switchable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..config import (
    RELEVANCE_GATE_DEPTH,
    RELEVANCE_GATE_ENABLED,
    RELEVANCE_GATE_FLOOR,
    CROSS_ENCODER_MAX_CHARS,
)
from ..retrieval.retriever import Candidate
from ..retrieval.store import get_store


@dataclass
class RelevanceVerdict:
    checked: bool
    passed: bool
    score: float
    floor: float
    latency_ms: float

    def to_dict(self) -> dict:
        return {
            "checked": self.checked,
            "passed": self.passed,
            "score": round(self.score, 3),
            "floor": self.floor,
            "latency_ms": round(self.latency_ms, 2),
        }


def check(query: str, candidates: list[Candidate], enabled: bool | None = None) -> RelevanceVerdict:
    """Does anything retrieved actually answer the question?"""
    active = RELEVANCE_GATE_ENABLED if enabled is None else enabled
    if not active or not candidates:
        return RelevanceVerdict(False, bool(candidates), 0.0, RELEVANCE_GATE_FLOOR, 0.0)

    start = time.perf_counter()
    head = candidates[:RELEVANCE_GATE_DEPTH]
    pairs = [(query, c.text[:CROSS_ENCODER_MAX_CHARS]) for c in head]
    scores = get_store().cross_encoder.predict(
        pairs, batch_size=len(pairs), show_progress_bar=False
    )
    best = max(float(s) for s in scores)

    # Record it on the candidate so the trace can show why a turn was declined.
    for cand, score in zip(head, scores):
        cand.raw_scores["relevance"] = float(score)

    return RelevanceVerdict(
        checked=True,
        passed=best >= RELEVANCE_GATE_FLOOR,
        score=best,
        floor=RELEVANCE_GATE_FLOOR,
        latency_ms=(time.perf_counter() - start) * 1000,
    )
