"""Grounding check for Tier 2 output.

Tier 2 is free prose from a language model; Tier 1 is a span lifted verbatim
from a retrieved passage. Before Tier 2 is allowed to be spoken it has to be
shown to stay inside the retrieved evidence. Two cheap, complementary tests:

* lexical - what fraction of the generated content words appear in the
  retrieved passages (catches invented names, numbers, dates)
* semantic - cosine between the generated answer and the retrieved evidence in
  the same embedding space used for retrieval (catches fluent paraphrase that
  drifts to a different claim)

Failing either test drops the answer back to the Tier 1 span, which cannot
hallucinate because it was never generated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..index.text import tokenize
from ..retrieval.retriever import Candidate
from ..retrieval.store import get_store

LEXICAL_FLOOR = 0.45
SEMANTIC_FLOOR = 0.72
# Content words are what matter; very short tokens are mostly grammar.
MIN_TOKEN_LEN = 3


@dataclass
class GroundingVerdict:
    supported: bool
    lexical: float
    semantic: float
    reason: str
    latency_ms: float

    def to_dict(self) -> dict:
        return {
            "supported": self.supported,
            "lexical": round(self.lexical, 3),
            "semantic": round(self.semantic, 3),
            "reason": self.reason,
            "latency_ms": round(self.latency_ms, 2),
        }


def check(answer: str, candidates: list[Candidate], tier1_span: str) -> GroundingVerdict:
    start = time.perf_counter()
    if not answer.strip():
        return GroundingVerdict(False, 0.0, 0.0, "empty generation", 0.0)

    evidence_texts = [c.text for c in candidates] + [tier1_span]
    evidence_tokens: set[str] = set()
    for text in evidence_texts:
        evidence_tokens.update(t for t in tokenize(text) if len(t) >= MIN_TOKEN_LEN)

    answer_tokens = [t for t in tokenize(answer) if len(t) >= MIN_TOKEN_LEN]
    lexical = (
        sum(1 for t in answer_tokens if t in evidence_tokens) / len(answer_tokens)
        if answer_tokens
        else 0.0
    )

    store = get_store()
    answer_vec = store.encode_query(answer)[0]
    rows = [
        store.passages.by_uid[c.passage_id].row
        for c in candidates
        if c.passage_id in store.passages.by_uid
    ]
    if rows:
        semantic = float(np.max(store.passages.vectors[rows] @ answer_vec))
    else:
        semantic = 0.0

    if lexical < LEXICAL_FLOOR and semantic < SEMANTIC_FLOOR:
        reason = f"unsupported: lexical {lexical:.2f} and semantic {semantic:.2f} both below floor"
        supported = False
    elif lexical < LEXICAL_FLOOR:
        reason = f"weak lexical support ({lexical:.2f}), carried by semantic {semantic:.2f}"
        supported = True
    elif semantic < SEMANTIC_FLOOR:
        reason = f"weak semantic support ({semantic:.2f}), carried by lexical {lexical:.2f}"
        supported = True
    else:
        reason = "supported by retrieved passages"
        supported = True

    return GroundingVerdict(
        supported=supported,
        lexical=lexical,
        semantic=semantic,
        reason=reason,
        latency_ms=(time.perf_counter() - start) * 1000,
    )
