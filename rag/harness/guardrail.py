"""Query-level guardrail: runs before retrieval, rejects what the corpus cannot answer.

Two checks, both cheap enough to sit in front of the timed path:

* **in-domain** - cosine to the nearest query-cluster centroid. Measured
  separation is weak and the report says so: on a 1,200-query slice of MS
  MARCO, "book me a cab" scores about as close to the corpus's question
  distribution as a question the corpus actually answers, because both are
  ordinary well-formed questions. The floor is therefore set permissively
  (~1% false-reject) to catch only egregious cases at near-zero cost, and the
  real "I have no evidence for this" decision is made after retrieval by the
  confidence tier, which can see what came back.
* **unsafe**    - a small pattern list for requests that should be refused
  outright rather than answered from passages.

Kept deliberately small. Over-engineering this stage buys nothing: the
grounding check downstream is the real safety net for answer quality.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import numpy as np

from ..config import DOMAIN_FILTER_ENABLED, REPORT_DIR
from ..retrieval.store import get_store

DEFAULT_DOMAIN_FLOOR = 0.70

UNSAFE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(synthes|manufactur|make)\w*\s+(a\s+)?(bomb|explosive|nerve agent|meth)",
        r"\bhow\s+to\s+(kill|poison|harm)\b",
        r"\b(child\s+sexual|csam)\b",
        r"\bcredit\s+card\s+(dump|skim)",
    )
]


def _domain_floor() -> float:
    path = REPORT_DIR / "guardrail.json"
    if path.exists():
        try:
            return float(json.loads(path.read_text(encoding="utf-8"))["domain_floor"])
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    return DEFAULT_DOMAIN_FLOOR


@dataclass
class GuardrailVerdict:
    allowed: bool
    reason: str
    domain_similarity: float
    latency_ms: float

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "domain_similarity": round(self.domain_similarity, 3),
            "latency_ms": round(self.latency_ms, 2),
        }


def check_query(query: str, qvec: np.ndarray) -> GuardrailVerdict:
    start = time.perf_counter()

    for pattern in UNSAFE_PATTERNS:
        if pattern.search(query):
            return GuardrailVerdict(
                allowed=False,
                reason="unsafe request",
                domain_similarity=0.0,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

    if not query.strip():
        return GuardrailVerdict(False, "empty query", 0.0, (time.perf_counter() - start) * 1000)

    centroids = get_store().clusters.centroids
    similarity = float(np.max(centroids @ qvec[0]))
    floor = _domain_floor()

    if not DOMAIN_FILTER_ENABLED:
        # Score it anyway — the number is shown in the trace — but let the query
        # through so the user sees what was searched before anything is refused.
        return GuardrailVerdict(
            allowed=True,
            reason=f"domain similarity {similarity:.3f} (pre-filter reporting only)",
            domain_similarity=similarity,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    allowed = similarity >= floor
    return GuardrailVerdict(
        allowed=allowed,
        reason="in domain" if allowed else f"off topic: {similarity:.3f} below floor {floor:.3f}",
        domain_similarity=similarity,
        latency_ms=(time.perf_counter() - start) * 1000,
    )
