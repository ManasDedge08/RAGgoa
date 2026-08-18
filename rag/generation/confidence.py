"""Confidence scoring over a retrieval result.

Three signals, all already computed inside the Tier 1 budget:

* ``similarity`` - dense cosine of the top candidate. E5 similarities are
  compressed into a narrow band, so the raw value is rescaled before use.
* ``margin``     - how far the top candidate sits above the rest of the final
  list. A flat list means the retriever could not discriminate.
* ``agreement``  - how many of the five strategies voted for the top candidate.

The thresholds are calibrated against gold labels by
``scripts/calibrate_confidence.py`` and stored in ``reports/confidence.json``;
the values here are the fallback when that file is absent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..config import REPORT_DIR
from ..retrieval.retriever import RetrievalResult

# E5 cosine similarities for this corpus sit roughly in [0.72, 0.92]; rescale
# that band to [0, 1] so the score uses its full range.
SIM_FLOOR = 0.74
SIM_CEIL = 0.90

WEIGHTS = {"similarity": 0.55, "margin": 0.25, "agreement": 0.20}
DEFAULT_THRESHOLDS = {"high": 0.62, "low": 0.34}


def _thresholds() -> dict[str, float]:
    path = REPORT_DIR / "confidence.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))["thresholds"]
        except (KeyError, json.JSONDecodeError):
            pass
    return DEFAULT_THRESHOLDS


@dataclass(frozen=True)
class Confidence:
    score: float
    tier: str  # "high" | "low" | "refuse"
    similarity: float
    margin: float
    agreement: float

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 3),
            "tier": self.tier,
            "similarity": round(self.similarity, 3),
            "margin": round(self.margin, 3),
            "agreement": round(self.agreement, 3),
        }


def score_retrieval(result: RetrievalResult) -> Confidence:
    cands = result.candidates
    if not cands:
        return Confidence(0.0, "refuse", 0.0, 0.0, 0.0)

    top = cands[0]
    sim_raw = top.raw_scores.get("dense_final", 0.0)
    similarity = max(0.0, min(1.0, (sim_raw - SIM_FLOOR) / (SIM_CEIL - SIM_FLOOR)))

    if len(cands) > 1:
        rest = sum(c.rerank_score for c in cands[1:]) / (len(cands) - 1)
        spread = top.rerank_score - rest
        margin = max(0.0, min(1.0, spread / 0.30))
    else:
        margin = 0.5

    agreement = len(top.strategies) / 5.0

    score = (
        WEIGHTS["similarity"] * similarity
        + WEIGHTS["margin"] * margin
        + WEIGHTS["agreement"] * agreement
    )
    thresholds = _thresholds()
    if score >= thresholds["high"]:
        tier = "high"
    elif score >= thresholds["low"]:
        tier = "low"
    else:
        tier = "refuse"
    return Confidence(score, tier, similarity, margin, agreement)
