"""Calibrate the guardrail floor and the confidence tier thresholds.

Both are decision boundaries, so both are set from measured distributions
rather than guessed. Results are written to ``reports/guardrail.json`` and
``reports/confidence.json``, which the runtime reads on start.

* **Guardrail floor** - a cheap pre-retrieval filter. Measured separation on
  this corpus is weak, because an off-topic question is still a well-formed
  question; the floor is therefore permissive and the number is reported rather
  than dressed up.
* **Confidence thresholds** - the real gate, applied after retrieval where
  there is evidence to judge. ``high`` is where the top hit is usually right;
  ``refuse`` is where retrieval has genuinely returned nothing usable.

Run after the index is built: ``python scripts/calibrate.py``
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import CORPUS_DIR, RANDOM_SEED, REPORT_DIR  # noqa: E402
from rag.generation.confidence import score_retrieval  # noqa: E402
from rag.harness.guardrail import check_query  # noqa: E402
from rag.retrieval.retriever import Retriever  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 240

# Deliberately outside the corpus: personal, local, temporal, operational.
# Written per language so the floor is not tuned to English alone.
OFF_TOPIC = [
    "what is the weather in Goa tomorrow",
    "book me a cab to the airport at 6pm",
    "write a python function that reverses a linked list",
    "what did I have for lunch yesterday",
    "send an email to my manager about the leave request",
    "who won the cricket match last night",
    "गोवा में कल मौसम कैसा रहेगा",
    "मेरे लिए हवाई अड्डे तक टैक्सी बुक करो",
    "मुझे कल की मीटिंग याद दिलाना",
    "நாளை வானிலை எப்படி இருக்கும்",
    "எனக்கு விமான நிலையத்திற்கு டாக்ஸி பதிவு செய்",
    "இந்த குறியீட்டில் உள்ள பிழையை சரிசெய்",
    "আগামীকাল আবহাওয়া কেমন থাকবে",
    "আমার জন্য একটি ট্যাক্সি বুক করো",
    "গতকালের সভার নোট পাঠাও",
]


def sample_queries(n: int) -> list[tuple[int, str, str]]:
    table = pq.read_table(CORPUS_DIR / "queries.parquet").to_pydict()
    rows = list(zip(table["query_id"], table["lang"], table["text"]))
    rng = random.Random(RANDOM_SEED + 99)
    rng.shuffle(rows)
    by_lang: dict[str, list] = defaultdict(list)
    for row in rows:
        by_lang[row[1]].append(row)
    per_lang = max(1, n // len(by_lang))
    return [r for lang_rows in by_lang.values() for r in lang_rows[:per_lang]]


def main() -> None:
    retriever = Retriever()
    retriever.retrieve("warmup")
    sample = sample_queries(N)

    passages = pq.read_table(CORPUS_DIR / "passages.parquet").to_pydict()
    gold: dict[int, set[str]] = defaultdict(set)
    for pid, qid, sel in zip(passages["passage_id"], passages["query_id"], passages["is_selected"]):
        if sel:
            gold[qid].add(pid)

    # ------------------------------------------------------------ guardrail ---
    in_domain = []
    for _, _, text in sample:
        qvec = retriever.store.encode_query(text)
        in_domain.append(check_query(text, qvec).domain_similarity)
    off_domain = []
    for text in OFF_TOPIC:
        qvec = retriever.store.encode_query(text)
        off_domain.append(check_query(text, qvec).domain_similarity)

    # A 5% false-reject floor turned out to sit inside the off-topic
    # distribution: on a 1,200-query slice of MS MARCO, a well-formed question
    # the corpus simply does not cover looks statistically like one it does.
    # This stage is therefore a cheap pre-filter set to reject ~1% of real
    # queries, and the genuine "no evidence" refusal happens after retrieval,
    # where there is actual evidence to judge. Both numbers are reported below.
    floor = float(np.percentile(in_domain, 1))
    false_reject = 100 * sum(1 for s in in_domain if s < floor) / len(in_domain)
    caught = 100 * sum(1 for s in off_domain if s < floor) / len(off_domain)

    guardrail = {
        "domain_floor": round(floor, 4),
        "in_domain": {
            "n": len(in_domain),
            "p5": round(float(np.percentile(in_domain, 5)), 4),
            "p50": round(float(np.median(in_domain)), 4),
        },
        "off_topic": {
            "n": len(off_domain),
            "p50": round(float(np.median(off_domain)), 4),
            "p95": round(float(np.percentile(off_domain, 95)), 4),
        },
        "false_reject_pct": round(false_reject, 1),
        "off_topic_caught_pct": round(caught, 1),
        "note": (
            "Pre-retrieval domain similarity separates weakly on this corpus: an "
            "off-topic question is still a well-formed question. This stage catches "
            "the egregious cases cheaply; the evidence-based refusal is the "
            "confidence tier, calibrated in confidence.json."
        ),
    }
    (REPORT_DIR / "guardrail.json").write_text(json.dumps(guardrail, indent=2))
    print(json.dumps(guardrail, indent=2))

    # ----------------------------------------------------------- confidence ---
    hits: list[tuple[float, bool, bool]] = []  # score, top1 correct, top5 correct
    for qid, _, text in sample:
        result = retriever.retrieve(text, exclude_qid=qid)
        confidence = score_retrieval(result)
        ids = [c.passage_id for c in result.candidates]
        hits.append(
            (
                confidence.score,
                bool(ids and ids[0] in gold[qid]),
                any(pid in gold[qid] for pid in ids),
            )
        )

    scores = np.array([h[0] for h in hits])
    top1 = np.array([h[1] for h in hits])
    top5 = np.array([h[2] for h in hits])

    def accuracy_above(threshold: float, labels: np.ndarray) -> float:
        mask = scores >= threshold
        return float(labels[mask].mean()) if mask.any() else 0.0

    def accuracy_below(threshold: float, labels: np.ndarray) -> float:
        mask = scores < threshold
        return float(labels[mask].mean()) if mask.any() else 0.0

    base_top1 = float(top1.mean())
    base_top5 = float(top5.mean())
    candidates = np.quantile(scores, np.linspace(0.05, 0.95, 37))

    # "high" — speak without hedging. The lowest score at which top-1 accuracy
    # is clearly better than the base rate, so the confident voice means
    # something rather than being the default.
    high = next(
        (float(t) for t in candidates if accuracy_above(t, top1) >= base_top1 * 1.5),
        float(np.quantile(scores, 0.80)),
    )

    # "refuse" — say nothing was found. Only where retrieval has genuinely
    # failed: the highest threshold at which the queries below it still almost
    # never contain a right answer. Anchored to the bottom of the distribution,
    # because refusing a question the system could have answered is a worse
    # failure than hedging one it got right. An earlier version put this at the
    # 60th percentile of failures, which sat on the median and refused half of
    # all valid questions.
    refuse_ceiling = float(np.quantile(scores, 0.15))
    low = next(
        (
            float(t)
            for t in reversed(candidates)
            if t <= refuse_ceiling and accuracy_below(t, top5) <= base_top5 * 0.34
        ),
        float(np.quantile(scores, 0.05)),
    )
    low = min(low, high - 0.05)

    confidence_report = {
        "thresholds": {"high": round(high, 3), "low": round(low, 3)},
        "base_rates": {
            "top1_accuracy": round(base_top1, 3),
            "top5_accuracy": round(float(top5.mean()), 3),
        },
        "accuracy_by_tier": {
            "high_top1": round(accuracy_above(high, top1), 3),
            "answered_top5": round(accuracy_above(low, top5), 3),
            "refused_top5": round(accuracy_below(low, top5), 3),
        },
        "refused_pct": round(100 * float((scores < low).mean()), 1),
        "high_pct": round(100 * float((scores >= high).mean()), 1),
        "score_distribution": {
            "p10": round(float(np.percentile(scores, 10)), 3),
            "p50": round(float(np.percentile(scores, 50)), 3),
            "p90": round(float(np.percentile(scores, 90)), 3),
        },
        "n": len(hits),
    }
    (REPORT_DIR / "confidence.json").write_text(json.dumps(confidence_report, indent=2))
    print(json.dumps(confidence_report, indent=2))


if __name__ == "__main__":
    main()
