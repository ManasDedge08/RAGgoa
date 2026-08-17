"""Quick Tier 1 retrieval latency probe (step 2 gate, before voice or LLM).

Runs a varied multilingual query sample through the retriever and prints the
per-stage breakdown plus P50/P70/P100 so the 200 ms decision is made on real
numbers rather than hope.
"""

from __future__ import annotations

import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import CORPUS_DIR, RANDOM_SEED, TIER1_TARGET_MS  # noqa: E402
from rag.retrieval.retriever import Retriever  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 120


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if p >= 100:
        return ordered[-1]
    k = (len(ordered) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def main() -> None:
    queries = pq.read_table(CORPUS_DIR / "queries.parquet").to_pydict()
    rows = list(zip(queries["query_id"], queries["lang"], queries["text"]))
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(rows)

    by_lang: dict[str, list[tuple]] = defaultdict(list)
    for row in rows:
        by_lang[row[1]].append(row)
    per_lang = max(1, N // len(by_lang))
    sample = [r for lang_rows in by_lang.values() for r in lang_rows[:per_lang]]
    rng.shuffle(sample)

    retriever = Retriever()
    retriever.retrieve("warmup query")  # exclude cold start from the numbers

    stage_totals: dict[str, list[float]] = defaultdict(list)
    totals: list[float] = []
    hits = 0

    for qid, lang, text in sample:
        result = retriever.retrieve(text)
        totals.append(result.timings_ms["total"])
        for stage, ms in result.timings_ms.items():
            stage_totals[stage].append(ms)
        gold = {c.passage_id for c in result.candidates if c.query_id == qid}
        hits += 1 if gold else 0

    print(f"queries: {len(sample)}  (langs: {sorted(by_lang)})")
    print("\nper-stage mean ms")
    for stage in ("detect", "embed", "dense", "bm25", "cluster", "fusion", "rerank", "total"):
        vals = stage_totals[stage]
        print(f"  {stage:9} {statistics.mean(vals):7.2f}   p95 {pct(vals, 95):7.2f}")

    print(f"\nTier 1 latency  P50 {pct(totals, 50):.1f} ms | P70 {pct(totals, 70):.1f} ms | P100 {pct(totals, 100):.1f} ms")
    print(f"target {TIER1_TARGET_MS:.0f} ms -> {'PASS' if pct(totals, 100) < TIER1_TARGET_MS else 'OVER at P100'}")
    print(f"top-{len(sample and result.candidates)} recall of the asking query's own passages: {100 * hits / len(sample):.1f}%")


if __name__ == "__main__":
    main()
