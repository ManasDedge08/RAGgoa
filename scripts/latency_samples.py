"""Record per-query latencies, not just their percentiles.

benchmark.py reports P50/P70/P100, which is what the target is stated in, but a
distribution shows things a percentile hides: whether the tail is a few
outliers or a fat shoulder, and whether languages differ in shape rather than
only in median. Writes reports/latency_samples.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark import stratified_sample  # noqa: E402
from rag.config import REPORT_DIR  # noqa: E402
from rag.generation import tier1 as tier1_mod  # noqa: E402
from rag.generation.confidence import score_retrieval  # noqa: E402
from rag.harness.guardrail import check_query  # noqa: E402
from rag.retrieval.baseline import NaiveRetriever, artefacts_exist  # noqa: E402
from rag.retrieval.retriever import Retriever  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def timed_tier1(retriever: Retriever, text: str, cross_encode: bool) -> tuple[float, dict]:
    t0 = time.perf_counter()
    qvec = retriever.store.encode_query(text)
    embed_ms = (time.perf_counter() - t0) * 1000
    guard = check_query(text, qvec)
    result = retriever.retrieve(text, qvec=qvec, cross_encode=cross_encode)
    answer = tier1_mod.build(result, score_retrieval(result))
    total = embed_ms + guard.latency_ms + result.timings_ms["total"] + answer.latency_ms
    stages = {k: v for k, v in result.timings_ms.items() if k not in ("embed", "total")}
    stages["embed"] = embed_ms
    stages["guardrail"] = guard.latency_ms
    return total, stages


def main() -> None:
    sample = stratified_sample(N)
    retriever = Retriever()
    retriever.retrieve("warmup")

    rows = []
    for qid, lang, text in sample:
        total, stages = timed_tier1(retriever, text, cross_encode=False)
        rows.append({"query_id": qid, "lang": lang, "tier1_ms": round(total, 3),
                     "stages": {k: round(v, 3) for k, v in stages.items()}})
    print(f"tier1: {len(rows)} queries")

    retriever.retrieve("warmup", cross_encode=True)
    for row, (_, _, text) in zip(rows, sample):
        row["precision_ms"] = round(timed_tier1(retriever, text, cross_encode=True)[0], 3)
    print("precision mode done")

    if artefacts_exist():
        naive = NaiveRetriever(retriever.store)
        naive.retrieve("warmup")
        for row, (_, _, text) in zip(rows, sample):
            row["naive_ms"] = round(naive.retrieve(text)["total_ms"], 3)
        print("naive baseline done")

    out = REPORT_DIR / "latency_samples.json"
    out.write_text(json.dumps({"n": len(rows), "samples": rows}, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
