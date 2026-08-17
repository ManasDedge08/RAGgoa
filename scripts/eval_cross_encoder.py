"""Does a cross-encoder rerank earn its place inside the 200 ms budget?

Tier 1 runs at ~10 ms P50, so there is real headroom. A cross-encoder scores
each (query, passage) pair jointly instead of comparing two independent
vectors, which is usually a large precision gain — but it costs a forward pass
per candidate, and that cost scales with shortlist depth.

This measures both sides at several depths on the same queries, so the choice
is made on numbers: quality gain against latency spent.

Run: ``python scripts/eval_cross_encoder.py [n_queries]``
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import CORPUS_DIR, RANDOM_SEED, REPORT_DIR, RERANK_MODEL  # noqa: E402
from rag.retrieval.retriever import Retriever  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
DEPTHS = (0, 5, 10, 20)
MAX_PASSAGE_CHARS = 500


def sample_queries(n: int) -> list[tuple[int, str, str]]:
    table = pq.read_table(CORPUS_DIR / "queries.parquet").to_pydict()
    rows = list(zip(table["query_id"], table["lang"], table["text"]))
    rng = random.Random(RANDOM_SEED + 13)
    rng.shuffle(rows)
    by_lang: dict[str, list] = defaultdict(list)
    for row in rows:
        by_lang[row[1]].append(row)
    per_lang = max(1, n // len(by_lang))
    return [r for lang_rows in by_lang.values() for r in lang_rows[:per_lang]]


def main() -> None:
    from sentence_transformers import CrossEncoder

    sample = sample_queries(N)
    passages = pq.read_table(CORPUS_DIR / "passages.parquet").to_pydict()
    gold: dict[int, set[str]] = defaultdict(set)
    for qid, p_idx, sel in zip(passages["query_id"], passages["p_idx"], passages["is_selected"]):
        if sel:
            gold[qid].add(f"{qid}:{p_idx}")

    retriever = Retriever()
    retriever.retrieve("warmup")

    print(f"cross-encoder: {RERANK_MODEL} (CPU)")
    encoder = CrossEncoder(RERANK_MODEL, device="cpu", max_length=256)
    encoder.predict([("warm", "up")])  # exclude graph construction from timings

    # Retrieve once; every depth reranks the same candidate lists.
    retrieved = []
    for qid, lang, text in sample:
        result = retriever.retrieve(text, top_k=20, exclude_qid=qid)
        retrieved.append((qid, text, result))

    report: dict[str, object] = {"model": RERANK_MODEL, "n_queries": len(sample), "depths": {}}

    for depth in DEPTHS:
        hits5 = hits1 = 0
        rerank_ms: list[float] = []

        for qid, text, result in retrieved:
            candidates = list(result.candidates)
            if depth:
                shortlist = candidates[:depth]
                pairs = [(text, c.text[:MAX_PASSAGE_CHARS]) for c in shortlist]
                start = time.perf_counter()
                scores = encoder.predict(pairs, batch_size=depth, show_progress_bar=False)
                rerank_ms.append((time.perf_counter() - start) * 1000)
                order = np.argsort(-np.asarray(scores))
                candidates = [shortlist[i] for i in order] + candidates[depth:]

            top5 = [c.group_id for c in candidates[:5]]
            if top5 and top5[0] in gold[qid]:
                hits1 += 1
            if set(top5) & gold[qid]:
                hits5 += 1

        n = len(sample)
        entry = {
            "recall@5": round(100 * hits5 / n, 1),
            "precision@1": round(100 * hits1 / n, 1),
            "rerank_p50_ms": round(float(np.percentile(rerank_ms, 50)), 1) if rerank_ms else 0.0,
            "rerank_p100_ms": round(float(np.percentile(rerank_ms, 100)), 1) if rerank_ms else 0.0,
        }
        # Tier 1 without a cross-encoder measured 10.5 ms P50 / 13.9 ms P100.
        entry["tier1_p50_ms_estimate"] = round(10.5 + entry["rerank_p50_ms"], 1)
        entry["tier1_p100_ms_estimate"] = round(13.9 + entry["rerank_p100_ms"], 1)
        entry["within_200ms"] = entry["tier1_p100_ms_estimate"] <= 200
        label = "no cross-encoder" if depth == 0 else f"top {depth}"
        report["depths"][label] = entry
        print(f"  {label:18} recall@5 {entry['recall@5']:5.1f}  P@1 {entry['precision@1']:5.1f}  "
              f"rerank P50 {entry['rerank_p50_ms']:7.1f} ms  Tier1 P100 est "
              f"{entry['tier1_p100_ms_estimate']:7.1f} ms  {'fits' if entry['within_200ms'] else 'OVER BUDGET'}")

    out = REPORT_DIR / "cross_encoder_eval.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
