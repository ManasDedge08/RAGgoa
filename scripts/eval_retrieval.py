"""Retrieval quality evaluation: strategy ablation and rerank weight sweep.

Gold labels come from the dataset's ``is_selected`` flag. A hit is any selected
passage belonging to the asking query, in any language, because cross-lingual
retrieval is a feature here rather than a mistake.

Run: ``python scripts/eval_retrieval.py [n_queries]``
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import CORPUS_DIR, RANDOM_SEED, REPORT_DIR  # noqa: E402
from rag.retrieval import retriever as retriever_mod  # noqa: E402
from rag.retrieval.retriever import Retriever  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def sample_queries(n: int) -> list[tuple[int, str, str]]:
    queries = pq.read_table(CORPUS_DIR / "queries.parquet").to_pydict()
    rows = list(zip(queries["query_id"], queries["lang"], queries["text"]))
    rng = random.Random(RANDOM_SEED + 7)
    rng.shuffle(rows)
    by_lang: dict[str, list] = defaultdict(list)
    for row in rows:
        by_lang[row[1]].append(row)
    per_lang = max(1, n // len(by_lang))
    out = [r for lang_rows in by_lang.values() for r in lang_rows[:per_lang]]
    rng.shuffle(out)
    return out


def score(retriever: Retriever, sample, gold, disable: set[str] | None = None) -> dict:
    disable = disable or set()
    hits5 = hits1 = 0
    mrr = 0.0
    latencies = []
    for qid, lang, text in sample:
        result = retriever.retrieve(text, top_k=5, exclude_qid=qid)
        latencies.append(result.timings_ms["total"])
        ranked = [c for c in result.candidates]
        gold_ids = gold.get(qid, set())
        rank = next((i for i, c in enumerate(ranked, 1) if c.passage_id in gold_ids), None)
        if rank:
            mrr += 1.0 / rank
            hits5 += 1
            hits1 += 1 if rank == 1 else 0
    n = len(sample)
    return {
        "recall@5": round(100 * hits5 / n, 1),
        "precision@1": round(100 * hits1 / n, 1),
        "mrr@5": round(mrr / n, 3),
        "mean_ms": round(sum(latencies) / n, 2),
    }


def main() -> None:
    sample = sample_queries(N)
    passages = pq.read_table(CORPUS_DIR / "passages.parquet").to_pydict()
    gold: dict[int, set[str]] = defaultdict(set)
    for pid, qid, sel in zip(passages["passage_id"], passages["query_id"], passages["is_selected"]):
        if sel:
            gold[qid].add(pid)

    retriever = Retriever()
    retriever.retrieve("warmup")

    report: dict[str, object] = {"n_queries": len(sample)}

    # ---- weight sweep on the cheap reranker
    base = dict(retriever_mod.RERANK_WEIGHTS)
    variants = {
        "baseline": base,
        "dense_heavy": {**base, "dense": 1.4, "bm25": 0.2, "overlap": 0.25},
        "lexical_heavy": {**base, "dense": 0.7, "bm25": 0.6, "overlap": 0.7},
        "fusion_heavy": {**base, "rrf": 1.0, "dense": 0.9},
        "dense_only": {"dense": 1.0, "bm25": 0.0, "rrf": 0.0, "overlap": 0.0, "lang": 0.0, "strategies": 0.0},
        "consensus": {**base, "dense": 1.2, "bm25": 0.25, "rrf": 0.7, "overlap": 0.3, "strategies": 0.25},
    }
    sweep = {}
    for name, weights in variants.items():
        retriever_mod.RERANK_WEIGHTS.clear()
        retriever_mod.RERANK_WEIGHTS.update(weights)
        sweep[name] = score(retriever, sample, gold)
        print(f"[weights] {name:14} {sweep[name]}")
    retriever_mod.RERANK_WEIGHTS.clear()
    retriever_mod.RERANK_WEIGHTS.update(base)
    report["rerank_weights"] = sweep

    # ---- strategy ablation: drop one strategy at a time
    ablation = {}
    original = Retriever.retrieve

    for drop in ("dense_passage", "dense_sentence", "bm25_passage", "bm25_sentence", "cluster", "none"):
        retriever_mod.DROP_STRATEGY = None if drop == "none" else drop
        ablation[drop] = score(retriever, sample, gold)
        print(f"[ablation] drop {drop:15} {ablation[drop]}")
    retriever_mod.DROP_STRATEGY = None
    report["ablation"] = ablation
    del original

    out = REPORT_DIR / "retrieval_eval.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
