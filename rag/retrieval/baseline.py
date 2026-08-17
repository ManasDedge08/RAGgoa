"""The obvious approach, built to be raced against.

This is what a first-pass RAG looks like: glue the source documents together,
cut them into fixed 512-character chunks, embed them, and do one brute-force
nearest-neighbour lookup. No sparse signal, no multi-granularity index, no
fusion, no reranking, no language routing.

It exists so the multi-strategy pipeline's latency and quality are stated as a
comparison against something real, on the same machine and the same corpus,
rather than as an unanchored number.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ..config import CHUNK_CHARS, INDEX_DIR, QUERY_PREFIX
from .store import Store

BASELINE_VECTORS = INDEX_DIR / "baseline.npy"
BASELINE_META = INDEX_DIR / "baseline.meta.parquet"


def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Fixed-width character chunking, boundaries ignored on purpose."""
    return [text[i : i + size] for i in range(0, len(text), size)] or [text]


class NaiveRetriever:
    def __init__(self, store: Store) -> None:
        self.store = store
        if not BASELINE_VECTORS.exists():
            raise FileNotFoundError(
                f"{BASELINE_VECTORS} missing - run `python -m rag.index.build_baseline` first"
            )
        self.vectors = np.load(BASELINE_VECTORS)
        meta = pq.read_table(BASELINE_META).to_pydict()
        self.texts = meta["text"]
        self.langs = meta["lang"]
        self.query_ids = meta["query_id"]

    def retrieve(self, query: str, top_k: int = 5) -> dict:
        stages: dict[str, float] = {}
        start = time.perf_counter()

        t0 = time.perf_counter()
        qvec = self.store.model.encode(
            [QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        stages["embed"] = (time.perf_counter() - t0) * 1000

        # Brute-force scan of every chunk: the naive default.
        t0 = time.perf_counter()
        scores = self.vectors @ qvec[0]
        top = np.argpartition(-scores, top_k)[:top_k]
        top = top[np.argsort(-scores[top])]
        stages["search"] = (time.perf_counter() - t0) * 1000

        return {
            "total_ms": (time.perf_counter() - start) * 1000,
            "stages_ms": {k: round(v, 2) for k, v in stages.items()},
            "top": [
                {
                    "chunk_index": int(i),
                    "lang": self.langs[int(i)],
                    "query_id": int(self.query_ids[int(i)]),
                    "score": float(scores[int(i)]),
                    "snippet": self.texts[int(i)][:180],
                }
                for i in top
            ],
        }


def artefacts_exist() -> bool:
    return Path(BASELINE_VECTORS).exists() and Path(BASELINE_META).exists()
