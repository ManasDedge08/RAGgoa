"""Loads every index artefact once and exposes them to the retrieval strategies."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from functools import lru_cache

import faiss
import numpy as np
import pyarrow.parquet as pq

from ..config import EMBED_MODEL, INDEX_DIR, QUERY_PREFIX, RERANK_MODEL
from ..index.bm25 import BM25Index


@dataclass
class Unit:
    """One retrievable unit at either sentence or passage granularity."""

    uid: str
    passage_id: str
    query_id: int
    lang: str
    is_selected: int
    text: str
    row: int


@dataclass
class Level:
    name: str
    faiss_index: faiss.Index
    bm25: BM25Index
    vectors: np.ndarray
    units: list[Unit]
    by_uid: dict[str, Unit] = field(default_factory=dict)
    rows_by_passage: dict[str, list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.by_uid = {u.uid: u for u in self.units}
        for u in self.units:
            self.rows_by_passage.setdefault(u.passage_id, []).append(u.row)


def _load_level(name: str) -> Level:
    table = pq.read_table(INDEX_DIR / f"{name}.meta.parquet").to_pydict()
    id_key = "passage_id" if name == "passages" else "sentence_id"
    units = [
        Unit(
            uid=table[id_key][i],
            passage_id=table["passage_id"][i],
            query_id=table["query_id"][i],
            lang=table["lang"][i],
            is_selected=table["is_selected"][i],
            text=table["text"][i],
            row=i,
        )
        for i in range(len(table[id_key]))
    ]
    return Level(
        name=name,
        # mmap keeps the HNSW graph and the raw vectors out of resident memory,
        # which is what lets the deployed instance stay on a small plan.
        faiss_index=faiss.read_index(str(INDEX_DIR / f"{name}.faiss"), faiss.IO_FLAG_MMAP),
        bm25=BM25Index.load(INDEX_DIR / f"{name}.bm25"),
        vectors=np.load(INDEX_DIR / f"{name}.npy", mmap_mode="r"),
        units=units,
    )


@dataclass
class ClusterIndex:
    centroids: np.ndarray
    labels: np.ndarray
    query_ids: np.ndarray
    langs: np.ndarray
    members: dict[int, list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for row, label in enumerate(self.labels):
            self.members.setdefault(int(label), []).append(row)


class Store:
    """Process-wide singleton holding indexes and the embedding model."""

    def __init__(self) -> None:
        self.passages = _load_level("passages")
        self.sentences = _load_level("sentences")

        blob = np.load(INDEX_DIR / "clusters.npz", allow_pickle=True)
        self.clusters = ClusterIndex(
            centroids=blob["centroids"],
            labels=blob["labels"],
            query_ids=blob["query_ids"],
            langs=blob["langs"],
        )
        queries = pq.read_table(INDEX_DIR / "queries.meta.parquet").to_pydict()
        self.query_rows = queries
        self.query_vectors = np.load(INDEX_DIR / "queries.npy", mmap_mode="r")

        # Gold passages per query id, used by the cluster strategy and by eval.
        self.gold_by_query: dict[int, list[str]] = {}
        for u in self.passages.units:
            if u.is_selected:
                self.gold_by_query.setdefault(u.query_id, []).append(u.passage_id)

        self.stats = json.loads((INDEX_DIR / "stats.json").read_text(encoding="utf-8"))
        self._model = None
        self._cross_encoder = None
        self._model_lock = threading.Lock()

    @property
    def model(self):
        """Query encoder, loaded lazily and pinned to CPU.

        Single-query encoding on MPS pays a fixed dispatch cost that is larger
        than the CPU forward pass at batch size 1, so the timed path stays on
        CPU. Bulk index building still uses the accelerator.
        """
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(EMBED_MODEL, device="cpu")
                    self._model.eval()
                    # Warm the graph so the first real query is not an outlier.
                    self._model.encode([QUERY_PREFIX + "warmup"], normalize_embeddings=True)
        return self._model

    @property
    def cross_encoder(self):
        """Cross-encoder for precision mode, loaded on first use.

        Kept lazy because it is several hundred megabytes and most turns never
        ask for it.
        """
        if self._cross_encoder is None:
            with self._model_lock:
                if self._cross_encoder is None:
                    from sentence_transformers import CrossEncoder

                    encoder = CrossEncoder(RERANK_MODEL, device="cpu", max_length=256)
                    encoder.predict([("warm", "up")])
                    self._cross_encoder = encoder
        return self._cross_encoder

    def encode_query(self, text: str) -> np.ndarray:
        vec = self.model.encode(
            [QUERY_PREFIX + text], normalize_embeddings=True, convert_to_numpy=True
        )
        return vec.astype(np.float32)


@lru_cache(maxsize=1)
def get_store() -> Store:
    return Store()
