"""Build the three-granularity index: sentences, passages, query clusters.

Artefacts land in ``data/index``:

* ``passages.faiss`` / ``sentences.faiss``  - dense HNSW indexes (inner product
  on L2-normalised vectors, i.e. cosine)
* ``passages.bm25`` / ``sentences.bm25``    - sparse BM25 indexes
* ``clusters.npz``                          - query-cluster centroids
* ``*.meta.parquet``                        - row order, ids, language tags

The build runs in stages and each stage is resumable, because encoding 188k
units and holding a 215 MB vector matrix alongside a loaded encoder is enough
to get the process OOM-killed on a 16 GB laptop. The encoder is released before
the FAISS and BM25 work begins, and ``--only`` runs one stage per process.

    python -m rag.index.build                  # everything, resuming what exists
    python -m rag.index.build --only passages  # one stage
    python -m rag.index.build --force          # re-encode even if vectors exist
"""

from __future__ import annotations

import argparse
import gc
import json
import time

import faiss
import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans

from ..config import (
    CORPUS_DIR,
    EMBED_DIM,
    EMBED_MODEL,
    INDEX_DIR,
    N_QUERY_CLUSTERS,
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    RANDOM_SEED,
)
from .bm25 import BM25Index

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 80
HNSW_EF_SEARCH = 64
BATCH_SIZE = 128

STAGES = ("passages", "sentences", "clusters")


def _release(model) -> None:
    """Drop the encoder and its accelerator allocations before the index work."""
    import torch

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def encode(texts: list[str], label: str, prefix: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    print(f"  encoder {EMBED_MODEL} on {model.device}", flush=True)
    start = time.perf_counter()
    vectors = model.encode(
        [prefix + t for t in texts],
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    elapsed = time.perf_counter() - start
    print(f"  encoded {len(texts)} {label} in {elapsed:.1f}s ({len(texts) / elapsed:.0f}/s)", flush=True)
    _release(model)
    return vectors


def build_hnsw(vectors: np.ndarray) -> faiss.Index:
    index = faiss.IndexHNSWFlat(EMBED_DIM, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    index.add(vectors)
    return index


def build_level(level: str, force: bool) -> int:
    table = pq.read_table(CORPUS_DIR / f"{level}.parquet")
    cols = table.to_pydict()
    id_key = "passage_id" if level == "passages" else "sentence_id"
    ids = cols[id_key]
    texts = cols["text"]
    vector_path = INDEX_DIR / f"{level}.npy"

    print(f"\n[{level}] {len(ids)} units", flush=True)
    if vector_path.exists() and not force:
        vectors = np.load(vector_path, mmap_mode="r")
        if vectors.shape == (len(ids), EMBED_DIM):
            print(f"  reusing {vector_path.name}", flush=True)
        else:
            print(f"  {vector_path.name} has shape {vectors.shape}, re-encoding", flush=True)
            vectors = None
    else:
        vectors = None

    if vectors is None:
        vectors = encode(texts, level, PASSAGE_PREFIX)
        np.save(vector_path, vectors)

    start = time.perf_counter()
    # faiss reads the buffer directly, so it needs an owned contiguous array;
    # handing it a memmap segfaults rather than raising.
    dense = np.ascontiguousarray(np.array(vectors, dtype=np.float32, copy=True))
    faiss.write_index(build_hnsw(dense), str(INDEX_DIR / f"{level}.faiss"))
    del dense
    print(f"  hnsw built in {time.perf_counter() - start:.1f}s", flush=True)
    del vectors
    gc.collect()

    start = time.perf_counter()
    BM25Index.build(ids, texts).save(INDEX_DIR / f"{level}.bm25")
    print(f"  bm25 built in {time.perf_counter() - start:.1f}s", flush=True)

    # Row order is the FAISS id space; keep the metadata aligned with it.
    pq.write_table(table, INDEX_DIR / f"{level}.meta.parquet")
    return len(ids)


def build_clusters(force: bool) -> int:
    table = pq.read_table(CORPUS_DIR / "queries.parquet")
    queries = table.to_pydict()
    print(f"\n[clusters] {len(queries['text'])} queries -> {N_QUERY_CLUSTERS} clusters", flush=True)

    vector_path = INDEX_DIR / "queries.npy"
    if vector_path.exists() and not force:
        vectors = np.array(np.load(vector_path, mmap_mode="r"), dtype=np.float32, copy=True)
        if vectors.shape != (len(queries["text"]), EMBED_DIM):
            vectors = None
    else:
        vectors = None
    if vectors is None:
        vectors = encode(queries["text"], "queries", QUERY_PREFIX)
        np.save(vector_path, vectors)

    kmeans = MiniBatchKMeans(
        n_clusters=N_QUERY_CLUSTERS,
        random_state=RANDOM_SEED,
        n_init=5,
        batch_size=1024,
    ).fit(vectors)
    centroids = kmeans.cluster_centers_.astype(np.float32)
    # Normalise so cluster matching is a cosine similarity too.
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9

    np.savez(
        INDEX_DIR / "clusters.npz",
        centroids=centroids,
        labels=kmeans.labels_.astype(np.int32),
        query_ids=np.asarray(queries["query_id"], dtype=np.int64),
        langs=np.asarray(queries["lang"]),
    )
    pq.write_table(table, INDEX_DIR / "queries.meta.parquet")
    return len(queries["text"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=STAGES, help="run a single stage")
    parser.add_argument("--force", action="store_true", help="re-encode even if vectors exist")
    args = parser.parse_args()

    stats_path = INDEX_DIR / "stats.json"
    stats: dict[str, object] = {}
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
    stats["embed_model"] = EMBED_MODEL
    stats["embed_dim"] = EMBED_DIM

    for stage in STAGES:
        if args.only and stage != args.only:
            continue
        if stage == "clusters":
            stats["queries"] = build_clusters(args.force)
            stats["clusters"] = N_QUERY_CLUSTERS
        else:
            stats[stage] = build_level(stage, args.force)

    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\nindex written to {INDEX_DIR}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
