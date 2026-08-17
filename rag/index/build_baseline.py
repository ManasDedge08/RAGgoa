"""Build the naive baseline index used for the side-by-side race.

Same corpus, same embedding model, deliberately naive structure: documents are
concatenated per query and language, cut into fixed-width chunks, embedded, and
stored as one flat matrix that gets scanned exhaustively at query time.
"""

from __future__ import annotations

import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

from ..config import CHUNK_CHARS, CORPUS_DIR, EMBED_MODEL, PASSAGE_PREFIX
from ..retrieval.baseline import BASELINE_META, BASELINE_VECTORS, chunk_text


def main() -> None:
    passages = pq.read_table(CORPUS_DIR / "passages.parquet").to_pydict()

    # Naive ingestion: everything for one query and language is one document.
    docs: dict[tuple[int, str], list[str]] = {}
    for qid, lang, text in zip(passages["query_id"], passages["lang"], passages["text"]):
        docs.setdefault((qid, lang), []).append(text)

    rows = []
    for (qid, lang), parts in docs.items():
        for i, chunk in enumerate(chunk_text(" ".join(parts))):
            rows.append({"chunk_id": f"{qid}:{lang}:{i}", "query_id": qid, "lang": lang, "text": chunk})

    print(f"baseline chunks: {len(rows)} (fixed {CHUNK_CHARS} chars, no overlap)")
    model = SentenceTransformer(EMBED_MODEL)
    start = time.perf_counter()
    vectors = model.encode(
        [PASSAGE_PREFIX + r["text"] for r in rows],
        batch_size=256,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    print(f"encoded in {time.perf_counter() - start:.0f}s")

    np.save(BASELINE_VECTORS, vectors)
    pq.write_table(pa.Table.from_pylist(rows), BASELINE_META)
    print(f"wrote {BASELINE_VECTORS} and {BASELINE_META}")


if __name__ == "__main__":
    main()
