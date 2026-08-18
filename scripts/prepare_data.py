"""Build the demo corpus from ai4bharat/MSMARCO-XI.

The dataset ships one parquet file per language, all of them aligned on
``query_id``. That alignment is what makes cross-lingual retrieval provable:
the same passage exists in English plus every translated language, so a Tamil
query can legitimately surface a Bengali-sourced passage.

Outputs (parquet, written to ``data/corpus``):

* ``queries.parquet``    - one row per (query_id, lang)
* ``passages.parquet``   - one row per (query_id, passage index, lang)
* ``sentences.parquet``  - passages split into sentence-level units

Run: ``python scripts/prepare_data.py``
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.index.text import tokenize  # noqa: E402
from rag.config import (  # noqa: E402
    CACHE_DIR,
    CORPUS_DIR,
    DATASET_DIR,
    DATASET_DIR_DEFAULT,
    DATASET_SPLIT,
    DATASET_LANGS,
    HF_DATASET,
    LANGUAGES,
    MIN_SENTENCE_TOKENS,
    N_QUERY_IDS,
    RANDOM_SEED,
)

# Only the columns the corpus is built from: the slices also carry source_lang,
# target_lang and meta, which are dead weight in every batch decoded.
SLICE_COLUMNS = [
    "query_id", "query_type", "passages", "query", "Answer", "Eng_Query", "Eng_Answer",
]
# Rows per decoded batch. Bounds peak memory; the slices are one row group, so
# without this the whole nested passages column is decoded at once.
BATCH_ROWS = 256

# Sentence boundaries for Latin, Devanagari (danda), and generic CJK-style stops.
_SENT_SPLIT = re.compile(r"(?<=[.!?।॥])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    """Split a passage into sentence units, merging fragments that are too short.

    Length is measured in tokens, not characters, so the same rule applies
    evenly across Latin and Indic scripts.
    """
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    merged: list[str] = []
    for part in parts:
        if merged and len(tokenize(merged[-1])) < MIN_SENTENCE_TOKENS:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged or [text.strip()]


def local_candidates(code: str) -> list[Path]:
    """Every path a hand-placed copy of one language slice may live at.

    Checked in order, so dropping files into ``data/dataset/`` needs no
    configuration at all. ``RAG_DATASET_DIR`` overrides the location, and both
    the dataset repo's own layout (``validation/hinval.parquet``) and a flat
    directory of the same files are accepted.
    """
    suffix = "val" if DATASET_SPLIT == "validation" else "train"
    filename = f"{code}{suffix}.parquet"

    roots = [Path(DATASET_DIR).expanduser()] if DATASET_DIR else [DATASET_DIR_DEFAULT]
    return [root / DATASET_SPLIT / filename for root in roots] + [root / filename for root in roots]


def language_path(code: str) -> Path:
    """Where one language's slice lives: a hand-placed copy first, else the Hub.

    Downloads are cached by ``huggingface_hub``, so calling this twice for the
    same language costs one filesystem check rather than a second download.
    """
    for candidate in local_candidates(code):
        if candidate.exists():
            print(f"  local file: {candidate}", flush=True)
            return candidate

    if DATASET_DIR:
        looked = "\n  ".join(str(c) for c in local_candidates(code))
        raise FileNotFoundError(
            f"RAG_DATASET_DIR is set but the file was not found. Looked in:\n  {looked}\n"
            "Unset RAG_DATASET_DIR to download from the Hub instead."
        )

    suffix = "val" if DATASET_SPLIT == "validation" else "train"
    print(f"  downloading {DATASET_SPLIT}/{code}{suffix}.parquet from the Hub", flush=True)
    return Path(
        hf_hub_download(
            HF_DATASET,
            f"{DATASET_SPLIT}/{code}{suffix}.parquet",
            repo_type="dataset",
            cache_dir=str(CACHE_DIR),
        )
    )


def load_language(code: str) -> pa.Table:
    """Read one language's slice in full."""
    return pq.read_table(language_path(code))


def main() -> None:
    slices = {lang: meta["file"] for lang, meta in LANGUAGES.items() if meta["file"]}

    # Pass one: the query ids only. Reading the single column keeps this to a
    # few megabytes per language, where the full slice is gigabytes once Arrow
    # decompresses it.
    id_sets = []
    for lang, code in slices.items():
        print(f"reading query ids for {LANGUAGES[lang]['name']} ...", flush=True)
        ids = pq.read_table(language_path(code), columns=["query_id"])
        id_sets.append(set(ids.column("query_id").to_pylist()))
        del ids

    # Every language slice covers the same query ids; intersect to be safe.
    shared = sorted(set.intersection(*id_sets))
    del id_sets
    print(f"shared query_ids: {len(shared)}")

    rng = random.Random(RANDOM_SEED)
    sampled = set(rng.sample(shared, min(N_QUERY_IDS, len(shared))))

    query_rows: list[dict] = []
    passage_rows: list[dict] = []
    sentence_rows: list[dict] = []
    seen_passage_text: dict[str, str] = {}

    # English is derived from the Hindi slice's English columns, which are
    # identical across slices. Emit it exactly once.
    english_done: set[int] = set()

    # Pass two: one language at a time, streamed in batches, filtered to the
    # sample before anything becomes a Python object.
    #
    # Holding every slice and calling to_pydict() on each is what needed 8 GB —
    # all 97,941 rows of all eleven languages, nested passage structs included,
    # resident at once. Measured on one Hindi slice: 5.2 GB read whole, 3.3 GB
    # with predicate pushdown, 1.0 GB streamed like this. The slices are a
    # single row group, so pushdown still has to decode the whole passages
    # column chunk; only batching bounds it.
    keep = pa.array(sorted(sampled))
    for lang, code in slices.items():
        print(f"reading {LANGUAGES[lang]['name']} ...", flush=True)
        reader = pq.ParquetFile(language_path(code))
        for batch in reader.iter_batches(batch_size=BATCH_ROWS, columns=SLICE_COLUMNS):
            batch = pa.Table.from_batches([batch])
            batch = batch.filter(pc.is_in(batch.column("query_id"), value_set=keep))
            if batch.num_rows == 0:
                continue
            cols = batch.to_pydict()
            del batch
            _emit_rows(
                cols, lang, sampled, english_done,
                query_rows, passage_rows, sentence_rows, seen_passage_text,
            )
            del cols
        del reader

    _write("queries", query_rows)
    _write("passages", passage_rows)
    _write("sentences", sentence_rows)

    langs = sorted({r["lang"] for r in passage_rows})
    print(f"\nquery_ids: {len(sampled)}  langs: {langs}")
    print(f"queries:   {len(query_rows)}")
    print(f"passages:  {len(passage_rows)}")
    print(f"sentences: {len(sentence_rows)}")


def _emit_rows(
    cols: dict,
    lang: str,
    sampled: set,
    english_done: set,
    query_rows: list[dict],
    passage_rows: list[dict],
    sentence_rows: list[dict],
    seen_passage_text: dict[str, str],
) -> None:
    """Turn one filtered batch into query, passage and sentence rows."""
    for i, qid in enumerate(cols["query_id"]):
        qtype = cols["query_type"][i]
        passages = cols["passages"][i]
        translated = passages.get("Translated_passages") or []
        english = passages.get("English_passages") or []
        selected = passages.get("is_selected") or []

        query_rows.append(
            {
                "query_id": qid,
                "lang": lang,
                "text": cols["query"][i],
                "answer": cols["Answer"][i],
                "query_type": qtype,
            }
        )

        emit_english = qid not in english_done
        if emit_english:
            english_done.add(qid)
            query_rows.append(
                {
                    "query_id": qid,
                    "lang": "eng_Latn",
                    "text": cols["Eng_Query"][i],
                    "answer": cols["Eng_Answer"][i],
                    "query_type": qtype,
                }
            )

        for p_idx, text in enumerate(translated):
            if not text or not text.strip():
                continue
            is_sel = int(selected[p_idx]) if p_idx < len(selected) else 0
            _add_passage(
                passage_rows,
                sentence_rows,
                seen_passage_text,
                qid,
                p_idx,
                lang,
                text,
                is_sel,
            )

        if emit_english:
            for p_idx, text in enumerate(english):
                if not text or not text.strip():
                    continue
                is_sel = int(selected[p_idx]) if p_idx < len(selected) else 0
                _add_passage(
                    passage_rows,
                    sentence_rows,
                    seen_passage_text,
                    qid,
                    p_idx,
                    "eng_Latn",
                    text,
                    is_sel,
                )


def _add_passage(
    passage_rows: list[dict],
    sentence_rows: list[dict],
    seen: dict[str, str],
    qid: int,
    p_idx: int,
    lang: str,
    text: str,
    is_selected: int,
) -> None:
    text = text.strip()
    key = f"{lang}:{hash(text)}"
    if key in seen:
        return
    pid = f"{qid}:{p_idx}:{lang}"
    seen[key] = pid
    passage_rows.append(
        {
            "passage_id": pid,
            "query_id": qid,
            "p_idx": p_idx,
            "lang": lang,
            "is_selected": is_selected,
            "text": text,
        }
    )
    for s_idx, sent in enumerate(split_sentences(text)):
        sentence_rows.append(
            {
                "sentence_id": f"{pid}#{s_idx}",
                "passage_id": pid,
                "query_id": qid,
                "lang": lang,
                "is_selected": is_selected,
                "text": sent,
            }
        )


def _write(name: str, rows: list[dict]) -> None:
    table = pa.Table.from_pylist(rows)
    out = CORPUS_DIR / f"{name}.parquet"
    pq.write_table(table, out)
    print(f"wrote {out} ({table.num_rows} rows)")


if __name__ == "__main__":
    main()
