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
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.index.text import tokenize  # noqa: E402
from rag.config import (  # noqa: E402
    CACHE_DIR,
    CORPUS_DIR,
    DATASET_DIR,
    DATASET_SPLIT,
    DATASET_LANGS,
    HF_DATASET,
    LANGUAGES,
    MIN_SENTENCE_TOKENS,
    N_QUERY_IDS,
    RANDOM_SEED,
)

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


def load_language(code: str) -> pa.Table:
    """Read one language's slice, from a local copy if one is configured.

    Set ``RAG_DATASET_DIR`` to a directory holding the dataset files and nothing
    is fetched from the Hub. Both the repo layout (``validation/hinval.parquet``)
    and a flat directory of the same files are accepted.
    """
    suffix = "val" if DATASET_SPLIT == "validation" else "train"
    filename = f"{code}{suffix}.parquet"

    if DATASET_DIR:
        root = Path(DATASET_DIR).expanduser()
        for candidate in (root / DATASET_SPLIT / filename, root / filename):
            if candidate.exists():
                print(f"  local: {candidate}", flush=True)
                return pq.read_table(candidate)
        raise FileNotFoundError(
            f"{filename} not found under {root} (looked in {root / DATASET_SPLIT} and {root}). "
            "Unset RAG_DATASET_DIR to download from the Hub instead."
        )

    path = hf_hub_download(
        HF_DATASET,
        f"{DATASET_SPLIT}/{filename}",
        repo_type="dataset",
        cache_dir=str(CACHE_DIR),
    )
    return pq.read_table(path)


def main() -> None:
    tables = {}
    for lang, meta in LANGUAGES.items():
        if meta["file"]:
            print(f"reading {meta['name']} ...", flush=True)
            tables[lang] = load_language(meta["file"])

    # Every language slice covers the same query ids; intersect to be safe.
    id_sets = [set(t.column("query_id").to_pylist()) for t in tables.values()]
    shared = sorted(set.intersection(*id_sets))
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

    for lang, table in tables.items():
        cols = table.to_pydict()
        for i, qid in enumerate(cols["query_id"]):
            if qid not in sampled:
                continue
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

    _write("queries", query_rows)
    _write("passages", passage_rows)
    _write("sentences", sentence_rows)

    langs = sorted({r["lang"] for r in passage_rows})
    print(f"\nquery_ids: {len(sampled)}  langs: {langs}")
    print(f"queries:   {len(query_rows)}")
    print(f"passages:  {len(passage_rows)}")
    print(f"sentences: {len(sentence_rows)}")


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
