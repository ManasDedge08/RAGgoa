# Drop MSMARCO-XI files here

Put `.parquet` files from
[`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
in this directory and `scripts/prepare_data.py` uses them automatically. No
environment variable, no flag. Anything not found here is downloaded from the
Hub instead, so a partial set is fine.

Either layout works:

```
data/dataset/validation/hinval.parquet     # the dataset repo's own layout
data/dataset/hinval.parquet                # or flat, same filenames
```

## Which files

Filenames are `<code>val.parquet` for the validation split and
`<code>train.parquet` for train. Only the languages listed in `LANGUAGES` in
`rag/config.py` are read — adding a file here does nothing until that language
is registered there too.

| Language | File | Size | Sarvam voice |
| --- | --- | --- | --- |
| Hindi | `hinval.parquet` | 440 MB | speech in + out |
| Tamil | `tamval.parquet` | 470 MB | speech in + out |
| Bengali | `benval.parquet` | 441 MB | speech in + out |
| Kannada | `kanval.parquet` | 483 MB | speech in + out |
| Malayalam | `malval.parquet` | 494 MB | speech in + out |
| Marathi | `marval.parquet` | 474 MB | speech in + out |
| Gujarati | `gujval.parquet` | 461 MB | speech in + out |
| Telugu | `telval.parquet` | 474 MB | speech in + out |
| Punjabi | `panval.parquet` | 460 MB | speech in + out |
| Odia | `orival.parquet` | 467 MB | speech in + out |
| Assamese | `asmval.parquet` | — | speech in only |
| Nepali | `nepval.parquet` | — | speech in only |
| Sanskrit | `sanval.parquet` | — | speech in only |
| Urdu | `urdval.parquet` | — | speech in only |

English needs no file: it comes from the `Eng_Query`, `Eng_Answer` and
`English_passages` columns present inside every language slice.

The train split is the same shape at roughly 10x the size (~3.8 GB per
language, ~980k queries). Set `RAG_DATASET_SPLIT=train` to use it. It is the
same kind of long-tail web query, so it widens coverage without adding
general-knowledge facts.

## Somewhere else on disk

```bash
RAG_DATASET_DIR=/Volumes/external/MSMARCO-XI python scripts/prepare_data.py
```

Same two layouts are accepted under that root. When `RAG_DATASET_DIR` is set
and a file is missing, the build fails with the paths it tried rather than
silently downloading.

## After adding files

```bash
python scripts/prepare_data.py      # rebuild the corpus
python -m rag.index.build --force   # re-encode (~4 min per language)
python scripts/calibrate.py         # thresholds shift with the corpus
```

Files in this directory are git-ignored.
