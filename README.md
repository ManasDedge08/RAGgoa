# The Measure — voice RAG over MS MARCO-XI

Ask a question out loud in **Hindi, Tamil, Bengali or English**. An extractive
answer comes back in about ten milliseconds and is shown against a 200 ms
budget. A synthesised spoken answer follows a moment later, checked against the
retrieved passages before it is allowed to speak.

Built on [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI),
with Sarvam AI for speech in, speech out, and generation.

* **Live demo:** _deploy pending — see [Deploying](#deploying)_
* **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)
* **Latency report:** [reports/latency_report.md](reports/latency_report.md)
* **Encoder A/B:** [reports/encoder_ab.md](reports/encoder_ab.md)

---

## The honest part: why there are two tiers

A 200 ms end-to-end voice-to-generated-answer target is not reachable with a
hosted LLM in the loop. The network round trip plus sequential token decoding
puts the floor in the hundreds of milliseconds before anything worth speaking
exists. Any single number claiming otherwise is measuring something narrower
than it sounds.

So this system answers twice, and says which is which everywhere — in the UI, in
the spoken output, and in the latency report:

| | Tier 1 | Tier 2 |
| --- | --- | --- |
| What it is | best-matching sentence span from the top retrieved passage | LLM synthesis across the top passages |
| Generation | none — the answer is a substring of the corpus | `sarvam-105b-conversations`, streamed |
| Hallucination risk | zero by construction | checked against the retrieved passages before speaking |
| Measured against 200 ms | **yes** | **no** — reported as its own series |

Tier 1 is measured end to end: guardrail, query embedding, all five retrieval
strategies, fusion, reranking, span selection. Nothing is excluded to make the
number look better. Tier 2's latency is logged separately and never averaged in.

When Tier 2's output fails the grounding check, the Tier 1 span is spoken
instead and the turn is marked degraded rather than quietly substituted.

---

## What it does

**Multi-strategy retrieval.** Five strategies vote on every query — dense and
BM25 at both passage and sentence granularity, plus a query-cluster index that
matches new questions against clusters of known ones. Reciprocal rank fusion
merges their ranked lists, and dense similarity orders the survivors. That
division of labour is measured, not assumed: letting BM25 score and fusion
position into the *final* ordering dropped recall@5 from 38.5% to 30.0%,
because a lexically noisy passage can out-rank the actual answer.

**Cross-lingual by construction.** Every passage exists in all four languages,
aligned on `query_id`. Ask in Tamil and the system can answer from a
Bengali-sourced passage, in Tamil, and say so. Three routing modes: `cross`
(all languages), `strict` (only the asker's), `pivot` (only other languages —
the cross-lingual proof).

**Audible guardrails.** Confidence changes how the answer *sounds*, not just
what a badge says. High confidence states it directly. Low confidence hedges
out loud — "I found something related, but I'm not fully sure this answers it"
— before the content. Off-topic gets a spoken refusal, not silence.

**Glass-box retrieval.** While Tier 2 is still generating, the UI shows every
retrieved passage, which strategy surfaced it, its fusion rank, and its rerank
score, streaming in as the pipeline runs.

**Latency as the page's structure.** The interface hangs from a single 0–200 ms
scale. Each Tier 1 stage is drawn as a segment on it, and the naive baseline
races on the same scale in the lane below.

**A precision mode you can watch cost you.** A cross-encoder rerank over the top
10 is one toggle away. Turning it on moves recall@5 from 43.3% to 50.0% and the
Tier 1 bar from 12 ms to 159 ms at P100 — still inside the budget, and visibly
eating it. Off by default, because a slower deployment box would not have the
room.

---

## Results

Measured on an Apple M4, 1,200 MS MARCO query ids across four languages:
47,780 passages, 143,963 sentences, 160 query clusters.

<!-- BENCHMARK:BEGIN -->
| | P50 | P70 | P100 | measured against 200 ms |
| --- | --- | --- | --- | --- |
| **Tier 1** — retrieval + extractive answer | 10.12 ms | 10.31 ms | 11.67 ms | yes |
| **Tier 2** — LLM synthesis | 2.63 s | 3.07 s | 4.01 s | no, by design |
| Naive baseline — fixed chunks, one dense scan | 6.14 ms | 6.28 ms | 7.34 ms | for comparison |

100.0% of Tier 1 queries land inside the budget; P100 is 11.67 ms against a 200 ms target.

Per-stage and per-language breakdowns: [reports/latency_report.md](reports/latency_report.md).
<!-- BENCHMARK:END -->

### Retrieval quality

| Metric | Multi-strategy | Naive baseline |
| --- | --- | --- |
| query hit@5 (any passage from the asking query) | **91.3%** | 84.3% |
| strict recall@5 (the one passage marked selected) | **37.0%** | — |
| P50 latency | 10.5 ms | 6.4 ms |

Two metrics, because they answer different questions. *Query hit@5* is whether
the answer could be grounded at all. *Strict recall@5* is whether the single
passage MS MARCO marked `is_selected` made the top five — a hard metric here,
since each query's other candidate passages are also on topic and compete for
the same slots.

The naive path is faster because it does less: one brute-force scan instead of
five strategies. It also finds the right material seven points less often.
Both sit far inside the budget, so the extra milliseconds are worth spending —
and the report says so plainly rather than claiming a speed win that is not
there.

### The voice path, end to end

Measured on a real turn: Hindi speech in, spoken Hindi answer out.

| Moment | Elapsed |
| --- | --- |
| transcript returned (Sarvam `saaras:v4`) | 345 ms |
| Tier 1 answer on screen | 382 ms |
| Tier 2 synthesis complete | 3.7 s |
| spoken answer playing (`bulbul:v2`) | 4.8 s |

Speech synthesis is the second-largest cost in a turn, and the model choice
matters more than expected: on answer-length Hindi, `bulbul:v2` returns in about
1.0 s where `bulbul:v3` takes 6.0 s for the same audio. v2 is the default.

### Where the guardrails actually catch things

Honest result: **pre-retrieval domain filtering barely works on this corpus.**
"Book me a cab to the airport" scores 0.863 against the corpus question
distribution; a real corpus question scores 0.905 at the median and 0.860 at
the 5th percentile. The distributions overlap because an off-topic question is
still a well-formed question. Setting a floor tight enough to catch it would
reject real queries.

So the pre-filter no longer rejects anything by default. It scores the query,
shows the number in the trace, and lets it through; the real refusal happens
*after* retrieval, where there is evidence to judge. That gate works: the
refused band runs at 13.3% top-5 accuracy against a 41.5% base rate, and "book
me a cab" is refused there. Set `RAG_DOMAIN_FILTER=1` to enforce the pre-filter
instead.

Refusing after retrieval also makes the refusal legible. Ask "who is the prime
minister of India" and you see the closest thing the corpus had — *"India,
officially the Republic of India, is a country in South Asia"* — followed by a
decline. The system shows its work even when the work came up empty.

### When the corpus has no answer

The default is to refuse, because refusing what it cannot ground is the
behaviour the guardrails exist to produce. But a retrieval demo over a fixed
corpus that only ever says no reads as broken to anyone who asks it an ordinary
question, so there is one deliberate escape hatch.

**General-knowledge mode** — a toggle in the UI, `allow_unsourced` on the API,
off by default. When retrieval finds nothing, the model answers from its own
knowledge and the answer is labelled as such: a red-edged card in the UI, and,
because a listener never sees the card, the provenance is spoken *before* the
content — "This isn't from my sources, but from general knowledge: …". The turn
ends in the `unsourced` state, which is distinct from both `done` and
`refused`, so analytics never conflate them.

Nothing about it is checked by the grounding check. There is no retrieved
evidence to check against — that is the entire reason the mode has to announce
itself.

| Question | Default | With the toggle |
| --- | --- | --- |
| "who is the prime minister of india" | refuses, shows what it searched | "Narendra Modi… since 2014", marked unsourced |

### What the corpus can and cannot answer

The corpus is 1,200 MS MARCO questions and the passages retrieved for them:
brake rotors, HSA premiums, legal definitions, NFL records. General-knowledge
questions are declined, and that is a property of the dataset rather than the
sample size — the full 97,941-query split contains two questions mentioning a
prime minister (neither asking who one is) and none about the capital of India.
Indexing all of it would cost roughly 18 hours of encoding and 64 GB of index
and still decline the question. The interface says so up front, and the example
questions are drawn from the corpus itself.

Full numbers: [reports/retrieval_eval.json](reports/retrieval_eval.json),
[reports/guardrail.json](reports/guardrail.json),
[reports/confidence.json](reports/confidence.json).

---

## Running it locally

Requires **Python 3.12 or newer** and Node 20+. The floor is not arbitrary:
numpy and scipy publish no Windows wheels below 3.12 at the versions used here,
so pip would try to compile them from source.

One command does all of it.

**macOS / Linux**

```bash
./scripts/setup.sh            # venv, deps, corpus, index, calibration, tests
./scripts/run.sh              # API on :8000, UI on :5173
```

**Windows 11 (PowerShell)**

```powershell
.\scripts\setup.ps1          # same steps
.\scripts\run.ps1            # API on :8000, UI on :5173
```

If PowerShell refuses to run the scripts, allow local ones for the session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Windows has no GPU acceleration here, so the index build runs on CPU and takes
noticeably longer than the ~4 minutes per language quoted below. Two Windows
specifics are handled in code rather than left to the user: console output is
forced to UTF-8, because the legacy code page cannot encode Indic text and
every script would otherwise crash on `print`; and `KMP_DUPLICATE_LIB_OK` is
set before faiss loads, because torch and faiss bundle OpenMP DLLs that cannot
be symlinked together the way they are on macOS.

Both are safe to re-run and reuse whatever already exists; `--rebuild` (bash)
or `-Rebuild` (PowerShell) forces a regenerate. The steps individually, on
macOS or Linux:

```bash
# 1. Environment
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/fix_macos_openmp.py   # macOS only, see rag/__init__.py
cp .env.example .env          # add SARVAM_API_KEY (optional — see below)

# 2. Corpus and indexes (~20 min: downloads 1.4 GB, encodes 190k units)
#    Already have the parquet files? Drop them in data/dataset/ and they are
#    used automatically — see data/dataset/README.md.
.venv/bin/python scripts/prepare_data.py
.venv/bin/python -m rag.index.build
.venv/bin/python -m rag.index.build_baseline     # the naive path, for the race
.venv/bin/python scripts/calibrate.py            # guardrail + confidence thresholds

# 3. Check it works
.venv/bin/python scripts/smoke.py                # four paths, including a refusal
.venv/bin/python benchmark.py --baseline         # writes reports/latency_report.md

# 4. Serve
.venv/bin/uvicorn rag.server:app --reload        # API on :8000
cd web && npm install && npm run dev             # UI on :5173
```

**Without a Sarvam key** the system still runs end to end: retrieval, timing,
the trace, Tier 1 answers, and the guardrails are all real. Speech in, speech
out, and Tier 2 text fall back to labelled stubs, and the UI says so. This is
deliberate — the retrieval demo should never be blocked on credentials.

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `SARVAM_API_KEY` | _(empty)_ | STT, TTS and Tier 2. Empty enables stub mode. |
| `RAG_N_QUERY_IDS` | `1200` | Corpus size. Lower it to fit a smaller instance. |
| `RAG_DATASET_DIR` | `data/dataset` | Where to look for local dataset files before downloading. |
| `RAG_DATASET_SPLIT` | `validation` | `validation` (97,941 queries/language) or `train` (~980k). |
| `RAG_EMBED_MODEL` | `intfloat/multilingual-e5-small` | Query and passage encoder. |
| `SARVAM_CHAT_MODEL` | `sarvam-105b-conversations` | Tier 2 model. |
| `SARVAM_STT_MODEL` | `saaras:v4` | Speech to text. |
| `RAG_DOMAIN_FILTER` | _(off)_ | `1` enforces the pre-retrieval domain filter. |
| `RAG_ALLOW_UNSOURCED` | _(off)_ | `1` makes general-knowledge fallback the default. |
| `SARVAM_TTS_MODEL` | `bulbul:v2` | Text to speech (6x faster than v3 on answer-length text). |
| `VITE_API_BASE` | `http://127.0.0.1:8000` | Where the UI finds the API. |

---

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /ask` | Text question. Streams SSE stage events. |
| `POST /ask-audio` | Audio question (multipart). Streams SSE stage events. |
| `POST /race` | Multi-strategy vs naive baseline on the same query. |
| `GET /meta` | Corpus stats, languages, whether voice is stubbed. |
| `GET /sample-queries` | Real corpus queries per language, for the demo chips. |
| `GET /health` | Liveness. |

Every answer endpoint emits the same event vocabulary: `state`, `transcript`,
`guardrail`, `trace`, `retrieval`, `tier1`, `tier2_delta`, `tier2`, `audio`,
`error`, `done`.

---

## Deploying

**API → Render.** `render.yaml` and the `Dockerfile` build the corpus and index
into the image, so containers start without downloading anything. Set
`SARVAM_API_KEY` in the Render dashboard.

Memory is the binding constraint, so here is the measurement rather than a
guess. Serving the full 1,200-query corpus sits at **~2.2 GB RSS**; the 800-id
build scales to roughly 1.4 GB. Index vectors and the HNSW graphs are
memory-mapped, so a good part of that is evictable page cache rather than a
hard floor — but the encoder itself is several hundred megabytes and is not.

`render.yaml` is currently set to the **free plan (512 MB)**, which is expected
to be killed once the encoder loads. It is configured that way deliberately, to
find out rather than assume. If it does get OOM-killed, in order of effect:

1. `RAG_N_QUERY_IDS=250` — shrinks every index proportionally
2. `plan: standard` — 2 GB, comfortable at 800 ids

**UI → Vercel.** Deploy `web/` with `VITE_API_BASE` pointing at the Render URL.

---

## Repository map

| Path | What's in it |
| --- | --- |
| `rag/index/` | corpus indexing: BM25, HNSW, query clusters, naive baseline |
| `rag/retrieval/` | the five strategies, RRF, reranking, language routing |
| `rag/generation/` | Tier 1, Tier 2, grounding check, confidence, spoken templates |
| `rag/harness/` | the state machine and the query guardrail |
| `rag/voice/` | Sarvam client with retries, timeouts and stub mode |
| `scripts/` | data prep, evaluation, calibration, smoke test |
| `benchmark.py` | the latency suite |
| `web/` | Vite + React client |
