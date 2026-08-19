# Peoples — voice RAG over MS MARCO-XI

Ask a question out loud in **any of eleven Indian languages** — English, Hindi,
Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi or Odia.
An extractive answer comes back inside a 200 ms budget and is shown against it
— 99 ms median on the deployed instance, of which retrieval is about 19 ms, the
relevance gate most of the rest, and pulling the span out of the passage costs
under a hundredth of a millisecond. A synthesised spoken answer follows a
moment later, checked against the retrieved passages before it is allowed to
speak, and is never folded into that number.

Built on [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI),
with Sarvam AI for speech in, speech out, and generation.

* **Live demo:** [peoples-hhgoa.me](https://peoples-hhgoa.me)
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

**Cross-lingual by construction.** Every passage exists in all eleven
languages, aligned on `query_id`. Ask in Tamil and the system can answer from a
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
| **Tier 1** — retrieval + extractive answer | 11.64 ms | 12.4 ms | 25.62 ms | yes |
| **Tier 2** — LLM synthesis | 2.77 s | 2.92 s | 16.66 s | no, by design |
| Naive baseline — fixed chunks, one dense scan | 6.33 ms | 6.44 ms | 7.52 ms | for comparison |

100.0% of Tier 1 queries land inside the budget; P100 is 25.62 ms against a 200 ms target.

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

This is deployed, and these are the machine and the steps it actually runs on.

**Where it runs.** An Azure `Standard_B4as_v2` — 4 vCPU, 16 GiB — in Central
India, Ubuntu 24.04, serving the full eleven-language corpus with the relevance
gate on. `ufw` allows 22, 80 and 443 only. uvicorn binds to loopback and
systemd keeps it up (`TimeoutStartSec=300`, because startup loads the encoder
and the gate's cross-encoder before the first request); Caddy terminates TLS in
front of it and proxies with `flush_interval -1`, without which the
server-sent-event trace arrives in one lump at the end instead of streaming.
Region matters more than it looks: the server calls Sarvam for speech and Tier
2, so hosting outside India adds a round trip to every one of those.

The index is not rebuilt on the server. The 2 GB of `data/index/` is copied up
and the checkout points at it, which takes the deploy from a two-hour build to a
file transfer.

`SARVAM_API_KEY` goes in `.env` on the box, mode 600. The demo endpoints are
rate limited per client IP — 12 a minute, 120 an hour — because the URL is
public, the API has no authentication, and every answered turn spends speech and
generation credit. Set `RAG_TRUST_PROXY=1` when something terminates TLS in
front, or every caller is counted as 127.0.0.1 and one visitor locks out the
rest.

**A note on `render.yaml`.** It is kept as a documented dead end rather than a
recommendation: Render's free plan cannot run this at any corpus size, and the
file says why. See the memory table below.

Memory is the binding constraint, so here is the measurement rather than a
guess — staged, peak RSS, taken by loading the process one piece at a time:

| stage | peak RSS | scales with corpus? |
|---|---|---|
| bare interpreter | 15 MB | no |
| + torch imported | 202 MB | no |
| + sentence-transformers | 494 MB | no |
| + encoder resident | 1,034 MB | no |
| + cross-encoder (relevance gate) | 1,545 MB | no |
| + full 1,200-id index, 11 languages | **3,547 MB** | yes |

The shape matters more than the total: **1,545 MB is spent before the corpus
exists at all.** Four fifths of the floor is the ML stack and two models, and
`multilingual-e5-small` is the same 471 MB whether it serves one language or
eleven — most of those 117.7M parameters are a 250k-token multilingual
vocabulary, which is the thing that makes it multilingual. Only the last row
responds to `RAG_N_QUERY_IDS`, at roughly **1.67 MB per query id**, and it is
memory-mapped, so it is the most evictable part of the footprint.

The practical consequence: no corpus setting fits a 512 MB instance. Zero query
ids still needs 1,545 MB. Sizing, if you are picking a machine:

| query ids | index | total | needs |
|---|---|---|---|
| 1,200 (full) | 2,002 MB | 3,547 MB | 4 GB |
| 800 | 1,335 MB | 2,880 MB | 4 GB |
| 400 | 667 MB | 2,212 MB | 3 GB |
| 250 | 417 MB | 1,962 MB | 2 GB, with a 4% margin — too close |

Deployed, on an Azure `B4as_v2` (4 vCPU, 16 GiB, Central India) running the full
eleven-language corpus with the gate enabled: **3,659 MB resident**, 25 s from
service start to a healthy `/health`, and Tier 1 at **120–132 ms** warm against
the 200 ms budget. `reports/latency_report.md` carries the full run.

CPU is the second constraint and is easy to miss behind the memory one. Tier 1
is a forward pass plus FAISS and BM25 over half a million vectors; on a tenth of
a vCPU that is seconds, not milliseconds, whatever the corpus size.

**The UI ships on the same box.** The API serves `web/dist` at `/` when it
exists, so there is one origin, no CORS, and one thing to keep alive. Deploying
`web/` separately — to Vercel or anywhere else — still works with
`VITE_API_BASE` pointed at the API, but then the API needs its own certificate
regardless: a page served over HTTPS cannot call an `http://` origin, and the
microphone needs a secure context to exist at all.

### Using the microphone from another machine

Browsers expose microphone capture only in a secure context — HTTPS, or
localhost. Opening the demo over plain HTTP from another machine's IP address
means `navigator.mediaDevices` does not exist and only typed questions work.
For a LAN demo, generate a certificate and serve over TLS:

```bash
python scripts/make_cert.py                       # writes .run/cert.pem, .run/key.pem
uvicorn rag.server:app --host 0.0.0.0 --port 8443 \
    --ssl-certfile .run/cert.pem --ssl-keyfile .run/key.pem
```

The browser warns once that the certificate is self-signed; the microphone stays
blocked until that warning is accepted.

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

---

## Built by

| | |
| --- | --- |
| **Manas Dedge** | [linkedin.com/in/manas-dedge](https://www.linkedin.com/in/manas-dedge/) |
| **Rahul Kotyal** | [linkedin.com/in/rahul-kotyal-279996220](https://www.linkedin.com/in/rahul-kotyal-279996220) |
| **Atharv Bhosale** | [linkedin.com/in/atharvbhosale555](https://www.linkedin.com/in/atharvbhosale555) |

Honourable mention: **Claude**, which wrote a good deal of this and measured
the rest — including the memory figures in [Deploying](#deploying), the gate
latency work, and the deployment it is running on.

Built for Hack Hyderabad Goa 2026, Task 2.
