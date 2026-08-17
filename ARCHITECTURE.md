# Architecture

## The shape of one turn

```
  microphone
      │
      ▼
┌─────────────┐   Sarvam saaras:v4, 25 s timeout, 1 retry
│ TRANSCRIBE  │   emits: transcript, detected language
└─────┬───────┘
      │  ── on failure ──▶ FAILED (nothing spoken; the client is told why)
      ▼
┌─────────────┐   one query embedding, shared by the next two stages
│  EMBED      │   multilingual-e5-small, CPU, ~5 ms
└─────┬───────┘
      ▼
┌─────────────┐   unsafe patterns; domain similarity scored, not enforced
│  GUARD      │   ~0.1 ms
└─────┬───────┘
      │  ── unsafe ──▶ REFUSED (spoken refusal, not silence)
      ▼
┌─────────────┐   five strategies → RRF → feature rerank
│  RETRIEVE   │   emits a trace event per strategy as it completes
└─────┬───────┘
      ▼
┌─────────────┐   best sentence span from the top passage, template-wrapped
│  TIER 1     │   ← this is the number measured against 200 ms
└─────┬───────┘
      │
      ├──────────────▶ TTS of the Tier 1 answer starts immediately
      ▼
┌─────────────┐   sarvam-105b-conversations, streamed, 30 s timeout
│  TIER 2     │   not counted against the budget, logged as its own series
└─────┬───────┘
      ▼
┌─────────────┐   lexical + semantic support against the retrieved passages
│  GROUNDING  │
└─────┬───────┘
      │  ── unsupported ──▶ DEGRADED (Tier 1 span spoken instead)
      ▼
┌─────────────┐   Sarvam bulbul:v3 in the asker's language
│  SPEAK      │
└─────┬───────┘
      ▼
    DONE
```

Terminal states are `DONE`, `DEGRADED`, `REFUSED`, and `FAILED`. The path the
machine actually took is recorded on every turn and returned to the client, so
the UI can show which branch ran rather than only the outcome.

## Why two tiers

A hosted LLM call crosses the public internet and decodes tokens sequentially.
The floor on that is hundreds of milliseconds before the model has produced
anything worth speaking. No amount of engineering inside this repository moves
that floor, so a 200 ms end-to-end generative target would only be reachable by
redefining what is being measured.

Instead the pipeline answers twice:

**Tier 1 is extractive.** Retrieval returns passages; the sentence within the
top passage that best matches the query is returned verbatim, wrapped in
confidence-conditioned framing. There is no generation step, so there is
nothing to hallucinate — the answer is a substring of the corpus by
construction. This is the path measured against 200 ms, and it is measured
end to end: guardrail, embedding, all five retrieval strategies, fusion,
reranking, and span selection.

**Tier 2 is generative.** In parallel, the top passages go to
`sarvam-105b-conversations`, which synthesises one natural spoken answer across
them and streams it back. Its latency is recorded as a separate series and
never averaged into the Tier 1 percentiles. Before it is allowed to reach the
speaker it must pass the grounding check; if it drifts, the Tier 1 span is
spoken instead and the turn is marked `DEGRADED`.

Both numbers appear in `reports/latency_report.md`, labelled, with no percentile
mixing the two.

## Retrieval

Five strategies vote, then reciprocal rank fusion merges their ranked lists:

| Strategy | Index | Catches |
| --- | --- | --- |
| `dense_passage` | passage HNSW | topical match on the whole passage |
| `dense_sentence` | sentence HNSW | one on-point sentence inside an otherwise diffuse passage |
| `bm25_passage` | passage sparse | exact terms, names, numbers the encoder blurs |
| `bm25_sentence` | sentence sparse | exact terms in a specific sentence |
| `cluster` | 160 query clusters | "questions like this one were answered by these passages" |

RRF (`k = 60`) needs no score normalisation between sparse and dense, which is
what makes fusing a BM25 score with a cosine defensible.

**Multi-granularity.** The same corpus is indexed at two granularities plus a
query-side index. Sentence hits are rolled up to their parent passage before
fusion, so granularity is a retrieval signal rather than a separate result set.

**Cross-language grouping.** Every passage exists in all four languages. Left
alone, the four variants of one passage would occupy four of the five final
slots with identical content. They are collapsed into one candidate whose
strategy votes are pooled; the variant shown is the asker's language when it was
retrieved, otherwise whichever language surfaced it. Language routing has three
modes: `cross` (default, all languages eligible), `strict` (only the asker's
language), and `pivot` (only other languages — the cross-lingual demo).

**Reranking.** A feature re-scorer over the top 20 fused candidates, ~0.5 ms of
vector arithmetic. It began with six weighted features — dense cosine,
normalised BM25, RRF position, token overlap, language match, strategy
agreement — and measurement removed most of them. Scoring on dense similarity
alone gave recall@5 38.5% and MRR@5 0.263, against 30.0% and 0.202 for the
mixed weights, because lexical noise and fusion position both promote passages
that merely share vocabulary with the question. Fusion earns its place by
assembling the candidate pool; it does not belong in the final ordering.

The strategy ablations are similarly unflattering and are reported as measured:
dropping any single strategy moves recall@5 by at most 1.5 points on this
corpus, because the five overlap heavily. They buy breadth of recall and the
visible reasoning the demo depends on, not a large accuracy jump.

A cross-encoder was left out of the timed path: 20 candidates on CPU costs more
than the whole Tier 1 budget. With P100 at 13.9 ms there is room to run one over
a shorter shortlist, which is the most promising unexplored quality lever.

**BM25.** Term weights are precomputed at build time into a scipy CSR matrix, so
scoring a query is one sparse mat-vec instead of a Python loop over 47k
documents.

## Sizing

| Artefact | Count |
| --- | --- |
| MS MARCO query ids | 1,200 (local) / 800 (deployed) |
| Languages | English, Hindi, Tamil, Bengali |
| Queries | 4,800 |
| Passages | 47,780 |
| Sentences | 143,963 |
| Query clusters | 160 |

Index artefacts are memory-mapped rather than read into the heap, which is what
keeps the deployed container inside a small instance.

## Guardrails

Three, at different points, doing different jobs:

1. **Query guardrail** (before retrieval) — unsafe-pattern matching, plus a
   domain-similarity score that is *reported rather than enforced* by default.
   Calibration showed off-topic and in-corpus questions overlap almost
   completely in that similarity, because an off-topic question is still a
   well-formed question. Worse, rejecting before retrieval hides the reasoning
   trace, so the user sees a dead end with no explanation. The score is shown
   in the trace and `RAG_DOMAIN_FILTER=1` turns enforcement back on. See
   `reports/guardrail.json`.
2. **Grounding check** (after Tier 2, before speech) — lexical overlap and
   semantic similarity against the retrieved passages. Failing both floors sends
   the turn back to the Tier 1 span. Either signal alone can carry an answer,
   which matters for cross-lingual turns: a Hindi question answered from a
   Bengali passage scores 0.00 lexical and 0.91 semantic, and is correctly
   accepted. `scripts/selftest.py` includes a negative control so a check that
   passes everything cannot masquerade as a working one.
3. **Audible tiering** (at speech synthesis, ~1 s with bulbul:v2) — the confidence tier selects the
   spoken framing, and is also where genuine refusal happens, because by this
   point there is retrieved evidence to judge. High states the answer directly;
   low hedges out loud before the content; refuse says so plainly. Thresholds
   are calibrated against gold labels: the "high" band runs at 31.4% top-1
   against a 20.5% base rate, and the refused band at 13.3% top-5 against 41.5%.
   See `reports/confidence.json`.

## Layout

```
rag/
  config.py              all tunables, environment-overridable
  index/
    text.py              Unicode-aware tokenisation for four scripts
    bm25.py              CSR-backed BM25
    build.py             sentence + passage + cluster index build
    build_baseline.py    the naive index, for the race
  retrieval/
    store.py             mmap'd artefact loading, single encoder instance
    lang.py              script-range language detection
    retriever.py         five strategies, RRF, rerank, grouping
    baseline.py          fixed-chunk brute-force path
  generation/
    confidence.py        similarity + margin + agreement → tier
    templates.py         spoken framing per language per tier
    tier1.py             extractive answer
    grounding.py         lexical + semantic support check
    tier2.py             streamed synthesis, gated on grounding
  harness/
    guardrail.py         query-level admission
    pipeline.py          the state machine
  voice/
    sarvam.py            STT, TTS, chat; retries, timeouts, stub mode
  server.py              SSE endpoints
scripts/
  prepare_data.py        corpus construction from MSMARCO-XI
  eval_retrieval.py      ablation and rerank weight sweep
  calibrate.py           guardrail floor and confidence thresholds
  latency_probe.py       fast Tier 1 gate check
benchmark.py             P50/P70/P100 for both tiers
web/                     Vite + React client
```
