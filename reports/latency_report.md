# Latency report

Generated 2026-08-18T05:41:14+00:00 on arm64 Darwin py3.13.7.

Two tiers are measured, and they are never averaged together.

* **Tier 1** — guardrail, query embedding, multi-strategy retrieval, fusion,
  rerank, extractive span selection. No model generation, no network call.
  This is the series measured against the 200 ms target.
* **Tier 2** — hosted LLM synthesis, streamed. Network-bound by definition and
  reported separately.

Query set: 300 queries, stratified across 4 languages.

## Tier 1 (target-bound)

| Metric | ms |
| --- | --- |
| P50 | 11.64 |
| P70 | 12.4 |
| P90 | 13.65 |
| P100 | 25.62 |
| mean | 11.97 |

Within 200 ms: **100.0%** of queries; P100 inside target.

### Per stage (mean / P100 ms)

| Stage | mean | P100 |
| --- | --- | --- |
| embed | 5.85 | 9.3 |
| guardrail | 0.11 | 0.32 |
| detect | 0.04 | 0.14 |
| dense | 0.58 | 1.1 |
| bm25 | 2.81 | 5.25 |
| cluster | 0.52 | 4.13 |
| fusion | 0.03 | 0.03 |
| rerank | 1.44 | 4.67 |
| retrieval_subtotal | 6.01 | 15.99 |
| extract | 0.0 | 0.01 |

### Per language (ms)

| Language | P50 | P70 | P100 |
| --- | --- | --- | --- |
| ben_Beng | 11.73 | 12.28 | 25.62 |
| tam_Taml | 12.08 | 12.78 | 17.18 |
| eng_Latn | 11.14 | 11.52 | 14.55 |
| hin_Deva | 11.93 | 12.5 | 15.7 |

## Against the naive baseline

Same corpus, same encoder, same machine, same queries. The naive path glues
each query's passages into one document, cuts it into fixed 512-character
chunks, and does a single brute-force scan. It runs one strategy where the
pipeline runs five, so latency alone would be a misleading comparison — both
are scored on retrieval quality too.

| | P50 ms | P100 ms | query hit@5 |
| --- | --- | --- | --- |
| Multi-strategy | 11.64 | 25.62 | 90.3% |
| Naive | 6.33 | 7.52 | 84.3% |

The naive path is 1.8x faster at P50 — it does strictly less work. It also reaches the asking query's passages 6.0 points less often. The extra milliseconds buy the recall, and both numbers sit far inside the budget.
Multi-strategy strict recall@5 (the one passage MS MARCO marked selected): 37.3%.

## Tier 1 with the cross-encoder rerank (precision mode)

A cross-encoder scores each query-passage pair jointly rather than
comparing two independent vectors. It is off by default: it fits this
budget on this machine, but it spends most of it, and a slower deployment
box would not have the room. It is switchable per request so the demo can
show the trade rather than assert it.

| Metric | fast path | precision mode |
| --- | --- | --- |
| P50 | 11.64 ms | 94.42 ms |
| P100 | 25.62 ms | 161.97 ms |
| within 200 ms | 100.0% | 100.0% |

Quality at depth 10, measured separately over 120 queries
(`reports/cross_encoder_eval.json`): recall@5 43.3% -> 50.0%,
precision@1 21.7% -> 24.2%.

## Tier 2 (generative, not target-bound)

| Metric | ms |
| --- | --- |
| P50 total | 2768.22 |
| P70 total | 2922.98 |
| P100 total | 16656.35 |
| P50 first token | 302.61 |

Grounding check passed on 100.0% of generations; 0.0% fell back to the Tier 1 span.
