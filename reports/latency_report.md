# Latency report

Generated 2026-08-17T06:48:12+00:00 on arm64 Darwin py3.13.7.

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
| P50 | 10.12 |
| P70 | 10.31 |
| P90 | 10.68 |
| P100 | 11.67 |
| mean | 10.15 |

Within 200 ms: **100.0%** of queries; P100 inside target.

### Per stage (mean / P100 ms)

| Stage | mean | P100 |
| --- | --- | --- |
| embed | 5.47 | 6.76 |
| guardrail | 0.11 | 0.19 |
| detect | 0.02 | 0.07 |
| dense | 0.57 | 0.82 |
| bm25 | 2.91 | 3.48 |
| cluster | 0.49 | 1.01 |
| fusion | 0.03 | 0.03 |
| rerank | 0.51 | 1.24 |
| retrieval_subtotal | 4.57 | 5.67 |
| extract | 0.0 | 0.01 |

### Per language (ms)

| Language | P50 | P70 | P100 |
| --- | --- | --- | --- |
| ben_Beng | 10.04 | 10.32 | 11.67 |
| tam_Taml | 10.23 | 10.44 | 11.31 |
| eng_Latn | 9.99 | 10.14 | 11.35 |
| hin_Deva | 10.18 | 10.5 | 11.17 |

## Against the naive baseline

Same corpus, same encoder, same machine, same queries. The naive path glues
each query's passages into one document, cuts it into fixed 512-character
chunks, and does a single brute-force scan. It runs one strategy where the
pipeline runs five, so latency alone would be a misleading comparison — both
are scored on retrieval quality too.

| | P50 ms | P100 ms | query hit@5 |
| --- | --- | --- | --- |
| Multi-strategy | 10.12 | 11.67 | 91.3% |
| Naive | 6.14 | 7.34 | 84.3% |

The naive path is 1.6x faster at P50 — it does strictly less work. It also reaches the asking query's passages 7.0 points less often. The extra milliseconds buy the recall, and both numbers sit far inside the budget.
Multi-strategy strict recall@5 (the one passage MS MARCO marked selected): 37.0%.

## Tier 1 with the cross-encoder rerank (precision mode)

A cross-encoder scores each query-passage pair jointly rather than
comparing two independent vectors. It is off by default: it fits this
budget on this machine, but it spends most of it, and a slower deployment
box would not have the room. It is switchable per request so the demo can
show the trade rather than assert it.

| Metric | fast path | precision mode |
| --- | --- | --- |
| P50 | 10.12 ms | 92.7 ms |
| P100 | 11.67 ms | 158.57 ms |
| within 200 ms | 100.0% | 100.0% |

Quality at depth 10, measured separately over 120 queries
(`reports/cross_encoder_eval.json`): recall@5 43.3% -> 50.0%,
precision@1 21.7% -> 24.2%.

## Tier 2 (generative, not target-bound)

| Metric | ms |
| --- | --- |
| P50 total | 2633.4 |
| P70 total | 3066.24 |
| P100 total | 4008.5 |
| P50 first token | 301.49 |

Grounding check passed on 100.0% of generations; 0.0% fell back to the Tier 1 span.
