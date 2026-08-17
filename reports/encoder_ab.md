# Encoder A/B

Dense-only retrieval, 200 stratified queries across all four languages, group-level
scoring (a passage's four language variants count as one document). Ranked over the
full 47,780-passage corpus on an Apple M4.

| Encoder | dim | strict recall@5 | query hit@5 | index encode | CPU query |
| --- | --- | --- | --- | --- | --- |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 18.5% | 56.5% | 144 s | 5.0 ms |
| `intfloat/multilingual-e5-small` | 384 | **37.5%** | **90.5%** | 231 s | 5.2 ms |
| `intfloat/multilingual-e5-base` | 768 | not measured | — | — | — |

**Chosen: `multilingual-e5-small`.**

Two metrics are reported because they answer different questions. *Strict recall@5* asks
whether the one passage MS MARCO marked `is_selected` appears in the top five — a hard
metric, since each query's other candidate passages are also topically on point and
compete for the same slots. *Query hit@5* asks whether any passage belonging to the
asking query was retrieved, which is what determines whether the answer can be grounded
at all.

The paraphrase model is symmetric: it was trained to score whether two sentences mean the
same thing. Query-to-passage retrieval is asymmetric — a short question against a long
document — which is what E5's `query:` / `passage:` prefixes encode. Swapping the encoder
doubled strict recall and lifted query hit by 34 points at effectively identical query
cost, because both models are 384-dimensional and similar depth.

`multilingual-e5-base` was queued for comparison but stalled on download and was cut
rather than left blocking the build. It is worth revisiting only if quality becomes the
binding constraint: at 768 dimensions it doubles index memory, which matters on the
deployment instance, and the small variant already clears the latency budget by 13×.
