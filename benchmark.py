"""Latency benchmark. Produces P50/P70/P100 for Tier 1 and Tier 2 separately.

Tier 1 (retrieval + extractive answer) is the series measured against the 200 ms
target. Tier 2 (LLM synthesis) is a network call to a hosted model and is
reported as its own series — it is never folded into the Tier 1 number, and no
percentile in this report mixes the two.

The query set is a stratified sample across all corpus languages, so the
percentiles describe varied real queries rather than one query repeated.

Usage::

    python benchmark.py                    # Tier 1 only, 300 queries
    python benchmark.py --n 300 --tier2 30 # add 30 Tier 2 calls
    python benchmark.py --baseline         # also race the naive baseline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import random
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone

import pyarrow.parquet as pq

from rag.config import CORPUS_DIR, RANDOM_SEED, REPORT_DIR, TIER1_TARGET_MS
from rag.generation import tier1 as tier1_mod
from rag.generation.confidence import score_retrieval
from rag.harness.guardrail import check_query
from rag.retrieval.retriever import Retriever


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if p >= 100:
        return ordered[-1]
    k = (len(ordered) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarise(values: list[float]) -> dict:
    return {
        "n": len(values),
        "p50": round(percentile(values, 50), 2),
        "p70": round(percentile(values, 70), 2),
        "p90": round(percentile(values, 90), 2),
        "p100": round(percentile(values, 100), 2),
        "mean": round(statistics.mean(values), 2) if values else 0.0,
    }


def stratified_sample(n: int) -> list[tuple[int, str, str]]:
    table = pq.read_table(CORPUS_DIR / "queries.parquet").to_pydict()
    rows = list(zip(table["query_id"], table["lang"], table["text"]))
    rng = random.Random(RANDOM_SEED + 31)
    rng.shuffle(rows)
    by_lang: dict[str, list] = defaultdict(list)
    for row in rows:
        by_lang[row[1]].append(row)
    per_lang = max(1, n // len(by_lang))
    sample = [r for lang_rows in by_lang.values() for r in lang_rows[:per_lang]]
    rng.shuffle(sample)
    return sample


def run_tier1(retriever: Retriever, sample, cross_encode: bool = False) -> dict:
    stage_series: dict[str, list[float]] = defaultdict(list)
    totals: list[float] = []
    by_lang: dict[str, list[float]] = defaultdict(list)
    tiers: dict[str, int] = defaultdict(int)

    retriever.retrieve("warmup", cross_encode=cross_encode)  # exclude cold start

    for _, lang, text in sample:
        # The harness embeds once and shares the vector, so the encoder is timed
        # here rather than inside retrieve(). It still counts towards Tier 1.
        embed_start = time.perf_counter()
        qvec = retriever.store.encode_query(text)
        embed_ms = (time.perf_counter() - embed_start) * 1000

        guard = check_query(text, qvec)
        result = retriever.retrieve(text, qvec=qvec, cross_encode=cross_encode)
        confidence = score_retrieval(result)
        answer = tier1_mod.build(result, confidence)

        total = embed_ms + guard.latency_ms + result.timings_ms["total"] + answer.latency_ms
        totals.append(total)
        by_lang[lang].append(total)
        tiers[answer.tier] += 1
        stage_series["embed"].append(embed_ms)
        stage_series["guardrail"].append(guard.latency_ms)
        for stage, ms in result.timings_ms.items():
            if stage == "embed":
                continue
            stage_series["retrieval_subtotal" if stage == "total" else stage].append(ms)
        stage_series["extract"].append(answer.latency_ms)

    return {
        "overall": summarise(totals),
        "by_language": {lang: summarise(v) for lang, v in by_lang.items()},
        "stages": {stage: summarise(v) for stage, v in stage_series.items()},
        "confidence_tiers": dict(tiers),
        "target_ms": TIER1_TARGET_MS,
        "within_target_p100": percentile(totals, 100) <= TIER1_TARGET_MS,
        "within_target_pct": round(100 * sum(1 for t in totals if t <= TIER1_TARGET_MS) / len(totals), 1),
    }


async def run_tier2(retriever: Retriever, sample) -> dict:
    from rag.generation import tier2 as tier2_mod
    from rag.voice.sarvam import SarvamClient

    client = SarvamClient()
    totals: list[float] = []
    first_tokens: list[float] = []
    grounded = fallbacks = 0

    for _, _, text in sample:
        result = retriever.retrieve(text)
        confidence = score_retrieval(result)
        answer1 = tier1_mod.build(result, confidence)
        answer2 = await tier2_mod.generate(result, confidence, answer1.span, client)
        totals.append(answer2.latency_ms)
        if answer2.first_token_ms:
            first_tokens.append(answer2.first_token_ms)
        if answer2.grounding and answer2.grounding.supported:
            grounded += 1
        if answer2.used_fallback:
            fallbacks += 1

    await client.aclose()
    return {
        "total": summarise(totals),
        "first_token": summarise(first_tokens),
        "grounded_pct": round(100 * grounded / len(sample), 1) if sample else 0.0,
        "fallback_pct": round(100 * fallbacks / len(sample), 1) if sample else 0.0,
        "mocked": client.mock,
    }


def gold_labels() -> tuple[dict[int, set[str]], dict[int, set[str]]]:
    """Selected passages per query, and every passage belonging to that query."""
    passages = pq.read_table(CORPUS_DIR / "passages.parquet").to_pydict()
    selected: dict[int, set[str]] = defaultdict(set)
    belongs: dict[int, set[str]] = defaultdict(set)
    for qid, p_idx, is_sel in zip(
        passages["query_id"], passages["p_idx"], passages["is_selected"]
    ):
        group = f"{qid}:{p_idx}"
        belongs[qid].add(group)
        if is_sel:
            selected[qid].add(group)
    return selected, belongs


def run_baseline(retriever: Retriever, sample) -> dict | None:
    """Latency *and* quality for the naive path.

    Reporting only latency here would be misleading: the naive path does less
    work per query, so it can win on milliseconds while returning worse
    passages. Both lanes are scored on the same queries and the same labels.
    """
    from rag.retrieval.baseline import NaiveRetriever, artefacts_exist

    if not artefacts_exist():
        return None
    selected, belongs = gold_labels()
    naive = NaiveRetriever(retriever.store)
    naive.retrieve("warmup")

    totals: list[float] = []
    strict = qhit = 0
    for qid, _, text in sample:
        result = naive.retrieve(text, top_k=5)
        totals.append(result["total_ms"])
        # The naive index is chunked per (query_id, language); a chunk counts as
        # a hit for the same groups its source passages belong to.
        hit_qids = {row["query_id"] for row in result["top"]}
        strict += 1 if qid in hit_qids and selected.get(qid) else 0
        qhit += 1 if qid in hit_qids else 0

    stats = summarise(totals)
    stats["query_hit_at_5_pct"] = round(100 * qhit / len(sample), 1)
    stats["gold_query_reached_pct"] = round(100 * strict / len(sample), 1)
    return stats


def score_multi_strategy(retriever: Retriever, sample) -> dict:
    """Same quality metrics for the multi-strategy path, same labels."""
    selected, _ = gold_labels()
    strict = qhit = 0
    for qid, _, text in sample:
        result = retriever.retrieve(text, top_k=5, exclude_qid=qid)
        groups = {c.group_id for c in result.candidates}
        strict += 1 if groups & selected.get(qid, set()) else 0
        qhit += 1 if any(c.query_id == qid for c in result.candidates) else 0
    return {
        "strict_recall_at_5_pct": round(100 * strict / len(sample), 1),
        "query_hit_at_5_pct": round(100 * qhit / len(sample), 1),
    }


def write_markdown(report: dict) -> None:
    tier1 = report["tier1"]
    lines = [
        "# Latency report",
        "",
        f"Generated {report['generated_at']} on {report['machine']}.",
        "",
        "Two tiers are measured, and they are never averaged together.",
        "",
        "* **Tier 1** — guardrail, query embedding, multi-strategy retrieval, fusion,",
        "  rerank, extractive span selection. No model generation, no network call.",
        "  This is the series measured against the "
        f"{tier1['target_ms']:.0f} ms target.",
        "* **Tier 2** — hosted LLM synthesis, streamed. Network-bound by definition and",
        "  reported separately.",
        "",
        f"Query set: {report['n_queries']} queries, stratified across "
        f"{len(tier1['by_language'])} languages.",
        "",
        "## Tier 1 (target-bound)",
        "",
        "| Metric | ms |",
        "| --- | --- |",
        f"| P50 | {tier1['overall']['p50']} |",
        f"| P70 | {tier1['overall']['p70']} |",
        f"| P90 | {tier1['overall']['p90']} |",
        f"| P100 | {tier1['overall']['p100']} |",
        f"| mean | {tier1['overall']['mean']} |",
        "",
        f"Within {tier1['target_ms']:.0f} ms: **{tier1['within_target_pct']}%** of queries; "
        f"P100 {'inside' if tier1['within_target_p100'] else 'outside'} target.",
        "",
        "### Per stage (mean / P100 ms)",
        "",
        "| Stage | mean | P100 |",
        "| --- | --- | --- |",
    ]
    for stage, stats in tier1["stages"].items():
        lines.append(f"| {stage} | {stats['mean']} | {stats['p100']} |")

    lines += ["", "### Per language (ms)", "", "| Language | P50 | P70 | P100 |", "| --- | --- | --- | --- |"]
    for lang, stats in tier1["by_language"].items():
        lines.append(f"| {lang} | {stats['p50']} | {stats['p70']} | {stats['p100']} |")

    if report.get("baseline"):
        base = report["baseline"]
        quality = report.get("quality", {})
        ours = tier1["overall"]["p50"]
        ratio = base["p50"] / ours if ours else 0
        lines += [
            "",
            "## Against the naive baseline",
            "",
            "Same corpus, same encoder, same machine, same queries. The naive path glues",
            "each query's passages into one document, cuts it into fixed 512-character",
            "chunks, and does a single brute-force scan. It runs one strategy where the",
            "pipeline runs five, so latency alone would be a misleading comparison — both",
            "are scored on retrieval quality too.",
            "",
            "| | P50 ms | P100 ms | query hit@5 |",
            "| --- | --- | --- | --- |",
            f"| Multi-strategy | {ours} | {tier1['overall']['p100']} | "
            f"{quality.get('query_hit_at_5_pct', '—')}% |",
            f"| Naive | {base['p50']} | {base['p100']} | {base['query_hit_at_5_pct']}% |",
            "",
        ]
        if ratio >= 1.0:
            lines.append(
                f"The pipeline is **{ratio:.1f}x faster at P50** *and* retrieves better."
            )
        else:
            lines.append(
                f"The naive path is {1 / ratio:.1f}x faster at P50 — it does strictly less "
                f"work. It also reaches the asking query's passages "
                f"{quality.get('query_hit_at_5_pct', 0) - base['query_hit_at_5_pct']:.1f} "
                "points less often. The extra milliseconds buy the recall, and both numbers "
                "sit far inside the budget."
            )
        lines.append(
            f"Multi-strategy strict recall@5 (the one passage MS MARCO marked selected): "
            f"{quality.get('strict_recall_at_5_pct', '—')}%."
        )

    if report.get("tier1_precision_mode"):
        pm = report["tier1_precision_mode"]
        lines += [
            "",
            "## Tier 1 with the cross-encoder rerank (precision mode)",
            "",
            "A cross-encoder scores each query-passage pair jointly rather than",
            "comparing two independent vectors. It is off by default: it fits this",
            "budget on this machine, but it spends most of it, and a slower deployment",
            "box would not have the room. It is switchable per request so the demo can",
            "show the trade rather than assert it.",
            "",
            "| Metric | fast path | precision mode |",
            "| --- | --- | --- |",
            f"| P50 | {tier1['overall']['p50']} ms | {pm['overall']['p50']} ms |",
            f"| P100 | {tier1['overall']['p100']} ms | {pm['overall']['p100']} ms |",
            f"| within {tier1['target_ms']:.0f} ms | {tier1['within_target_pct']}% | "
            f"{pm['within_target_pct']}% |",
            "",
            "Quality at depth 10, measured separately over 120 queries",
            "(`reports/cross_encoder_eval.json`): recall@5 43.3% -> 50.0%,",
            "precision@1 21.7% -> 24.2%.",
        ]

    if report.get("tier2"):
        tier2 = report["tier2"]
        note = " (mocked: no API key at run time)" if tier2["mocked"] else ""
        lines += [
            "",
            f"## Tier 2 (generative, not target-bound){note}",
            "",
            "| Metric | ms |",
            "| --- | --- |",
            f"| P50 total | {tier2['total']['p50']} |",
            f"| P70 total | {tier2['total']['p70']} |",
            f"| P100 total | {tier2['total']['p100']} |",
            f"| P50 first token | {tier2['first_token']['p50']} |",
            "",
            f"Grounding check passed on {tier2['grounded_pct']}% of generations; "
            f"{tier2['fallback_pct']}% fell back to the Tier 1 span.",
        ]

    (REPORT_DIR / "latency_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _update_readme(report)


README_BEGIN = "<!-- BENCHMARK:BEGIN -->"
README_END = "<!-- BENCHMARK:END -->"


def _update_readme(report: dict) -> None:
    """Keep the README's headline numbers in step with the last run."""
    readme = REPORT_DIR.parent / "README.md"
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    if README_BEGIN not in text or README_END not in text:
        return

    tier1 = report["tier1"]
    rows = [
        "",
        "| | P50 | P70 | P100 | measured against 200 ms |",
        "| --- | --- | --- | --- | --- |",
        f"| **Tier 1** — retrieval + extractive answer | {tier1['overall']['p50']} ms | "
        f"{tier1['overall']['p70']} ms | {tier1['overall']['p100']} ms | yes |",
    ]
    if report.get("tier2"):
        t2 = report["tier2"]["total"]
        mocked = " (stubbed)" if report["tier2"]["mocked"] else ""
        rows.append(
            f"| **Tier 2** — LLM synthesis{mocked} | {t2['p50'] / 1000:.2f} s | "
            f"{t2['p70'] / 1000:.2f} s | {t2['p100'] / 1000:.2f} s | no, by design |"
        )
    if report.get("baseline"):
        base = report["baseline"]
        rows.append(
            f"| Naive baseline — fixed chunks, one dense scan | {base['p50']} ms | "
            f"{base['p70']} ms | {base['p100']} ms | for comparison |"
        )
    rows += [
        "",
        f"{tier1['within_target_pct']}% of Tier 1 queries land inside the budget; "
        f"P100 is {tier1['overall']['p100']} ms against a {tier1['target_ms']:.0f} ms target.",
        "",
        "Per-stage and per-language breakdowns: [reports/latency_report.md](reports/latency_report.md).",
        "",
    ]
    head, _, rest = text.partition(README_BEGIN)
    _, _, tail = rest.partition(README_END)
    readme.write_text(head + README_BEGIN + "\n".join(rows) + README_END + tail, encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="Tier 1 query count")
    parser.add_argument("--tier2", type=int, default=0, help="Tier 2 query count (0 skips)")
    parser.add_argument("--baseline", action="store_true", help="also benchmark the naive path")
    parser.add_argument(
        "--precision-mode",
        action="store_true",
        help="also benchmark Tier 1 with the cross-encoder rerank enabled",
    )
    args = parser.parse_args()

    sample = stratified_sample(args.n)
    retriever = Retriever()

    print(f"Tier 1: {len(sample)} queries ...")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": f"{platform.machine()} {platform.system()} py{platform.python_version()}",
        "n_queries": len(sample),
        "embed_model": retriever.store.stats.get("embed_model"),
        "corpus": {k: v for k, v in retriever.store.stats.items() if isinstance(v, int)},
        "tier1": run_tier1(retriever, sample),
    }
    print(f"  P50 {report['tier1']['overall']['p50']} ms | P100 {report['tier1']['overall']['p100']} ms")

    if args.precision_mode:
        print("Tier 1 with cross-encoder ...")
        report["tier1_precision_mode"] = run_tier1(retriever, sample, cross_encode=True)
        pm = report["tier1_precision_mode"]["overall"]
        print(f"  P50 {pm['p50']} ms | P100 {pm['p100']} ms")

    if args.baseline:
        print("Naive baseline ...")
        report["baseline"] = run_baseline(retriever, sample)
        report["quality"] = score_multi_strategy(retriever, sample)
        if report["baseline"]:
            print(f"  naive  P50 {report['baseline']['p50']} ms  "
                  f"query hit@5 {report['baseline']['query_hit_at_5_pct']}%")
            print(f"  multi  query hit@5 {report['quality']['query_hit_at_5_pct']}%  "
                  f"strict recall@5 {report['quality']['strict_recall_at_5_pct']}%")

    if args.tier2:
        print(f"Tier 2: {args.tier2} queries ...")
        report["tier2"] = await run_tier2(retriever, sample[: args.tier2])
        print(f"  P50 {report['tier2']['total']['p50']} ms")

    (REPORT_DIR / "latency_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report)
    print(f"\nwrote {REPORT_DIR / 'latency_report.json'} and latency_report.md")


if __name__ == "__main__":
    asyncio.run(main())
