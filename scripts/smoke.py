"""End-to-end smoke test over the harness, no browser and no microphone.

Exercises four paths that must all behave differently:

1. an in-domain English question          -> answered, high or low confidence
2. an in-domain Hindi question            -> answered in Hindi
3. the same Hindi question in pivot mode  -> answered from a non-Hindi source
4. an off-topic question                  -> refused before retrieval runs

Run: ``python scripts/smoke.py``
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.harness.pipeline import Pipeline  # noqa: E402

CASES = [
    ("what is a corporation", "cross", "in-domain English"),
    ("कॉर्पोरेशन क्या है", "cross", "in-domain Hindi"),
    ("कॉर्पोरेशन क्या है", "pivot", "Hindi asking, non-Hindi sources only"),
    ("book me a cab to the airport at 6pm", "cross", "off topic"),
]


async def main() -> None:
    pipeline = Pipeline()
    for query, mode, label in CASES:
        print(f"\n{'=' * 78}\n{label}  [{mode}]  {query}\n{'=' * 78}")
        tier2_chars = 0
        async for event in pipeline.run(text=query, lang_mode=mode, speak=False):
            kind = event["type"]
            if kind == "guardrail":
                print(f"  guardrail  allowed={event['allowed']} sim={event['domain_similarity']:.3f} "
                      f"({event['latency_ms']:.2f} ms)")
            elif kind == "refusal":
                print(f"  REFUSED    {event['text'][:90]}")
            elif kind == "retrieval":
                print(f"  retrieval  lang={event['lang']} cross_lingual={event['cross_lingual']} "
                      f"total={event['timings_ms']['total']:.1f} ms")
                for candidate in event["candidates"][:3]:
                    strategies = ",".join(candidate["strategies"])
                    print(f"    [{candidate['lang']}] rr={candidate['rerank_score']:.3f} "
                          f"fuse#{candidate['fusion_rank']} via={strategies}")
                    print(f"      {candidate['best_sentence'][:100]}")
            elif kind == "tier1":
                print(f"  TIER 1     {event['tier1_total_ms']:.1f} ms  tier={event['tier']} "
                      f"confidence={event['confidence']['score']:.2f}")
                print(f"    {event['text'][:140]}")
            elif kind == "tier2_delta":
                tier2_chars += len(event["text"])
            elif kind == "tier2":
                grounding = event.get("grounding")
                print(f"  TIER 2     {event.get('latency_ms', 0):.0f} ms  streamed {tier2_chars} chars  "
                      f"fallback={event.get('used_fallback')}")
                if grounding:
                    print(f"    grounding lexical={grounding['lexical']:.2f} "
                          f"semantic={grounding['semantic']:.2f} -> {grounding['reason']}")
                if event.get("error"):
                    print(f"    error: {event['error']}")
            elif kind == "error":
                print(f"  ERROR      {event['stage']}: {event['message']}")
            elif kind == "done":
                turn = event["turn"]
                print(f"  state={turn['state']}  path={' -> '.join(turn['path'])}")
                if turn["errors"]:
                    print(f"  errors={turn['errors']}")

    await pipeline.client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
