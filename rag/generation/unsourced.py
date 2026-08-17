"""The escape hatch: answering from the model's own knowledge, and saying so.

Everything else in this package is grounded — Tier 1 quotes the corpus verbatim,
Tier 2 is checked against it before it may speak. This module deliberately is
not, and exists because a retrieval demo over a fixed corpus otherwise looks
broken to anyone who asks it an ordinary question. The corpus is 1,200 MS MARCO
questions; "who is the prime minister of India" is not among them and is not in
the full dataset either.

Two rules make that acceptable rather than a hole in the guardrails:

1. It is **off by default**. The system refuses first; this runs only when the
   caller explicitly allows it.
2. The provenance is stated **before the content**, in the answer text itself,
   so a listener knows what they are hearing while they hear it — not from a
   badge they may never look at.

Nothing here is checked by ``grounding.check``: there is no retrieved evidence
to check against, which is the whole point. The turn is tagged ``unsourced``
everywhere downstream — event stream, UI, and the turn record.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from ..retrieval.lang import display_name
from ..voice.sarvam import SarvamClient, SarvamError
from .templates import unsourced_prefix

SYSTEM_PROMPT = (
    "You answer general-knowledge questions out loud for a voice assistant. Rules:\n"
    "1. Answer in {language} — the same language the question was asked in.\n"
    "2. One or two spoken sentences. No lists, no markdown.\n"
    "3. State only what you are confident about. If you are unsure, say you are unsure\n"
    "   rather than guessing at specifics like dates, numbers or names.\n"
    "4. Do not claim to have consulted any document — this answer has no source."
)


@dataclass
class UnsourcedAnswer:
    text: str
    spoken_text: str
    latency_ms: float
    first_token_ms: float
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "spoken_text": self.spoken_text,
            "latency_ms": round(self.latency_ms, 2),
            "first_token_ms": round(self.first_token_ms, 2),
            "error": self.error,
            "unsourced": True,
        }


async def generate(
    query: str,
    lang: str,
    client: SarvamClient,
    on_delta: Callable[[str], None] | None = None,
) -> UnsourcedAnswer:
    start = time.perf_counter()
    first_token = 0.0
    chunks: list[str] = []
    error: str | None = None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(language=display_name(lang))},
        {"role": "user", "content": query},
    ]

    try:
        async for delta in client.chat_stream(messages, max_tokens=300):
            if not chunks:
                first_token = (time.perf_counter() - start) * 1000
            chunks.append(delta)
            if on_delta:
                on_delta(delta)
    except SarvamError as exc:
        error = str(exc)

    text = "".join(chunks).strip()
    return UnsourcedAnswer(
        text=text,
        # The disclaimer leads, so it is heard before the claim it qualifies.
        spoken_text=(unsourced_prefix(lang) + text) if text else "",
        latency_ms=(time.perf_counter() - start) * 1000,
        first_token_ms=first_token,
        error=error,
    )
