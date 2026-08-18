"""The harness: an explicit state machine over the whole answer pipeline.

States and the transitions between them are enumerated rather than implied by
call order, because the UI renders this machine live and the demo narrates it.
Each stage has its own timeout, retry policy, and degradation rule; a stage
failing moves the machine to a degraded state that still answers, instead of
raising out of the pipeline.

    RECEIVED -> TRANSCRIBE -> GUARD -> RETRIEVE -> TIER1 -> TIER2 -> SPEAK -> DONE
                    |           |                    |        |
                    |           v                    |        +-> DEGRADED (Tier 1 spoken)
                    |        REFUSED                 +-> DEGRADED
                    +-> FAILED

Every stage emits trace events as it runs; the API layer forwards them to the
browser unmodified, which is what makes the retrieval reasoning visible while
Tier 2 is still generating.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import AsyncIterator

from ..config import ALLOW_UNSOURCED_DEFAULT, CROSS_ENCODER_DEFAULT
from ..generation import tier1 as tier1_mod
from ..generation import tier2 as tier2_mod
from ..generation import unsourced as unsourced_mod
from ..generation import relevance as relevance_mod
from ..generation.confidence import score_retrieval
from ..generation.templates import off_topic_for
from ..retrieval.lang import detect, from_sarvam_code, has_voice, sarvam_code
from ..retrieval.retriever import LangMode, Retriever
from ..voice.sarvam import SarvamClient, describe
from .guardrail import check_query


class State(str, Enum):
    RECEIVED = "received"
    TRANSCRIBE = "transcribe"
    GUARD = "guard"
    RETRIEVE = "retrieve"
    TIER1 = "tier1"
    TIER2 = "tier2"
    SPEAK = "speak"
    DONE = "done"
    REFUSED = "refused"
    UNSOURCED = "unsourced"
    DEGRADED = "degraded"
    FAILED = "failed"


# Per-stage wall-clock ceilings. Retrieval's is generous relative to its
# measured cost so a cold page-in cannot fail a request outright.
TIMEOUTS_S = {
    State.TRANSCRIBE: 60.0,
    State.GUARD: 2.0,
    State.RETRIEVE: 5.0,
    State.TIER1: 2.0,
    State.TIER2: 100.0,
    State.SPEAK: 60.0,
}
RETRIES = {State.TRANSCRIBE: 1, State.SPEAK: 1}


@dataclass
class Turn:
    """Everything one question produced, including the machine's own path."""

    turn_id: str
    query: str = ""
    lang: str = "eng_Latn"
    state: State = State.RECEIVED
    path: list[str] = field(default_factory=list)
    stage_ms: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "query": self.query,
            "lang": self.lang,
            "state": self.state.value,
            "path": self.path,
            "stage_ms": {k: round(v, 2) for k, v in self.stage_ms.items()},
            "errors": self.errors,
            **self.payload,
        }


class Pipeline:
    def __init__(
        self,
        retriever: Retriever | None = None,
        client: SarvamClient | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.client = client or SarvamClient()

    # ------------------------------------------------------------- stages ---
    async def _stage(self, turn: Turn, state: State, coro, default=None):
        """Run one stage under its timeout/retry policy, recording the outcome."""
        turn.state = state
        turn.path.append(state.value)
        attempts = RETRIES.get(state, 0) + 1
        start = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                result = await asyncio.wait_for(coro(), timeout=TIMEOUTS_S[state])
                turn.stage_ms[state.value] = (time.perf_counter() - start) * 1000
                return result, None
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"{state.value} exceeded {TIMEOUTS_S[state]}s")
            except Exception as exc:  # noqa: BLE001 - degradation is the point
                last_error = exc
            if attempt < attempts - 1:
                await asyncio.sleep(0.2 * (2**attempt))

        turn.stage_ms[state.value] = (time.perf_counter() - start) * 1000
        turn.errors[state.value] = describe(last_error)
        return default, last_error

    # ---------------------------------------------------------------- run ---
    async def run(
        self,
        *,
        text: str | None = None,
        audio: bytes | None = None,
        audio_filename: str = "audio.webm",
        language_code: str | None = None,
        lang_mode: LangMode = "cross",
        speak: bool = True,
        speak_tier1: bool = False,
        cross_encode: bool = CROSS_ENCODER_DEFAULT,
        allow_unsourced: bool = ALLOW_UNSOURCED_DEFAULT,
        relevance_gate: bool | None = None,
    ) -> AsyncIterator[dict]:
        """Drive one turn, yielding trace events as each stage completes.

        ``speak_tier1`` is off by default. Synthesising both tiers means two
        audio clips for one question, and since speech synthesis takes about a
        second either way, the "instant" Tier 1 clip can still arrive after the
        Tier 2 one. Tier 1 is shown instantly and spoken only when asked for.
        """
        turn = Turn(turn_id=uuid.uuid4().hex[:12])
        pending_detection: dict | None = None
        yield {"type": "state", "state": State.RECEIVED.value, "turn_id": turn.turn_id}

        # ---- transcribe -----------------------------------------------------
        if audio is not None:
            async def _transcribe():
                return await self.client.transcribe(audio, audio_filename, language_code)

            transcript, error = await self._stage(turn, State.TRANSCRIBE, _transcribe)
            if error or transcript is None:
                turn.state = State.FAILED
                yield {"type": "error", "stage": "transcribe", "message": str(error)}
                yield {"type": "done", "turn": turn.to_dict()}
                return
            turn.query = transcript.text
            turn.lang = from_sarvam_code(transcript.language_code or language_code or "en-IN")
            turn.payload["transcript"] = {
                "text": transcript.text,
                "language_code": transcript.language_code,
                "latency_ms": round(transcript.latency_ms, 2),
                "mocked": transcript.mocked,
            }
            yield {"type": "transcript", **turn.payload["transcript"]}
        else:
            turn.query = (text or "").strip()
            if language_code:
                turn.lang = from_sarvam_code(language_code)
            else:
                detection = detect(turn.query)
                turn.lang = detection.lang
                turn.payload["detection"] = detection.to_dict()
                pending_detection = detection.to_dict()

        if pending_detection is not None:
            yield {"type": "detection", "detection": pending_detection}

        # ---- embed once, shared by guardrail and retrieval -------------------
        embed_start = time.perf_counter()
        qvec = await asyncio.to_thread(self.retriever.store.encode_query, turn.query)
        turn.stage_ms["embed"] = (time.perf_counter() - embed_start) * 1000

        # ---- guardrail -------------------------------------------------------
        async def _guard():
            return check_query(turn.query, qvec)

        verdict, error = await self._stage(turn, State.GUARD, _guard)
        if verdict is not None:
            turn.payload["guardrail"] = verdict.to_dict()
            yield {"type": "guardrail", **verdict.to_dict()}
        if verdict is not None and not verdict.allowed:
            turn.state = State.REFUSED
            refusal = off_topic_for(turn.lang)
            turn.payload["answer"] = {"text": refusal, "tier": "refuse"}
            yield {"type": "refusal", "text": refusal}
            if speak:
                audio_event = await self._speak(turn, refusal, label="refusal")
                if audio_event:
                    yield audio_event
            yield {"type": "done", "turn": turn.to_dict()}
            return

        # ---- retrieval (the timed path) --------------------------------------
        events: list[dict] = []

        async def _retrieve():
            return await asyncio.to_thread(
                self.retriever.retrieve,
                turn.query,
                lang_mode,
                5,
                lambda kind, data: events.append({"type": "trace", "stage": kind, **data}),
                turn.lang,
                None,
                qvec,
                cross_encode,
            )

        result, error = await self._stage(turn, State.RETRIEVE, _retrieve)
        for event in events:
            yield event
        if result is None:
            turn.state = State.FAILED
            yield {"type": "error", "stage": "retrieve", "message": str(error)}
            yield {"type": "done", "turn": turn.to_dict()}
            return

        turn.payload["retrieval"] = result.to_dict()
        yield {"type": "retrieval", **result.to_dict()}

        # ---- Tier 1: extractive, grounded by construction ---------------------
        confidence = score_retrieval(result)

        # The confidence score is relative and can look healthy when nothing
        # relevant came back. Ask a cross-encoder whether anything actually
        # answers the question, and decline outright when it does not.
        verdict = await asyncio.to_thread(
            relevance_mod.check, turn.query, result.candidates, relevance_gate
        )
        turn.payload["relevance"] = verdict.to_dict()
        if verdict.checked:
            turn.stage_ms["relevance"] = verdict.latency_ms
            yield {"type": "relevance", **verdict.to_dict()}
            if not verdict.passed:
                confidence = replace(confidence, tier="refuse")

        turn.payload["confidence"] = confidence.to_dict()

        async def _tier1():
            return tier1_mod.build(result, confidence)

        answer1, _ = await self._stage(turn, State.TIER1, _tier1)
        if answer1 is None:
            turn.state = State.FAILED
            yield {"type": "done", "turn": turn.to_dict()}
            return

        # Tier 1 total is the number measured against the 200 ms target: the
        # embed, the retrieval stages, and the extractive assembly.
        tier1_total = (
            turn.stage_ms["embed"]
            + result.timings_ms["total"]
            + turn.stage_ms.get("relevance", 0.0)
            + answer1.latency_ms
        )
        turn.payload["tier1"] = {**answer1.to_dict(), "tier1_total_ms": round(tier1_total, 2)}
        yield {
            "type": "tier1",
            "confidence": confidence.to_dict(),
            "tier1_total_ms": round(tier1_total, 2),
            # The harness owns the embed and guardrail clocks; the retriever
            # only reports its own stages. Send both so the UI's scale shows
            # the same total that the benchmark measures.
            "harness_ms": {
                "embed": round(turn.stage_ms["embed"], 2),
                "guardrail": round(turn.stage_ms.get("guard", 0.0), 2),
                "relevance": round(turn.stage_ms.get("relevance", 0.0), 2),
            },
            **answer1.to_dict(),
        }

        speak_tasks: list[asyncio.Task] = []
        if speak and speak_tier1 and answer1.tier != "refuse":
            speak_tasks.append(asyncio.create_task(self._speak(turn, answer1.text, label="tier1")))

        if confidence.tier == "refuse":
            for task in speak_tasks:
                await task

            if not allow_unsourced:
                turn.state = State.REFUSED
                yield {"type": "done", "turn": turn.to_dict()}
                return

            # Nothing in the corpus answers this, and the caller has allowed an
            # unsourced answer. It is generated, labelled, and never grounded —
            # there is no evidence to ground it against.
            turn.state = State.UNSOURCED
            turn.path.append(State.UNSOURCED.value)
            unsourced_queue: asyncio.Queue[str] = asyncio.Queue()
            unsourced_task = asyncio.create_task(
                asyncio.wait_for(
                    unsourced_mod.generate(
                        turn.query, turn.lang, self.client,
                        on_delta=lambda d: unsourced_queue.put_nowait(d),
                    ),
                    timeout=TIMEOUTS_S[State.TIER2],
                )
            )
            while not unsourced_task.done() or not unsourced_queue.empty():
                try:
                    delta = await asyncio.wait_for(unsourced_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                yield {"type": "unsourced_delta", "text": delta}

            try:
                answer3 = await unsourced_task
            except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
                turn.errors["unsourced"] = describe(exc)
                answer3 = None

            if answer3 is None or not answer3.text:
                turn.state = State.REFUSED
                yield {"type": "done", "turn": turn.to_dict()}
                return

            turn.payload["unsourced"] = answer3.to_dict()
            yield {"type": "unsourced", **answer3.to_dict()}
            if speak:
                audio_event = await self._speak(turn, answer3.spoken_text, label="unsourced")
                if audio_event:
                    yield audio_event
            yield {"type": "done", "turn": turn.to_dict()}
            return

        # ---- Tier 2: generative, streamed, untimed ---------------------------
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def _tier2():
            return await tier2_mod.generate(
                result,
                confidence,
                answer1.span,
                self.client,
                on_delta=lambda d: queue.put_nowait(d),
            )

        turn.state = State.TIER2
        turn.path.append(State.TIER2.value)
        tier2_task = asyncio.create_task(
            asyncio.wait_for(_tier2(), timeout=TIMEOUTS_S[State.TIER2])
        )
        tier2_start = time.perf_counter()

        while not tier2_task.done() or not queue.empty():
            # Tier 1 audio is released the moment it is ready, mid-stream, so
            # the listener hears the fast answer before Tier 2 finishes.
            for task in list(speak_tasks):
                if task.done():
                    speak_tasks.remove(task)
                    event = task.result()
                    if event:
                        yield event
            try:
                delta = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            if delta is None:
                break
            yield {"type": "tier2_delta", "text": delta}

        try:
            answer2 = await tier2_task
        except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
            turn.errors["tier2"] = str(exc)
            answer2 = None
        turn.stage_ms["tier2"] = (time.perf_counter() - tier2_start) * 1000

        if answer2 is None:
            turn.state = State.DEGRADED
            turn.payload["tier2"] = {"error": turn.errors.get("tier2"), "used_fallback": True}
            yield {"type": "tier2", **turn.payload["tier2"]}
            spoken = answer1.text
        else:
            turn.payload["tier2"] = answer2.to_dict()
            turn.state = State.DEGRADED if answer2.used_fallback else State.TIER2
            yield {"type": "tier2", **answer2.to_dict()}
            spoken = answer2.spoken_text

        # ---- speak -----------------------------------------------------------
        if speak:
            audio_event = await self._speak(turn, spoken, label="final")
            if audio_event:
                yield audio_event
        for task in speak_tasks:
            event = await task
            if event:
                yield event

        if turn.state not in (State.DEGRADED, State.REFUSED, State.FAILED):
            turn.state = State.DONE
        turn.path.append(turn.state.value)
        yield {"type": "done", "turn": turn.to_dict()}

    # -------------------------------------------------------------- speech ---
    async def _speak(self, turn: Turn, text: str, label: str = "final") -> dict | None:
        if not has_voice(turn.lang):
            # Sarvam has speech-to-text but no Bulbul voice for this language.
            # Say so rather than emitting nothing and looking broken.
            return {
                "type": "audio",
                "label": label,
                "audio_b64": "",
                "mocked": False,
                "latency_ms": 0.0,
                "unavailable": f"no voice available for {turn.lang}",
            }

        async def _tts():
            return await self.client.synthesize(text, sarvam_code(turn.lang))

        speech, error = await self._stage(turn, State.SPEAK, _tts)
        if speech is None or error:
            return None
        turn.stage_ms[f"speak_{label}"] = speech.latency_ms
        return {
            "type": "audio",
            "label": label,
            "audio_b64": speech.audio_b64,
            "mocked": speech.mocked,
            "latency_ms": round(speech.latency_ms, 2),
        }
