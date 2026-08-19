"""FastAPI surface for the voice RAG pipeline.

Every answer endpoint streams server-sent events straight from the harness, so
the browser sees each stage as it happens rather than one response at the end.
That live stream is what powers the visible retrieval trace and the per-stage
stopwatch in the UI.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections import deque
from typing import AsyncIterator

import pyarrow.parquet as pq
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from pathlib import Path

from .config import (
    ALLOW_UNSOURCED_DEFAULT,
    CORPUS_DIR,
    CROSS_ENCODER_DEFAULT,
    LANGUAGES,
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_PER_HOUR,
    RATE_LIMIT_PER_MINUTE,
    RATE_LIMIT_TRUST_PROXY,
    RELEVANCE_GATE_ENABLED,
    TIER1_TARGET_MS,
)
from .harness.pipeline import Pipeline
from .retrieval.retriever import Retriever
from .retrieval.store import get_store

app = FastAPI(title="Voice RAG over MSMARCO-XI", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints that cost money or CPU. Everything else — health, meta, the sample
# questions, the built UI — is free to hammer and is deliberately not counted.
_METERED_PATHS = ("/ask", "/ask-audio", "/race")
# One deque of request timestamps per caller, newest last. Bounded below.
_rate_history: dict[str, deque[float]] = {}


def _client_ip(request: Request) -> str:
    """The caller as best we can tell, honouring a front proxy when trusted."""
    if RATE_LIMIT_TRUST_PROXY:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            # Left-most entry is the original client; everything after it is a
            # proxy that appended itself.
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Per-IP ceiling on the endpoints that spend the Sarvam key.

    A sliding hour rather than a fixed bucket: a fixed window lets a caller
    spend the whole allowance in the last second of one window and again in the
    first second of the next, which is exactly the burst worth stopping.
    """
    if not RATE_LIMIT_ENABLED or not request.url.path.startswith(_METERED_PATHS):
        return await call_next(request)

    now = time.monotonic()
    ip = _client_ip(request)
    history = _rate_history.setdefault(ip, deque())
    while history and now - history[0] > 3600:
        history.popleft()

    in_last_minute = sum(1 for stamp in history if now - stamp <= 60)
    over_minute = in_last_minute >= RATE_LIMIT_PER_MINUTE
    if over_minute or len(history) >= RATE_LIMIT_PER_HOUR:
        window = "minute" if over_minute else "hour"
        retry_after = 60 if over_minute else 3600
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(retry_after)},
            content={
                "error": "rate_limited",
                "detail": (
                    f"Too many questions from this address in the last {window}. "
                    "This is a demo instance and each answer costs speech and "
                    "generation credit; try again shortly."
                ),
            },
        )

    history.append(now)
    # Bound the table. Without this, a scan across many source addresses turns
    # the limiter itself into the leak it was added to prevent.
    if len(_rate_history) > 4096:
        for addr in [a for a, h in _rate_history.items() if not h or now - h[-1] > 3600]:
            del _rate_history[addr]

    return await call_next(request)


_pipeline: Pipeline | None = None


def pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline(retriever=Retriever())
    return _pipeline


@app.on_event("startup")
async def warm_up() -> None:
    """Load indexes, the encoder, and the gate's cross-encoder up front.

    Cold-loading the encoder inside a user's first query would put seconds into
    a turn the whole demo claims is fast.

    The relevance gate's cross-encoder is loaded here for the same reason, and
    it is the larger of the two costs: measured on the deployed instance, the
    first gated question paid 5,506 ms of lazy model load inside a turn that is
    120 ms warm. Left lazy, that whole penalty lands on whoever opens the link
    first, which on a demo is precisely the person you least want it to land on.
    Loading it at startup only moves the cost — the model is 552 MB either way —
    but it moves it to where nobody is timing it.
    """
    await asyncio.to_thread(lambda: pipeline().retriever.retrieve("warmup"))
    if RELEVANCE_GATE_ENABLED:
        await asyncio.to_thread(lambda: get_store().cross_encoder)


class AskRequest(BaseModel):
    query: str
    lang_mode: str = "cross"
    language_code: str | None = None
    speak: bool = True
    speak_tier1: bool = False
    cross_encode: bool = CROSS_ENCODER_DEFAULT
    allow_unsourced: bool = ALLOW_UNSOURCED_DEFAULT


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def _stream(**kwargs) -> AsyncIterator[str]:
    try:
        async for event in pipeline().run(**kwargs):
            yield sse(event)
    except Exception as exc:  # noqa: BLE001 - never leave the client hanging
        yield sse({"type": "error", "stage": "pipeline", "message": str(exc)})
    yield "data: [DONE]\n\n"


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.get("/meta")
async def meta() -> dict:
    store = get_store()
    return {
        "corpus": store.stats,
        "languages": [
            {"code": code, "name": info["name"], "sarvam": info["sarvam"], "voice": info["voice"]}
            for code, info in LANGUAGES.items()
        ],
        "tier1_target_ms": TIER1_TARGET_MS,
        "mock_voice": pipeline().client.mock,
        "cross_encoder_default": CROSS_ENCODER_DEFAULT,
        "allow_unsourced_default": ALLOW_UNSOURCED_DEFAULT,
        "voice_languages": [c for c, i in LANGUAGES.items() if i["voice"]],
    }


# Read once. The file cannot change while the process runs, and this endpoint
# is now a button rather than a page-load, so re-reading and converting 13,200
# rows per press would be work done to produce the same list of rows every time.
_sample_rows: list[tuple] | None = None


def _query_rows() -> list[tuple]:
    global _sample_rows
    if _sample_rows is None:
        table = pq.read_table(CORPUS_DIR / "queries.parquet").to_pydict()
        _sample_rows = list(zip(table["query_id"], table["lang"], table["text"]))
    return _sample_rows


@app.get("/sample-queries")
async def sample_queries(n: int = 4) -> dict:
    rows = list(_query_rows())
    rng = random.Random()
    by_lang: dict[str, list[dict]] = {}
    rng.shuffle(rows)
    for qid, lang, text in rows:
        bucket = by_lang.setdefault(lang, [])
        if len(bucket) < n:
            bucket.append({"query_id": qid, "text": text})
    return {"samples": by_lang}


@app.post("/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream(
            text=request.query,
            language_code=request.language_code,
            lang_mode=request.lang_mode,
            speak=request.speak,
            speak_tier1=request.speak_tier1,
            cross_encode=request.cross_encode,
            allow_unsourced=request.allow_unsourced,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.post("/ask-audio")
async def ask_audio(
    file: UploadFile = File(...),
    language_code: str | None = Form(None),
    lang_mode: str = Form("cross"),
    speak: bool = Form(True),
    cross_encode: bool = Form(CROSS_ENCODER_DEFAULT),
    allow_unsourced: bool = Form(ALLOW_UNSOURCED_DEFAULT),
) -> StreamingResponse:
    audio = await file.read()
    return StreamingResponse(
        _stream(
            audio=audio,
            audio_filename=file.filename or "audio.webm",
            language_code=language_code,
            lang_mode=lang_mode,
            speak=speak,
            cross_encode=cross_encode,
            allow_unsourced=allow_unsourced,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


class RaceRequest(BaseModel):
    query: str
    language_code: str | None = None


@app.post("/race")
async def race(request: RaceRequest) -> dict:
    """Multi-strategy pipeline against the naive baseline, same query, same box."""
    from .retrieval.baseline import NaiveRetriever

    retriever = pipeline().retriever
    naive = NaiveRetriever(retriever.store)
    smart = await asyncio.to_thread(retriever.retrieve, request.query)
    dumb = await asyncio.to_thread(naive.retrieve, request.query)
    return {
        "query": request.query,
        "multi_strategy": {
            "total_ms": round(smart.timings_ms["total"], 2),
            "stages_ms": {k: round(v, 2) for k, v in smart.timings_ms.items()},
            "top": [
                {"passage_id": c.passage_id, "lang": c.lang, "snippet": c.best_sentence[:180]}
                for c in smart.candidates[:3]
            ],
        },
        "naive": {
            "total_ms": round(dumb["total_ms"], 2),
            "stages_ms": dumb["stages_ms"],
            "top": dumb["top"],
        },
    }


# ---------------------------------------------------------------- static UI ---
# Serving the built frontend from the API means the demo needs no Node on the
# machine that runs it, and no CORS or second origin in deployment. Mounted last
# so it never shadows an API route. Regenerate with `npm run build` in web/.
_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="ui")
