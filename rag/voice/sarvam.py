"""Sarvam AI client: speech-to-text, text-to-speech, chat completions.

Every call is bounded by a timeout and retried a fixed number of times on
transport errors and 5xx responses. When no API key is configured the client
degrades to deterministic stubs so the retrieval and harness layers can be
exercised — and demoed — without credentials.

Endpoints verified against docs.sarvam.ai (August 2026):

* ``POST /speech-to-text``   - multipart, models ``saaras:v3`` / ``saaras:v4``
* ``POST /text-to-speech``   - JSON, models ``bulbul:v2`` / ``bulbul:v3``
* ``POST /v1/chat/completions`` - OpenAI-shaped, SSE when ``stream: true``.
  Note ``sarvam-m`` is deprecated; ``sarvam-105b-conversations`` is the
  voice-workload model.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from ..config import (
    MOCK_VOICE,
    SARVAM_API_KEY,
    SARVAM_BASE,
    SARVAM_CHAT_MODEL,
    SARVAM_CONNECT_TIMEOUT_S,
    SARVAM_READ_TIMEOUT_S,
    SARVAM_REASONING_EFFORT,
    SARVAM_RETRIES,
    SARVAM_STT_MODEL,
    SARVAM_TIMEOUT_S,
    SARVAM_TTS_MODEL,
    SARVAM_TTS_SPEAKER,
)


class SarvamError(RuntimeError):
    """Raised when a Sarvam call fails after all retries."""


def describe(exc: BaseException | None) -> str:
    """A message that survives exceptions carrying no message of their own.

    httpx timeout and connect errors stringify to "", so reporting str(exc) put
    an empty string in the error field and left nothing to debug from. The type
    name is the entire useful content in exactly those cases.
    """
    if exc is None:
        return "unknown failure"
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


@dataclass
class Transcript:
    text: str
    language_code: str | None
    latency_ms: float
    mocked: bool = False
    raw: dict = field(default_factory=dict)


@dataclass
class Speech:
    audio_b64: str
    latency_ms: float
    mocked: bool = False


def _headers(json_body: bool) -> dict[str, str]:
    headers = {"api-subscription-key": SARVAM_API_KEY}
    if SARVAM_API_KEY:
        headers["Authorization"] = f"Bearer {SARVAM_API_KEY}"
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


class SarvamClient:
    def __init__(self, mock: bool | None = None) -> None:
        self.mock = MOCK_VOICE if mock is None else mock
        self._client: httpx.AsyncClient | None = None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=SARVAM_BASE,
                timeout=httpx.Timeout(SARVAM_READ_TIMEOUT_S, connect=SARVAM_CONNECT_TIMEOUT_S),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _post(self, path: str, **kwargs) -> httpx.Response:
        """POST with bounded retries on transport errors and 5xx."""
        client = await self._http()
        last: Exception | None = None
        for attempt in range(SARVAM_RETRIES + 1):
            try:
                response = await client.post(path, **kwargs)
                if response.status_code >= 500:
                    raise SarvamError(f"{path} -> HTTP {response.status_code}")
                if response.status_code >= 400:
                    raise SarvamError(f"{path} -> HTTP {response.status_code}: {response.text[:200]}")
                return response
            except (httpx.HTTPError, SarvamError) as exc:
                last = exc
                if attempt < SARVAM_RETRIES:
                    await asyncio.sleep(0.25 * (2**attempt))
        raise SarvamError(describe(last))

    # ---------------------------------------------------------------- STT ---
    async def transcribe(
        self,
        audio: bytes,
        filename: str = "audio.webm",
        language_code: str | None = None,
    ) -> Transcript:
        start = time.perf_counter()
        if self.mock:
            await asyncio.sleep(0.05)
            return Transcript(
                text="[mock transcript: no SARVAM_API_KEY configured]",
                language_code=language_code or "en-IN",
                latency_ms=(time.perf_counter() - start) * 1000,
                mocked=True,
            )

        data = {"model": SARVAM_STT_MODEL, "mode": "transcribe"}
        if language_code:
            data["language_code"] = language_code
        response = await self._post(
            "/speech-to-text",
            headers=_headers(json_body=False),
            files={"file": (filename, audio, "application/octet-stream")},
            data=data,
        )
        payload = response.json()
        return Transcript(
            text=payload.get("transcript", ""),
            language_code=payload.get("language_code"),
            latency_ms=(time.perf_counter() - start) * 1000,
            raw=payload,
        )

    # ---------------------------------------------------------------- TTS ---
    async def synthesize(self, text: str, language_code: str, speaker: str | None = None) -> Speech:
        start = time.perf_counter()
        if self.mock:
            await asyncio.sleep(0.05)
            return Speech(audio_b64="", latency_ms=(time.perf_counter() - start) * 1000, mocked=True)

        # bulbul:v2 caps at 1500 characters per request.
        response = await self._post(
            "/text-to-speech",
            headers=_headers(json_body=True),
            json={
                "text": text[:1450],
                "language_code": language_code,
                "model": SARVAM_TTS_MODEL,
                "speaker": speaker or SARVAM_TTS_SPEAKER,
                "output_audio_codec": "mp3",
            },
        )
        audios = response.json().get("audios") or [""]
        return Speech(audio_b64=audios[0], latency_ms=(time.perf_counter() - start) * 1000)

    # --------------------------------------------------------------- chat ---
    async def chat_stream(
        self,
        messages: list[dict],
        max_tokens: int = 400,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        """Yield content deltas from the chat completions SSE stream."""
        if self.mock:
            for chunk in _mock_completion(messages):
                await asyncio.sleep(0.02)
                yield chunk
            return

        client = await self._http()
        body = {
            "model": SARVAM_CHAT_MODEL,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning_effort": SARVAM_REASONING_EFFORT,
        }
        async with client.stream(
            "POST", "/v1/chat/completions", headers=_headers(json_body=True), json=body
        ) as response:
            if response.status_code >= 400:
                detail = (await response.aread())[:200].decode("utf-8", "replace")
                raise SarvamError(f"/v1/chat/completions -> HTTP {response.status_code}: {detail}")
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload in ("", "[DONE]"):
                    if payload == "[DONE]":
                        break
                    continue
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
                if delta:
                    yield delta


def _mock_completion(messages: list[dict]) -> list[str]:
    """Deterministic stand-in that echoes the grounded span back as prose."""
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    span = ""
    for line in user.splitlines():
        if line.startswith("[1]"):
            span = line[3:].strip()
            break
    body = span or "no grounded span was supplied"
    words = f"(mock generation, no API key) {body}".split(" ")
    return [w + " " for w in words]


def decode_audio(audio_b64: str) -> bytes:
    return base64.b64decode(audio_b64) if audio_b64 else b""
