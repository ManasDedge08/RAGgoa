import type { MetaDto, RaceResult, StreamEvent } from "./types";

export const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

/** Reads an SSE body from a POST response and hands back one event at a time. */
async function* readEvents(response: Response): AsyncGenerator<StreamEvent> {
  if (!response.body) throw new Error("no response body");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (payload === "[DONE]") return;
      try {
        yield JSON.parse(payload) as StreamEvent;
      } catch {
        // A partial frame; the next chunk completes it.
      }
    }
  }
}

export async function* askText(
  query: string,
  options: { langMode: string; speak: boolean; crossEncode: boolean; languageCode?: string },
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      lang_mode: options.langMode,
      speak: options.speak,
      cross_encode: options.crossEncode,
      language_code: options.languageCode ?? null,
    }),
  });
  if (!response.ok) throw new Error(`ask failed: ${response.status}`);
  yield* readEvents(response);
}

export async function* askAudio(
  blob: Blob,
  options: { langMode: string; speak: boolean; crossEncode: boolean; languageCode?: string },
): AsyncGenerator<StreamEvent> {
  const form = new FormData();
  form.append("file", blob, "question.webm");
  form.append("lang_mode", options.langMode);
  form.append("speak", String(options.speak));
  form.append("cross_encode", String(options.crossEncode));
  if (options.languageCode) form.append("language_code", options.languageCode);

  const response = await fetch(`${API_BASE}/ask-audio`, { method: "POST", body: form });
  if (!response.ok) throw new Error(`ask-audio failed: ${response.status}`);
  yield* readEvents(response);
}

export async function fetchMeta(): Promise<MetaDto> {
  const response = await fetch(`${API_BASE}/meta`);
  if (!response.ok) throw new Error(`meta failed: ${response.status}`);
  return response.json();
}

export async function runRace(query: string): Promise<RaceResult> {
  const response = await fetch(`${API_BASE}/race`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!response.ok) throw new Error(`race failed: ${response.status}`);
  return response.json();
}

export async function fetchSamples(): Promise<Record<string, { query_id: number; text: string }[]>> {
  const response = await fetch(`${API_BASE}/sample-queries`);
  if (!response.ok) throw new Error(`samples failed: ${response.status}`);
  const data = await response.json();
  return data.samples;
}
