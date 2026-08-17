import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { askAudio, askText, fetchMeta, fetchSamples, runRace } from "./api";
import { Measure } from "./components/Measure";
import { Race } from "./components/Race";
import { TraceStream } from "./components/TraceStream";
import { useRecorder } from "./useRecorder";
import type {
  CandidateDto,
  ConfidenceDto,
  GroundingDto,
  MetaDto,
  RaceResult,
  StreamEvent,
} from "./types";

const STATES = ["received", "transcribe", "guard", "retrieve", "tier1", "tier2", "speak", "done"];
/** Routing modes, named by what they do rather than by their internal keys. */
const LANG_MODES = [
  { mode: "cross", label: "any language", hint: "Retrieve from all four languages" },
  { mode: "strict", label: "same language", hint: "Retrieve only in the language you asked in" },
  { mode: "pivot", label: "other languages", hint: "Answer only from sources in a different language" },
] as const;

interface TurnView {
  transcript: string;
  queryLang: string;
  detection: { script: string; alternatives: string[]; ambiguous: boolean } | null;
  guardrail: { allowed: boolean; reason: string; latency_ms: number } | null;
  candidates: CandidateDto[];
  stages: Record<string, number>;
  tier1: { text: string; total: number; tier: string; crossLingual: boolean; sourceLang: string } | null;
  confidence: ConfidenceDto | null;
  tier2: string;
  tier2Done: { grounding: GroundingDto | null; usedFallback: boolean; latency: number; error?: string | null } | null;
  unsourced: string;
  unsourcedDone: boolean;
  refusal: string | null;
  noVoice: boolean;
  state: string;
  errors: string[];
}

const EMPTY: TurnView = {
  transcript: "",
  queryLang: "eng_Latn",
  detection: null,
  guardrail: null,
  candidates: [],
  stages: {},
  tier1: null,
  confidence: null,
  tier2: "",
  tier2Done: null,
  unsourced: "",
  unsourcedDone: false,
  refusal: null,
  noVoice: false,
  state: "",
  errors: [],
};

export default function App() {
  const [meta, setMeta] = useState<MetaDto | null>(null);
  const [samples, setSamples] = useState<Record<string, { query_id: number; text: string }[]>>({});
  const [text, setText] = useState("");
  const [langMode, setLangMode] = useState<"cross" | "strict" | "pivot">("cross");
  const [speak, setSpeak] = useState(true);
  const [crossEncode, setCrossEncode] = useState(false);
  const [allowUnsourced, setAllowUnsourced] = useState(false);
  const [busy, setBusy] = useState(false);
  const [turn, setTurn] = useState<TurnView>(EMPTY);
  const [race, setRace] = useState<RaceResult | null>(null);
  const [racing, setRacing] = useState(false);
  const [lastQuery, setLastQuery] = useState("");
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const recorder = useRecorder();

  // Display names come from /meta so adding a language to the corpus needs no
  // frontend change.
  const langLabel = useCallback(
    (code: string) => meta?.languages.find((l) => l.code === code)?.name ?? code,
    [meta],
  );

  useEffect(() => {
    fetchMeta().then(setMeta).catch(() => setMeta(null));
    fetchSamples().then(setSamples).catch(() => setSamples({}));
  }, []);

  const consume = useCallback(async (stream: AsyncGenerator<StreamEvent>) => {
    setBusy(true);
    setTurn({ ...EMPTY, state: "received" });
    setRace(null);
    try {
      for await (const event of stream) {
        setTurn((prev) => reduce(prev, event));
        if (event.type === "audio" && event.audio_b64) {
          const audio = new Audio(`data:audio/mp3;base64,${event.audio_b64}`);
          audioRef.current?.pause();
          audioRef.current = audio;
          void audio.play().catch(() => undefined);
        }
      }
    } catch (err) {
      setTurn((prev) => ({
        ...prev,
        errors: [...prev.errors, err instanceof Error ? err.message : String(err)],
      }));
    } finally {
      setBusy(false);
    }
  }, []);

  const submitText = useCallback(
    (value: string, languageCode?: string) => {
      const query = value.trim();
      if (!query) return;
      setLastQuery(query);
      void consume(askText(query, { langMode, speak, crossEncode, allowUnsourced, languageCode }));
    },
    [allowUnsourced, consume, crossEncode, langMode, speak],
  );

  const toggleMic = useCallback(async () => {
    if (recorder.recording) {
      const recording = await recorder.stop();
      if (recording) {
        void consume(askAudio(recording, { langMode, speak, crossEncode, allowUnsourced }));
      }
      return;
    }
    await recorder.start();
  }, [allowUnsourced, consume, crossEncode, langMode, recorder, speak]);

  const onRace = useCallback(() => {
    if (!lastQuery) return;
    setRacing(true);
    runRace(lastQuery)
      .then(setRace)
      .catch(() => setRace(null))
      .finally(() => setRacing(false));
  }, [lastQuery]);

  const activeState = turn.state || "";
  const sampleChips = useMemo(
    () =>
      Object.entries(samples).flatMap(([lang, items]) =>
        items.slice(0, 1).map((item) => ({ lang, ...item })),
      ),
    [samples],
  );

  return (
    <div className="shell">
      <header className="masthead">
        <div className="masthead__eyebrow">
          <span>voice RAG · MS MARCO-XI</span>
          <span>four languages, one index</span>
          <span>{meta ? `${meta.corpus.passages} passages · ${meta.corpus.sentences} sentences` : "loading corpus…"}</span>
        </div>
        <h1 className="masthead__title">
          The Measure<em>.</em>
        </h1>
        <p className="masthead__scripts">
          <span>प्रश्न पूछिए</span>
          <span>கேளுங்கள்</span>
          <span>জিজ্ঞাসা করুন</span>
          <span>Ask out loud</span>
        </p>
        <p className="masthead__standfirst">
          Speak a question in Hindi, Tamil, Bengali or English. The extractive answer lands in
          milliseconds and is measured against a 200 ms budget on the line below. The spoken,
          synthesised answer follows after — separately timed, never folded into that number.
        </p>
        <p className="masthead__scope">
          <strong>What it knows:</strong> 1,200 questions from MS MARCO and the web passages
          retrieved for them — brake rotors, HSA premiums, legal definitions, NFL records. It is a
          retrieval demo, not an encyclopedia, so it will decline general-knowledge questions like
          who runs a country rather than guess at them. The examples below are drawn from the
          corpus itself.
        </p>
      </header>

      <Measure
        stages={turn.stages}
        tier1Total={turn.tier1?.total ?? null}
        naiveTotal={race?.naive.total_ms ?? null}
        targetMs={meta?.tier1_target_ms ?? 200}
      />

      {meta?.mock_voice && (
        <p className="notice">
          Voice services are stubbed: no Sarvam API key on the server. Retrieval, timing and the
          trace are real; speech in, speech out and Tier 2 text are placeholders.
        </p>
      )}
      {turn.detection?.ambiguous && (
        <p className="notice">
          <strong>{turn.detection.script}</strong> is written by more than one language here, so
          the script alone cannot say which you meant. Answering as{" "}
          <strong>{langLabel(turn.queryLang)}</strong>. Ask again as:{" "}
          {turn.detection.alternatives.map((code) => (
            <button
              key={code}
              className="chip chip--inline"
              onClick={() =>
                submitText(
                  turn.transcript || lastQuery,
                  meta?.languages.find((l) => l.code === code)?.sarvam,
                )
              }
            >
              {langLabel(code)}
            </button>
          ))}{" "}
          Speaking the question avoids the guess entirely — Sarvam returns the language.
        </p>
      )}
      {turn.noVoice && (
        <p className="notice">
          {langLabel(turn.queryLang)} has speech recognition but no Sarvam voice, so this answer is
          text only.
        </p>
      )}
      {recorder.error && <p className="notice notice--error">{recorder.error}</p>}
      {turn.errors.map((message) => (
        <p className="notice notice--error" key={message}>
          {message}
        </p>
      ))}

      <div className="ask">
        <button
          className="mic"
          data-recording={recorder.recording}
          onClick={toggleMic}
          disabled={busy && !recorder.recording}
        >
          <span className="mic__pulse" />
          {recorder.recording ? "Stop and ask" : "Ask out loud"}
        </button>

        <form
          className="ask__field"
          onSubmit={(event) => {
            event.preventDefault();
            submitText(text);
          }}
        >
          <input
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="…or type the question"
            aria-label="Question"
          />
          <button type="submit" disabled={busy}>
            Ask
          </button>
        </form>

        <div className="modes" role="group" aria-label="Language routing">
          {LANG_MODES.map(({ mode, label, hint }) => (
            <button
              key={mode}
              aria-pressed={langMode === mode}
              onClick={() => setLangMode(mode)}
              title={hint}
            >
              {label}
            </button>
          ))}
          <button aria-pressed={speak} onClick={() => setSpeak((value) => !value)}>
            {speak ? "voice on" : "voice off"}
          </button>
          <button
            aria-pressed={allowUnsourced}
            onClick={() => setAllowUnsourced((value) => !value)}
            title="If the corpus has no answer, let the model answer from its own knowledge — clearly marked as unsourced"
          >
            general knowledge
          </button>
          <button
            aria-pressed={crossEncode}
            onClick={() => setCrossEncode((value) => !value)}
            title="Re-score the top 10 with a cross-encoder: better ranking, ~70 ms more"
          >
            precision
          </button>
        </div>
      </div>

      {sampleChips.length > 0 && (
        <div className="samples">
          <span className="samples__label">real questions from the corpus</span>
          {sampleChips.map((chip) => (
            <button
              className="chip"
              key={`${chip.lang}-${chip.query_id}`}
              onClick={() => {
                setText(chip.text);
                submitText(chip.text);
              }}
            >
              {chip.text}
            </button>
          ))}
        </div>
      )}

      <div className="board">
        <section>
          <div className="column__head">
            <span>the answer</span>
            <span>{turn.tier1 ? langLabel(turn.queryLang) : ""}</span>
          </div>

          <div className="turnstate">
            {STATES.map((state) => (
              <span
                className="turnstate__step"
                key={state}
                data-active={activeState === state}
                data-terminal={["degraded", "refused", "failed"].includes(activeState) && state === "done" ? activeState : undefined}
              >
                {state}
              </span>
            ))}
          </div>

          <div className="transcript" data-empty={!turn.transcript}>
            {turn.transcript || "Transcript appears here as Sarvam returns it."}
          </div>

          {turn.refusal && (
            <article className="answer">
              <div className="answer__meta">
                <span className="badge badge--refuse">refused</span>
                <span>{turn.guardrail?.reason}</span>
              </div>
              <p className="answer__body">{turn.refusal}</p>
            </article>
          )}

          {turn.tier1 && (
            <article className="answer">
              <div className="answer__meta">
                <span className="answer__stamp">tier 1 · {turn.tier1.total.toFixed(1)} ms</span>
                <span>extractive, grounded by construction</span>
                <span className={`badge badge--${turn.tier1.tier}`}>{turn.tier1.tier} confidence</span>
                {turn.tier1.crossLingual && (
                  <span className="badge badge--cross">
                    source: {langLabel(turn.tier1.sourceLang)}
                  </span>
                )}
              </div>
              <p className="answer__body">{turn.tier1.text}</p>
            </article>
          )}

          {(turn.unsourced || turn.unsourcedDone) && (
            <article className="answer answer--unsourced">
              <div className="answer__meta">
                <span className="badge badge--unsourced">not from the corpus</span>
                <span>the model's own knowledge — nothing was retrieved to check it against</span>
              </div>
              <p className="answer__body">{turn.unsourced}</p>
            </article>
          )}

          {(turn.tier2 || turn.tier2Done) && (
            <article className={`answer answer--tier2 ${busy && !turn.tier2Done ? "answer--streaming" : ""}`}>
              <div className="answer__meta">
                <span>tier 2 · synthesised</span>
                <span>
                  {turn.tier2Done?.latency
                    ? `${(turn.tier2Done.latency / 1000).toFixed(2)} s — outside the budget by design`
                    : "streaming…"}
                </span>
                {turn.tier2Done?.grounding && (
                  <span
                    className={`badge badge--${turn.tier2Done.grounding.supported ? "high" : "refuse"}`}
                  >
                    {turn.tier2Done.grounding.supported ? "grounded" : "unsupported"} · lex{" "}
                    {turn.tier2Done.grounding.lexical.toFixed(2)} · sem{" "}
                    {turn.tier2Done.grounding.semantic.toFixed(2)}
                  </span>
                )}
                {turn.tier2Done?.usedFallback && (
                  <span className="badge badge--low">fell back to tier 1</span>
                )}
              </div>
              <p className="answer__body">{turn.tier2 || turn.tier2Done?.error}</p>
            </article>
          )}
        </section>

        <section>
          <div className="column__head">
            <span>what retrieval saw</span>
            <span>
              {turn.confidence
                ? `confidence ${turn.confidence.score.toFixed(2)} · agreement ${(turn.confidence.agreement * 5).toFixed(0)}/5`
                : ""}
            </span>
          </div>
          <TraceStream
            candidates={turn.candidates}
            queryLang={turn.queryLang}
            live={busy}
            langLabel={langLabel}
          />
        </section>
      </div>

      <Race result={race} running={racing} onRun={onRace} canRun={Boolean(lastQuery)} />

      <footer className="colophon">
        <span>corpus: ai4bharat/MSMARCO-XI</span>
        <span>encoder: {String(meta?.corpus.embed_model ?? "—")}</span>
        <span>speech: Sarvam saaras + bulbul</span>
        <span>tier 2: sarvam-105b-conversations</span>
      </footer>
    </div>
  );
}

function reduce(prev: TurnView, event: StreamEvent): TurnView {
  switch (event.type) {
    case "state":
      return { ...prev, state: event.state };
    case "transcript":
      return { ...prev, transcript: event.text };
    case "guardrail":
      return {
        ...prev,
        guardrail: { allowed: event.allowed, reason: event.reason, latency_ms: event.latency_ms },
        stages: { ...prev.stages, guardrail: event.latency_ms },
      };
    case "refusal":
      return { ...prev, refusal: event.text, state: "refused" };
    case "detection":
      return { ...prev, detection: event.detection, queryLang: event.detection.lang };
    case "retrieval":
      return {
        ...prev,
        candidates: event.candidates,
        queryLang: event.lang,
        transcript: prev.transcript || event.query,
        stages: { ...prev.stages, ...event.timings_ms },
      };
    case "tier1":
      return {
        ...prev,
        confidence: event.confidence,
        stages: { ...prev.stages, ...event.harness_ms, extract: event.latency_ms },
        tier1: {
          text: event.text,
          total: event.tier1_total_ms,
          tier: event.tier,
          crossLingual: event.cross_lingual,
          sourceLang: event.source_lang,
        },
      };
    case "tier2_delta":
      return { ...prev, tier2: prev.tier2 + event.text };
    case "unsourced_delta":
      return { ...prev, unsourced: prev.unsourced + event.text };
    case "unsourced":
      return { ...prev, unsourced: event.text, unsourcedDone: true };
    case "tier2":
      return {
        ...prev,
        tier2: event.text || prev.tier2,
        tier2Done: {
          grounding: event.grounding ?? null,
          usedFallback: Boolean(event.used_fallback),
          latency: event.latency_ms ?? 0,
          error: event.error ?? null,
        },
      };
    case "audio":
      return event.unavailable ? { ...prev, noVoice: true } : prev;
    case "error":
      return { ...prev, errors: [...prev.errors, `${event.stage}: ${event.message}`] };
    case "done":
      return { ...prev, state: event.turn.state };
    default:
      return prev;
  }
}
