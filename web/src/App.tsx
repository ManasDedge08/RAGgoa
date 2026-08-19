import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { askAudio, askText, fetchMeta, fetchSamples, runRace } from "./api";
import { Measure } from "./components/Measure";
import { Percentiles } from "./components/Percentiles";
import { Race } from "./components/Race";
import { TraceStream } from "./components/TraceStream";
import { useAudioPlayer } from "./useAudioPlayer";
import { useRecorder } from "./useRecorder";
import { useTheme } from "./useTheme";
import { BeachBand, Moon, Palm, Sun } from "./components/Scenery";
import { DrinkSticker, HibiscusSticker } from "./components/Stickers";
import { MicIcon } from "./components/MicIcon";
import type {
  CandidateDto,
  ConfidenceDto,
  GroundingDto,
  MetaDto,
  RaceResult,
  StreamEvent,
} from "./types";

const STATES = ["received", "transcribe", "guard", "retrieve", "tier1", "tier2", "speak", "done"];
/** The people who built it. Profile URLs are kept bare — the share links these
 *  came from carried utm parameters, which are tracking for whoever generated
 *  them and no use to a reader. */
const CREW = [
  { name: "Manas Dedge", url: "https://www.linkedin.com/in/manas-dedge/" },
  { name: "Rahul Kotyal", url: "https://www.linkedin.com/in/rahul-kotyal-279996220" },
  { name: "Atharv Bhosale", url: "https://www.linkedin.com/in/atharvbhosale555" },
] as const;
/** Button face per theme: what you are on now, and what one click does next. */
const THEME_LABEL: Record<string, { text: string; title: string }> = {
  light: { text: "theme · light", title: "Light theme — click for dark" },
  dark: { text: "theme · dark", title: "Dark theme — click for light" },
};
/** Routing modes, named by what they do rather than by their internal keys. The
 *  count in the "any language" hint comes from /meta, so it tracks whatever
 *  RAG_LANGUAGES the server was built with instead of hard-coding four. */
const langModes = (count: number) =>
  [
    {
      mode: "cross",
      label: "any language",
      hint: count ? `Retrieve from all ${count} indexed languages` : "Retrieve from every indexed language",
    },
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
  const [raceError, setRaceError] = useState<string | null>(null);
  const [lastQuery, setLastQuery] = useState("");
  // Percentiles for the questions asked in front of the viewer, kept per tier.
  const [tier1Times, setTier1Times] = useState<number[]>([]);
  const [tier2Times, setTier2Times] = useState<number[]>([]);
  const recorder = useRecorder();
  const player = useAudioPlayer();
  // The sample row scrolls sideways. Touch can swipe it; a mouse cannot, and
  // the fade at its right edge says "there is more" without offering a way to
  // reach it. These two track whether there is anything left to reach.
  const stripRef = useRef<HTMLDivElement>(null);
  const [stripAt, setStripAt] = useState({ start: true, end: true });
  const [refreshing, setRefreshing] = useState(false);

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
          player.play(event.audio_b64);
        }
        if (event.type === "tier1") {
          setTier1Times((prev) => [...prev, event.tier1_total_ms]);
        }
        if (event.type === "tier2" && event.latency_ms) {
          setTier2Times((prev) => [...prev, event.latency_ms as number]);
        }
        if (event.type === "transcript" && event.text) {
          // Spoken questions have to reach the race too; the transcript is the
          // only place the text exists.
          setLastQuery(event.text);
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
  }, [player]);

  const submitText = useCallback(
    (value: string, languageCode?: string) => {
      const query = value.trim();
      if (!query) return;
      // Satisfy autoplay policy now, while this click still counts as a
      // gesture; the spoken answer arrives seconds later.
      if (speak) player.unlock();
      setLastQuery(query);
      void consume(askText(query, { langMode, speak, crossEncode, allowUnsourced, languageCode }));
    },
    [allowUnsourced, consume, crossEncode, langMode, player, speak],
  );

  const toggleMic = useCallback(async () => {
    if (speak) player.unlock();
    if (recorder.recording) {
      const recording = await recorder.stop();
      if (recording) {
        void consume(askAudio(recording, { langMode, speak, crossEncode, allowUnsourced }));
      }
      return;
    }
    await recorder.start();
  }, [allowUnsourced, consume, crossEncode, langMode, player, recorder, speak]);

  const onRace = useCallback(() => {
    if (!lastQuery) return;
    setRacing(true);
    setRaceError(null);
    runRace(lastQuery)
      .then(setRace)
      .catch((err: unknown) => {
        setRace(null);
        setRaceError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => setRacing(false));
  }, [lastQuery]);

  const readStrip = useCallback(() => {
    const el = stripRef.current;
    if (!el) return;
    const max = el.scrollWidth - el.clientWidth;
    // A pixel of slack: sub-pixel layout leaves scrollLeft a hair short of max
    // on some zoom levels, which would strand the forward arrow enabled.
    setStripAt({ start: el.scrollLeft <= 1, end: el.scrollLeft >= max - 1 });
  }, []);

  useEffect(() => {
    readStrip();
    window.addEventListener("resize", readStrip);
    return () => window.removeEventListener("resize", readStrip);
  }, [readStrip, samples]);

  const refreshSamples = useCallback(() => {
    setRefreshing(true);
    fetchSamples()
      .then(setSamples)
      .catch(() => {
        /* Keep the questions already on screen: a failed refresh should cost
           the reader nothing they had. */
      })
      .finally(() => {
        setRefreshing(false);
        // Back to the first question, or the new row starts mid-scroll and
        // looks like it lost the ones before it.
        stripRef.current?.scrollTo({ left: 0, behavior: "auto" });
      });
  }, []);

  const nudgeStrip = useCallback((direction: 1 | -1) => {
    const el = stripRef.current;
    if (!el) return;
    // Most of a screenful, not all of it: a chip left in view is the thread
    // back to where you were.
    const step = el.clientWidth * 0.8 * direction;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    el.scrollBy({ left: step, behavior: reduced ? "auto" : "smooth" });
  }, []);

  const { theme, cycle: cycleTheme } = useTheme();

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
      <header className="masthead hero">
        <BeachBand className="hero__band" />
        {/* Both hang in the same spot the whole time. One is always the one
            that is up; the swap is a set and a rise, not a replacement. */}
        <Sun className="hero__sun hero__sun--day" />
        <Moon className="hero__sun hero__sun--night" />
        <Palm className="palm palm--left" />
        <Palm className="palm palm--right" flip />
        <DrinkSticker className="sticker sticker--drink" />
        <div className="masthead__eyebrow">
          <span>voice RAG · MS MARCO-XI</span>
          <span>{meta ? `${meta.languages.length} languages, one index` : "one index"}</span>
          <button
            className="theme"
            onClick={cycleTheme}
            title={THEME_LABEL[theme].title}
            aria-label={THEME_LABEL[theme].title}
          >
            {THEME_LABEL[theme].text}
          </button>
        </div>
        {/* Wordmark and corpus size share one baseline: the two things true
            about this page before a question is asked. */}
        <div className="masthead__lockup">
          <h1 className="masthead__title">
            Peoples<em>.</em>
          </h1>
          <span className="masthead__tagline">brewed in Goa, poured in eleven languages</span>
          <dl className="masthead__figures">
            <div>
              <dt>passages</dt>
              <dd>{meta ? meta.corpus.passages.toLocaleString() : "—"}</dd>
            </div>
            <div>
              <dt>sentences</dt>
              <dd>{meta ? meta.corpus.sentences.toLocaleString() : "—"}</dd>
            </div>
          </dl>
        </div>
        {/* Sentence case, not the eyebrow's caps: eleven language names in
            tracked-out uppercase read as a wall rather than a list. */}
        <p className="masthead__langs">
          {meta
            ? meta.languages.map((l, i) => (
                <span key={l.code} className="masthead__lang">
                  {i > 0 && <span aria-hidden="true"> · </span>}
                  {l.name}
                </span>
              ))
            : "loading corpus…"}
        </p>
      </header>

      {/* Above the fold: the answer and what retrieval saw, then the
          controls that produce them. The measurements moved below, where
          they are read after a question rather than before one. */}
      <main className="landing">
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

        {/* One deck. The corpus samples and the routing pills ride the top
            row, the two ways to ask sit under them. Source order is the phone's
            order; the desktop's two columns are grid areas over the top of it. */}
        <div className="ask">
          {sampleChips.length > 0 && (
            <div className="samples">
              <HibiscusSticker className="sticker sticker--hibiscus" />
              <div className="samples__head">
                {/* The label instructs and nothing else. What these questions
                    are — genuine MS MARCO-XI queries rather than demo copy — is
                    already said in the masthead and the colophon, and "corpus"
                    is our word for it, not the reader's. */}
                <span className="samples__label">tap any question to ask it</span>
                <div className="samples__nav">
                  <button
                    className="samples__arrow"
                    onClick={refreshSamples}
                    disabled={refreshing}
                    aria-label="Draw eleven new questions from the corpus"
                    title="Draw eleven new questions from the corpus"
                  >
                    <Refresh spinning={refreshing} />
                  </button>
                  <button
                    className="samples__arrow"
                    onClick={() => nudgeStrip(-1)}
                    disabled={stripAt.start}
                    aria-label="Earlier questions"
                  >
                    <Chevron />
                  </button>
                  <button
                    className="samples__arrow"
                    onClick={() => nudgeStrip(1)}
                    disabled={stripAt.end}
                    aria-label="More questions"
                  >
                    <Chevron forward />
                  </button>
                </div>
              </div>
              {/* One row that scrolls rather than a grid that grows: eleven
                  languages wrap to three rows at this width, and the deck has to
                  stay the height of the pills beside it. */}
              <div className="samples__strip" ref={stripRef} onScroll={readStrip}>
                {sampleChips.map((chip) => (
                  <button
                    className="chip"
                    key={`${chip.lang}-${chip.query_id}`}
                    onClick={() => {
                      setText(chip.text);
                      submitText(chip.text);
                    }}
                  >
                    {/* Which language you are about to ask in. Eleven scripts
                        run past here unlabelled otherwise, which makes the row
                        demonstrate the claim while naming none of it. */}
                    <span className="chip__lang">{langLabel(chip.lang)}</span>
                    {chip.text}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="modes" role="group" aria-label="Language routing">
            {langModes(meta?.languages.length ?? 0).map(({ mode, label, hint }) => (
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
              title="Re-score the top 10 with a cross-encoder: better ranking, and by far the slowest control here — about 70 ms on Apple silicon, about a second on the deployed instance"
            >
              precision
            </button>
          </div>

        {/* Both ways of asking share one row so the field can take every
            pixel the mic does not, rather than stopping at the column
            above it. */}
        <div className="asker">
            <button
              className="mic"
              data-recording={recorder.recording}
              onClick={toggleMic}
              disabled={busy && !recorder.recording}
            >
              <MicIcon className="mic__icon" />
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
        </div>
        </div>

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
        {player.blocked && (
          <p className="notice">
            The browser blocked audio that it did not start itself.{" "}
            <button className="chip chip--inline" onClick={player.playPending}>
              Play the answer
            </button>
          </p>
        )}
        {recorder.error && <p className="notice notice--error">{recorder.error}</p>}
        {turn.errors.map((message) => (
          <p className="notice notice--error" key={message}>
            {message}
          </p>
        ))}
      </main>

      <section className="measured">
        <Percentiles
          tier1={tier1Times}
          tier2={tier2Times}
          targetMs={meta?.tier1_target_ms ?? 200}
        />

        <Measure
          stages={turn.stages}
          tier1Total={turn.tier1?.total ?? null}
          naiveTotal={race?.naive.total_ms ?? null}
          targetMs={meta?.tier1_target_ms ?? 200}
        />
      </section>

      <Race
        result={race}
        running={racing}
        onRun={onRace}
        canRun={Boolean(lastQuery)}
        error={raceError}
        query={lastQuery}
      />

      {/* Who built it, above the machine facts rather than buried among them:
          the colophon lists what the system is made of, and people are not a
          dependency. */}
      <section className="crew">
        <h2 className="crew__label">meet the developers</h2>
        <ul className="crew__list">
          {CREW.map((person) => (
            <li key={person.name}>
              <a href={person.url} target="_blank" rel="noopener noreferrer">
                {person.name}
              </a>
            </li>
          ))}
        </ul>
        <p className="crew__note">
          Honourable mention: <strong>Claude</strong>, who wrote a good deal of
          this and measured the rest.
        </p>
      </section>

      <footer className="colophon">
        <span>corpus: ai4bharat/MSMARCO-XI</span>
        <span>encoder: {String(meta?.corpus.embed_model ?? "—")}</span>
        <span>speech: Sarvam saaras + bulbul</span>
        <span>tier 2: sarvam-105b-conversations</span>
      </footer>
    </div>
  );
}

/** A ring left open where the arrowhead sits, so the shape reads as a cycle
 *  rather than a circle. Spins only while a draw is in flight. */
function Refresh({ spinning }: { spinning?: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      className={spinning ? "is-spinning" : undefined}
    >
      <path d="M20 12a8 8 0 1 1-2.3-5.6" />
      <path d="M20 4v4.6h-4.6" />
    </svg>
  );
}

/** One stroke, mirrored for the other direction. Drawn rather than imported,
 *  like every other mark on this page, so it takes the button's colour. */
function Chevron({ forward }: { forward?: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      style={forward ? undefined : { transform: "scaleX(-1)" }}
    >
      <path d="M9 5l7 7-7 7" />
    </svg>
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
