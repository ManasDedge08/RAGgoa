/**
 * The measure: one 0–200 ms scale that the whole page hangs from.
 *
 * Tier 1's stages are drawn as adjoining segments on the top lane, so the
 * reader sees where the time actually goes. The naive baseline runs on the
 * lane below at the identical scale — when it overruns the budget marker, it
 * overruns it visibly rather than in a footnote.
 */

// Colours come from CSS custom properties rather than literals so the dark
// palette can restate them: the light indigos are unreadable on a dark lane.
const SEGMENTS: { key: string; label: string; color: string }[] = [
  { key: "guardrail", label: "guardrail", color: "var(--seg-guardrail)" },
  { key: "embed", label: "embed", color: "var(--seg-embed)" },
  { key: "dense", label: "dense", color: "var(--seg-dense)" },
  { key: "bm25", label: "bm25", color: "var(--seg-bm25)" },
  { key: "cluster", label: "cluster", color: "var(--seg-cluster)" },
  { key: "fusion", label: "fusion", color: "var(--seg-fusion)" },
  { key: "rerank", label: "rerank", color: "var(--seg-rerank)" },
  { key: "relevance", label: "relevance gate", color: "var(--seg-relevance)" },
  { key: "cross_encoder", label: "cross-encoder", color: "var(--seg-cross-encoder)" },
  { key: "extract", label: "extract", color: "var(--seg-extract)" },
];

const TICKS = [0, 50, 100, 150, 200];

interface Props {
  stages: Record<string, number>;
  tier1Total: number | null;
  naiveTotal: number | null;
  targetMs: number;
}

/** Scale runs to the target, or further when something actually overran it. */
function scaleMax(targetMs: number, ...totals: (number | null)[]): number {
  const highest = Math.max(targetMs, ...totals.map((t) => t ?? 0));
  return highest <= targetMs ? targetMs : Math.ceil((highest * 1.08) / 50) * 50;
}

export function Measure({ stages, tier1Total, naiveTotal, targetMs }: Props) {
  const max = scaleMax(targetMs, tier1Total, naiveTotal);
  const pct = (ms: number) => `${Math.max(0, Math.min(100, (ms / max) * 100))}%`;
  const present = SEGMENTS.filter((s) => (stages[s.key] ?? 0) > 0);

  return (
    <div className="measure">
      <div className="measure__label">
        <span>
          the measure — <strong>tier 1</strong> against a {targetMs} ms budget
        </span>
        <span>
          {tier1Total !== null ? (
            <strong>{tier1Total.toFixed(1)} ms</strong>
          ) : (
            "awaiting a question"
          )}
        </span>
      </div>

      <div className="measure__track" data-empty={tier1Total === null ? "true" : "false"}>
        {TICKS.filter((t) => t <= max).map((t) => (
          // The last label would hang off the right edge of the track, so it
          // flips to sit inside it.
          <div
            className={`measure__tick${t === max ? " measure__tick--end" : ""}`}
            key={t}
            style={{ left: pct(t) }}
          >
            <span>{t}</span>
          </div>
        ))}
        <div className="measure__budget" style={{ left: pct(targetMs) }} />

        <div className="lane lane--tier1">
          <span className="lane__tag">tier 1</span>
          {present.map((s) => (
            <div
              key={s.key}
              className="lane__seg"
              title={`${s.label} ${stages[s.key].toFixed(2)} ms`}
              style={{ width: pct(stages[s.key]), background: s.color }}
            />
          ))}
          {tier1Total !== null && (
            <span className="lane__total" style={{ left: pct(tier1Total) }}>
              {tier1Total.toFixed(1)} ms
            </span>
          )}
        </div>

        {naiveTotal !== null && (
          <div className="lane lane--naive">
            <span className="lane__tag">naive</span>
            <div
              className="lane__seg"
              style={{ width: pct(naiveTotal), background: "var(--naive)" }}
              title={`naive baseline ${naiveTotal.toFixed(2)} ms`}
            />
            <span className="lane__total" style={{ left: pct(naiveTotal), color: "var(--naive)" }}>
              {naiveTotal.toFixed(1)} ms
            </span>
          </div>
        )}
      </div>

      <div className="measure__legend">
        {SEGMENTS.map((s) => (
          <span key={s.key}>
            <b style={{ background: s.color }} />
            {s.label}
            {stages[s.key] ? ` ${stages[s.key].toFixed(2)}` : ""}
          </span>
        ))}
        <span>tier 2 is not drawn here — it is a network call, reported separately</span>
      </div>
    </div>
  );
}
