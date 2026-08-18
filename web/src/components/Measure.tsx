/**
 * The Measure: one 0–200 ms scale that the whole page hangs from.
 *
 * Tier 1's stages are drawn as adjoining segments on the top lane, so the
 * reader sees where the time actually goes. The naive baseline runs on the
 * lane below at the identical scale — when it overruns the budget marker, it
 * overruns it visibly rather than in a footnote.
 */

const SEGMENTS: { key: string; label: string; color: string }[] = [
  { key: "guardrail", label: "guardrail", color: "#8892b8" },
  { key: "embed", label: "embed", color: "#2d3e8c" },
  { key: "dense", label: "dense", color: "#3f56b8" },
  { key: "bm25", label: "bm25", color: "#5b74d1" },
  { key: "cluster", label: "cluster", color: "#7f92dd" },
  { key: "fusion", label: "fusion", color: "#2e7d6b" },
  { key: "rerank", label: "rerank", color: "#4a9e88" },
  { key: "relevance", label: "relevance gate", color: "#2e7d6b" },
  { key: "cross_encoder", label: "cross-encoder", color: "#a6402e" },
  { key: "extract", label: "extract", color: "#d99a1f" },
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

      <div className="measure__track">
        {TICKS.filter((t) => t <= max).map((t) => (
          <div className="measure__tick" key={t} style={{ left: pct(t) }}>
            <span>{t}</span>
          </div>
        ))}
        <div className="measure__budget" style={{ left: pct(targetMs) }}>
          <span>{targetMs} ms budget</span>
        </div>

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
              style={{ width: pct(naiveTotal), background: "#a6402e" }}
              title={`naive baseline ${naiveTotal.toFixed(2)} ms`}
            />
            <span className="lane__total" style={{ left: pct(naiveTotal), color: "#a6402e" }}>
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
