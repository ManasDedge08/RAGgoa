/**
 * Live P50 / P70 / P100 for the session.
 *
 * The latency report states these numbers for a 300-query benchmark, but a
 * judge watching a demo has no reason to take a committed file on trust. This
 * accumulates the same percentiles from the questions actually asked in front
 * of them, and keeps the two tiers in separate rows because averaging them
 * together is the thing this project refuses to do.
 */

export interface Percentile {
  p50: number;
  p70: number;
  p100: number;
  n: number;
}

export function percentiles(values: number[]): Percentile | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const at = (p: number) => {
    if (p >= 100) return sorted[sorted.length - 1];
    const k = ((sorted.length - 1) * p) / 100;
    const lo = Math.floor(k);
    const hi = Math.min(lo + 1, sorted.length - 1);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (k - lo);
  };
  return { p50: at(50), p70: at(70), p100: at(100), n: sorted.length };
}

interface Props {
  tier1: number[];
  tier2: number[];
  targetMs: number;
}

function fmt(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(1)} ms`;
}

function Row({
  label,
  note,
  stats,
  withinTarget,
}: {
  label: string;
  note: string;
  stats: Percentile | null;
  withinTarget?: boolean;
}) {
  return (
    <div className="pct__row">
      <div className="pct__head">
        <span className="pct__label">{label}</span>
        <span className="pct__note">{note}</span>
      </div>
      {stats ? (
        <div className="pct__values">
          <span><b>P50</b> {fmt(stats.p50)}</span>
          <span><b>P70</b> {fmt(stats.p70)}</span>
          <span><b>P100</b> {fmt(stats.p100)}</span>
          <span className="pct__n">
            {stats.n} {stats.n === 1 ? "question" : "questions"}
          </span>
          {withinTarget !== undefined && (
            <span className={`badge badge--${withinTarget ? "high" : "refuse"}`}>
              {withinTarget ? "inside budget" : "over budget"}
            </span>
          )}
        </div>
      ) : (
        <p className="pct__empty">No questions asked yet this session.</p>
      )}
    </div>
  );
}

export function Percentiles({ tier1, tier2, targetMs }: Props) {
  const one = percentiles(tier1);
  const two = percentiles(tier2);

  return (
    <section className="pct">
      <div className="column__head">
        <span>this session, measured live</span>
        <span>never averaged across tiers</span>
      </div>
      <Row
        label="Tier 1 · extractive"
        note={`retrieval to answer, against the ${targetMs} ms budget`}
        stats={one}
        withinTarget={one ? one.p100 <= targetMs : undefined}
      />
      <Row
        label="Tier 2 · generative"
        note="hosted model over the network, reported separately"
        stats={two}
      />
    </section>
  );
}
