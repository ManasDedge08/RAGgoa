/**
 * Side-by-side race: the multi-strategy pipeline against the obvious approach.
 * Same query, same machine, same corpus, same embedding model — the only
 * difference is the retrieval structure.
 */

import type { RaceResult } from "../types";

interface Props {
  result: RaceResult | null;
  running: boolean;
  onRun: () => void;
  canRun: boolean;
}

export function Race({ result, running, onRun, canRun }: Props) {
  const max = result
    ? Math.max(result.multi_strategy.total_ms, result.naive.total_ms) * 1.05
    : 1;
  const speedup =
    result && result.multi_strategy.total_ms > 0
      ? result.naive.total_ms / result.multi_strategy.total_ms
      : 0;

  return (
    <section className="race">
      <div className="race__head">
        <h2 className="race__title">Against the obvious approach</h2>
        <button className="mic" onClick={onRun} disabled={!canRun || running}>
          <span className="mic__pulse" />
          {running ? "Racing…" : "Race the last question"}
        </button>
      </div>
      <p className="race__note">
        The naive path is what a first cut looks like: documents glued together, cut into fixed
        512-character chunks, one brute-force scan over every chunk. No sparse signal, no
        sentence-level index, no fusion, no rerank. Same corpus, same encoder, same machine.
      </p>

      {!result && !running && (
        <p className="race__empty">Ask something first, then race it.</p>
      )}

      {result && (
        <>
          <div className="race__lanes">
            <div className="racelane">
              <span className="racelane__name">multi-strategy</span>
              <div className="racelane__bar">
                <div
                  className="racelane__fill"
                  style={{
                    width: `${(result.multi_strategy.total_ms / max) * 100}%`,
                    background: "#2d3e8c",
                  }}
                />
              </div>
              <span className="racelane__ms">{result.multi_strategy.total_ms.toFixed(1)} ms</span>
            </div>
            <div className="racelane">
              <span className="racelane__name">naive</span>
              <div className="racelane__bar">
                <div
                  className="racelane__fill"
                  style={{
                    width: `${(result.naive.total_ms / max) * 100}%`,
                    background: "#a6402e",
                  }}
                />
              </div>
              <span className="racelane__ms">{result.naive.total_ms.toFixed(1)} ms</span>
            </div>
          </div>
          <p className="race__verdict">
            {speedup >= 1 ? (
              <>
                Multi-strategy retrieval answered <strong>{speedup.toFixed(1)}× faster</strong> while
                running five strategies instead of one.
              </>
            ) : (
              <>
                The naive path was faster here by{" "}
                <strong>{(1 / speedup).toFixed(1)}×</strong> — it does less work. Compare what each
                one returned above before reading that as a win.
              </>
            )}
          </p>
        </>
      )}
    </section>
  );
}
