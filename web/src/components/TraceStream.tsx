/**
 * Glass-box retrieval: which passages came back, which strategy found each one,
 * where fusion put it, and what the reranker did to that position.
 *
 * Rows hang from the column rule as they arrive, while Tier 2 is still
 * generating — the reasoning is visible before the answer finishes.
 */

import type { CandidateDto, Strategy } from "../types";

const STRATEGY_LABEL: Record<Strategy, string> = {
  dense_passage: "dense·psg",
  dense_sentence: "dense·snt",
  bm25_passage: "bm25·psg",
  bm25_sentence: "bm25·snt",
  cluster: "cluster",
};

const LANG_LABEL: Record<string, string> = {
  eng_Latn: "English",
  hin_Deva: "Hindi",
  tam_Taml: "Tamil",
  ben_Beng: "Bengali",
};

interface Props {
  candidates: CandidateDto[];
  queryLang: string;
  live: boolean;
}

export function TraceStream({ candidates, queryLang, live }: Props) {
  if (candidates.length === 0) {
    return (
      <p className="trace__empty">
        {live
          ? "retrieving…"
          : "Nothing retrieved yet. Ask a question and the five strategies report here as they vote."}
      </p>
    );
  }

  return (
    <div className="trace">
      {candidates.map((candidate, index) => {
        const strategies = Object.entries(candidate.strategies) as [Strategy, number][];
        const crossLingual = candidate.lang !== queryLang;
        return (
          <article className="trace__row" key={candidate.group_id}>
            <div className="trace__top">
              <span className="trace__rank">{index + 1}</span>
              <div className="trace__strategies">
                {strategies.map(([name, rank]) => (
                  <span className="strategy" key={name}>
                    {STRATEGY_LABEL[name]}
                    <sup>{rank}</sup>
                  </span>
                ))}
              </div>
              <span className="trace__scores">
                fuse #{candidate.fusion_rank} · rr {candidate.rerank_score.toFixed(3)}
              </span>
            </div>
            <p className="trace__snippet">{candidate.best_sentence || candidate.text.slice(0, 220)}</p>
            <p className="trace__langs">
              {LANG_LABEL[candidate.lang] ?? candidate.lang}
              {crossLingual && <span className="badge badge--cross"> cross-lingual</span>} · also in{" "}
              {candidate.available_langs
                .filter((l) => l !== candidate.lang)
                .map((l) => LANG_LABEL[l] ?? l)
                .join(", ") || "no other language"}
            </p>
          </article>
        );
      })}
    </div>
  );
}
