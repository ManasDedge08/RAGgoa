export type Strategy =
  | "dense_passage"
  | "dense_sentence"
  | "bm25_passage"
  | "bm25_sentence"
  | "cluster";

export interface CandidateDto {
  passage_id: string;
  group_id: string;
  text: string;
  lang: string;
  query_id: number;
  strategies: Partial<Record<Strategy, number>>;
  raw_scores: Record<string, number>;
  rrf_score: number;
  fusion_rank: number;
  rerank_score: number;
  best_sentence: string;
  available_langs: string[];
  surfaced_by_langs: string[];
}

export interface ConfidenceDto {
  score: number;
  tier: "high" | "low" | "refuse";
  similarity: number;
  margin: number;
  agreement: number;
}

export interface GroundingDto {
  supported: boolean;
  lexical: number;
  semantic: number;
  reason: string;
  latency_ms: number;
}

export type StreamEvent =
  | { type: "state"; state: string; turn_id: string }
  | { type: "transcript"; text: string; language_code: string | null; latency_ms: number; mocked: boolean }
  | { type: "trace"; stage: string; [k: string]: unknown }
  | {
      type: "detection";
      detection: {
        lang: string;
        confidence: number;
        script: string;
        alternatives: string[];
        ambiguous: boolean;
      };
    }
  | { type: "guardrail"; allowed: boolean; reason: string; domain_similarity: number; latency_ms: number }
  | { type: "refusal"; text: string }
  | {
      type: "retrieval";
      query: string;
      lang: string;
      lang_confidence: number;
      cross_lingual: boolean;
      timings_ms: Record<string, number>;
      candidates: CandidateDto[];
    }
  | {
      type: "tier1";
      text: string;
      span: string;
      passage_id: string;
      source_lang: string;
      answer_lang: string;
      tier: string;
      cross_lingual: boolean;
      latency_ms: number;
      tier1_total_ms: number;
      confidence: ConfidenceDto;
      harness_ms: Record<string, number>;
    }
  | { type: "tier2_delta"; text: string }
  | {
      type: "tier2";
      text?: string;
      spoken_text?: string;
      grounding?: GroundingDto | null;
      used_fallback?: boolean;
      latency_ms?: number;
      first_token_ms?: number;
      error?: string | null;
    }
  | {
      type: "audio";
      label: string;
      audio_b64: string;
      mocked: boolean;
      latency_ms: number;
      unavailable?: string;
    }
  | { type: "error"; stage: string; message: string }
  | { type: "done"; turn: TurnDto };

export interface TurnDto {
  turn_id: string;
  query: string;
  lang: string;
  state: string;
  path: string[];
  stage_ms: Record<string, number>;
  errors: Record<string, string>;
  [k: string]: unknown;
}

export interface RaceResult {
  query: string;
  multi_strategy: {
    total_ms: number;
    stages_ms: Record<string, number>;
    top: { passage_id: string; lang: string; snippet: string }[];
  };
  naive: {
    total_ms: number;
    stages_ms: Record<string, number>;
    top: { chunk_index: number; lang: string; score: number; snippet: string }[];
  };
}

export interface MetaDto {
  corpus: Record<string, number | string>;
  languages: { code: string; name: string; sarvam: string; voice: boolean }[];
  voice_languages: string[];
  tier1_target_ms: number;
  mock_voice: boolean;
}
