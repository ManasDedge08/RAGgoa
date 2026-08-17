"""Central configuration for the voice-enabled RAG system."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Read .env into the environment without adding a dependency.

    Real environment variables win, so a value set by Render or Vercel is never
    overridden by a stray local file.
    """
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_dotenv()
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
INDEX_DIR = DATA_DIR / "index"
CACHE_DIR = DATA_DIR / "cache"
REPORT_DIR = ROOT / "reports"

for _d in (CORPUS_DIR, INDEX_DIR, CACHE_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- dataset ---
HF_DATASET = "ai4bharat/MSMARCO-XI"

# Languages carried end to end. Each entry maps the dataset's language code to
# the Sarvam speech-to-text locale and a display name.
LANGUAGES = {
    "eng_Latn": {"sarvam": "en-IN", "name": "English", "file": None},
    "hin_Deva": {"sarvam": "hi-IN", "name": "Hindi", "file": "hin"},
    "tam_Taml": {"sarvam": "ta-IN", "name": "Tamil", "file": "tam"},
    "ben_Beng": {"sarvam": "bn-IN", "name": "Bengali", "file": "ben"},
}
# Languages that have their own parquet slice in the dataset.
DATASET_LANGS = [k for k, v in LANGUAGES.items() if v["file"]]

# Number of MS MARCO query ids sampled into the demo corpus. Every sampled id
# is materialised in all four languages so cross-lingual retrieval is provable.
N_QUERY_IDS = int(os.getenv("RAG_N_QUERY_IDS", "1200"))
RANDOM_SEED = 20260822

# ------------------------------------------------------------------ models ---
# Multilingual E5. Asymmetric by design: queries and passages get different
# prefixes, which is what MS MARCO-style retrieval needs. A symmetric paraphrase
# encoder was measured first and scored half the recall at the same CPU cost
# (see reports/encoder_ab.md).
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "intfloat/multilingual-e5-small")
EMBED_DIM = int(os.getenv("RAG_EMBED_DIM", "384"))
QUERY_PREFIX = os.getenv("RAG_QUERY_PREFIX", "query: ")
PASSAGE_PREFIX = os.getenv("RAG_PASSAGE_PREFIX", "passage: ")
# Optional cross-encoder rerank ("precision mode"). Measured on an Apple M4 at
# depth 10: recall@5 43.3% -> 50.0%, precision@1 21.7% -> 24.2%, for ~81 ms —
# which fits the 200 ms budget here but would not on a slower box, so it is off
# by default and switchable per request. See reports/cross_encoder_eval.json.
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
CROSS_ENCODER_DEPTH = int(os.getenv("RAG_CROSS_ENCODER_DEPTH", "10"))
CROSS_ENCODER_MAX_CHARS = 500
CROSS_ENCODER_DEFAULT = os.getenv("RAG_CROSS_ENCODER", "") == "1"

# ------------------------------------------------------------------ sarvam ---
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_BASE = "https://api.sarvam.ai"
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v4")
# bulbul:v2, not v3. Measured on answer-length Hindi (about 200 characters):
# v2 returns in ~1.0 s, v3 in ~6.0 s for the same audio. v3 scales badly with
# length, and a voice demo cannot spend six seconds before it speaks.
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v2")
SARVAM_TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "anushka")
# sarvam-m is deprecated; the conversational 105b is the voice-workload model.
SARVAM_CHAT_MODEL = os.getenv("SARVAM_CHAT_MODEL", "sarvam-105b-conversations")
# Tier 2 reads three passages and speaks two sentences; extra reasoning effort
# buys nothing here and costs first-token latency.
SARVAM_REASONING_EFFORT = os.getenv("SARVAM_REASONING_EFFORT", "low")
SARVAM_TIMEOUT_S = float(os.getenv("SARVAM_TIMEOUT_S", "20"))
SARVAM_RETRIES = int(os.getenv("SARVAM_RETRIES", "2"))
# With no key present the pipeline still runs end to end against stub voice and
# chat providers, so the retrieval demo is never blocked on credentials.
MOCK_VOICE = os.getenv("RAG_MOCK_VOICE", "") == "1" or not SARVAM_API_KEY

# Pre-retrieval domain filtering is off by default. Calibration showed it
# barely separates off-topic from in-corpus questions (both are well-formed
# questions), and refusing before retrieval hides the reasoning trace that
# makes a refusal legible. The evidence-based refusal in the confidence tier is
# both stronger and explainable. Unsafe-pattern checks always run.
DOMAIN_FILTER_ENABLED = os.getenv("RAG_DOMAIN_FILTER", "") == "1"

# ---------------------------------------------------------------- indexing ---
N_QUERY_CLUSTERS = int(os.getenv("RAG_N_CLUSTERS", "160"))
# Fixed-width chunk size for the naive baseline index only.
CHUNK_CHARS = int(os.getenv("RAG_CHUNK_CHARS", "512"))
# Minimum sentence length, counted in tokens rather than characters. A
# character floor silently over-merges Indic sentences: Devanagari, Tamil and
# Bengali pack the same words into far fewer code points than Latin, so a
# 40-character rule splits English correctly and glues Hindi together.
MIN_SENTENCE_TOKENS = 6

# --------------------------------------------------------------- retrieval ---
TOP_K_PER_STRATEGY = 20
RRF_K = 60
RERANK_CANDIDATES = 20
FINAL_K = 5

# ----------------------------------------------------------------- latency ---
TIER1_TARGET_MS = 200.0
