"""Multi-strategy retrieval: BM25 + dense + cluster signals fused with RRF.

Five independent strategies vote on passages:

1. ``dense_passage``   - semantic nearest neighbours at passage granularity
2. ``dense_sentence``  - semantic nearest neighbours at sentence granularity,
   rolled up to their parent passage (catches passages whose overall topic is
   diffuse but which contain one highly on-point sentence)
3. ``bm25_passage``    - lexical match at passage granularity
4. ``bm25_sentence``   - lexical match at sentence granularity
5. ``cluster``         - the query is matched against clusters of known training
   queries; the gold passages of the nearest clusters are proposed as candidates

Their ranked lists are fused with reciprocal rank fusion, then a cheap feature
re-scorer reorders the top candidates. Everything here runs inside the Tier 1
latency budget, so no network calls and no cross-encoder by default.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from ..config import (
    CROSS_ENCODER_DEPTH,
    CROSS_ENCODER_MAX_CHARS,
    FINAL_K,
    LANGUAGE_CODES,
    RERANK_CANDIDATES,
    RRF_K,
    TOP_K_PER_STRATEGY,
)
from ..index.text import tokenize
from .lang import detect_lang
from .store import Store, Unit, get_store

LangMode = Literal["cross", "strict", "pivot"]
TraceCb = Callable[[str, dict], None]

# Over-fetch before language routing: HNSW cannot filter during the walk, so
# candidates are filtered afterwards and the fetch depth compensates. Every
# passage exists once per indexed language, so a raw hit list of depth N yields
# roughly N / len(LANGUAGE_CODES) distinct passages in cross mode, and the same
# again in strict mode where all but one language is discarded. Scaling with the
# language count keeps the post-merge yield constant as languages are added.
OVERFETCH = int(os.getenv("RAG_OVERFETCH", str(len(LANGUAGE_CODES))))

# Set by scripts/eval_retrieval.py to ablate one strategy at a time.
DROP_STRATEGY: str | None = None

# Measured, not guessed. Fusion is what finds the candidates; dense similarity
# is what orders them. The language bonus is zero because candidate grouping
# already presents the asker's language when that variant was retrieved, so
# adding it again here only cost recall (38.5% -> 37.5%). Weighting BM25 score, fusion position and token overlap
# into the final ordering measurably hurt: recall@5 fell from 38.5% to 30.0%
# and MRR@5 from 0.263 to 0.202 on the same 200 queries, because a lexically
# noisy passage can out-rank the actual answer. See reports/retrieval_eval.json.
RERANK_WEIGHTS = {
    "dense": 1.00,
    "bm25": 0.0,
    "rrf": 0.0,
    "overlap": 0.0,
    "lang": 0.0,
    "strategies": 0.0,
}


@dataclass
class Candidate:
    """One retrieved passage, merged across its language variants.

    The corpus holds every passage in every indexed language. Left un-merged
    the variants of one passage would fill the final slots with the same
    content, so they are collapsed into one candidate: strategy votes are
    pooled, and the surfacing language is recorded for the cross-lingual trace.
    """

    group_id: str  # query_id:passage_index, identical across languages
    passage_id: str
    text: str
    lang: str
    query_id: int
    strategies: dict[str, int] = field(default_factory=dict)  # strategy -> rank
    raw_scores: dict[str, float] = field(default_factory=dict)
    rrf_score: float = 0.0
    fusion_rank: int = 0
    rerank_score: float = 0.0
    best_sentence: str = ""
    is_selected: int = 0
    variants: dict[str, str] = field(default_factory=dict)  # lang -> passage_id
    source_langs: list[str] = field(default_factory=list)  # langs that surfaced it
    matched_lang: str | None = None  # language whose wording scored best

    def to_dict(self) -> dict:
        return {
            "passage_id": self.passage_id,
            "group_id": self.group_id,
            "text": self.text,
            "lang": self.lang,
            "query_id": self.query_id,
            "strategies": self.strategies,
            "raw_scores": {k: round(v, 4) for k, v in self.raw_scores.items()},
            "rrf_score": round(self.rrf_score, 5),
            "fusion_rank": self.fusion_rank,
            "rerank_score": round(self.rerank_score, 4),
            "best_sentence": self.best_sentence,
            "available_langs": sorted(self.variants),
            "surfaced_by_langs": self.source_langs,
            "matched_lang": self.matched_lang,
        }


@dataclass
class RetrievalResult:
    query: str
    lang: str
    lang_confidence: float
    candidates: list[Candidate]
    timings_ms: dict[str, float]
    query_vector: np.ndarray
    cross_lingual: bool

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "lang": self.lang,
            "lang_confidence": round(self.lang_confidence, 3),
            "cross_lingual": self.cross_lingual,
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
            "candidates": [c.to_dict() for c in self.candidates],
        }


class Retriever:
    def __init__(self, store: Store | None = None) -> None:
        self.store = store or get_store()

    # ------------------------------------------------------------ helpers ---
    def _dense(self, level, qvec: np.ndarray, top_k: int) -> list[tuple[Unit, float]]:
        scores, idxs = level.faiss_index.search(qvec, top_k)
        out = []
        for score, row in zip(scores[0], idxs[0]):
            if row < 0:
                continue
            out.append((level.units[row], float(score)))
        return out

    def _cluster_candidates(
        self, qvec: np.ndarray, top_k: int, exclude_qid: int | None = None
    ) -> list[tuple[str, float]]:
        clusters = self.store.clusters
        sims = clusters.centroids @ qvec[0]
        top_clusters = np.argsort(-sims)[:2]
        scored: dict[str, float] = {}
        for cid in top_clusters:
            weight = float(sims[cid])
            for row in clusters.members.get(int(cid), []):
                qid = int(clusters.query_ids[row])
                # Offline evaluation replays queries that are themselves in the
                # cluster index; without this the strategy would just hand back
                # the query's own gold passage.
                if exclude_qid is not None and qid == exclude_qid:
                    continue
                for pid in self.store.gold_by_query.get(qid, []):
                    # Similarity to the individual member query, not just the
                    # centroid, so tight matches inside a loose cluster win.
                    member_sim = float(self.store.query_vectors[row] @ qvec[0])
                    scored[pid] = max(scored.get(pid, 0.0), 0.5 * weight + 0.5 * member_sim)
        ranked = sorted(scored.items(), key=lambda kv: -kv[1])[:top_k]
        return ranked

    def _keep(self, lang: str, query_lang: str, mode: LangMode) -> bool:
        if mode == "strict":
            return lang == query_lang
        if mode == "pivot":
            return lang != query_lang
        return True

    # -------------------------------------------------------------- search ---
    def retrieve(
        self,
        query: str,
        lang_mode: LangMode = "cross",
        top_k: int = FINAL_K,
        trace: TraceCb | None = None,
        query_lang: str | None = None,
        exclude_qid: int | None = None,
        qvec: np.ndarray | None = None,
        cross_encode: bool = False,
    ) -> RetrievalResult:
        emit = trace or (lambda *_: None)
        timings: dict[str, float] = {}
        t_all = time.perf_counter()

        t0 = time.perf_counter()
        if query_lang:
            lang, confidence = query_lang, 1.0
        else:
            lang, confidence = detect_lang(query)
        timings["detect"] = (time.perf_counter() - t0) * 1000
        emit("lang", {"lang": lang, "confidence": confidence, "ms": timings["detect"]})

        # The harness embeds the query once and shares the vector with the
        # guardrail, so the encoder is not paid for twice per turn.
        t0 = time.perf_counter()
        if qvec is None:
            qvec = self.store.encode_query(query)
        timings["embed"] = (time.perf_counter() - t0) * 1000
        emit("embed", {"ms": timings["embed"]})

        fetch = TOP_K_PER_STRATEGY * OVERFETCH
        ranked_lists: dict[str, list[str]] = {}
        raw: dict[str, dict[str, float]] = {}
        pool: dict[str, Candidate] = {}
        best_sentence: dict[str, tuple[float, str]] = {}

        def register(unit_lang: str, pid: str, text: str, qid: int, is_sel: int) -> str | None:
            """Fold a hit into its cross-language group; return the group id."""
            if not self._keep(unit_lang, lang, lang_mode):
                return None
            gid = ":".join(pid.split(":")[:2])
            cand = pool.get(gid)
            if cand is None:
                cand = Candidate(
                    group_id=gid,
                    passage_id=pid,
                    text=text,
                    lang=unit_lang,
                    query_id=qid,
                    is_selected=is_sel,
                )
                pool[gid] = cand
            cand.variants[unit_lang] = pid
            if unit_lang not in cand.source_langs:
                cand.source_langs.append(unit_lang)
            # Present the group in the asker's language when that variant was
            # retrieved; otherwise keep whichever language surfaced first.
            if unit_lang == lang and cand.lang != lang:
                cand.lang, cand.passage_id, cand.text = unit_lang, pid, text
            return gid

        # ---- dense strategies
        t0 = time.perf_counter()
        for level_name, strategy in (("passages", "dense_passage"), ("sentences", "dense_sentence")):
            if strategy == DROP_STRATEGY:
                ranked_lists[strategy] = []
                continue
            level = getattr(self.store, level_name)
            hits = self._dense(level, qvec, fetch)
            ordered: list[str] = []
            for unit, score in hits:
                parent = self.store.passages.by_uid.get(unit.passage_id)
                if parent is None:
                    continue
                gid = register(parent.lang, parent.uid, parent.text, parent.query_id, parent.is_selected)
                if gid is None:
                    continue
                if strategy == "dense_sentence":
                    prev = best_sentence.get(gid)
                    if prev is None or score > prev[0]:
                        best_sentence[gid] = (score, unit.text)
                if gid not in ordered:
                    ordered.append(gid)
                    raw.setdefault(strategy, {})[gid] = score
                if len(ordered) >= TOP_K_PER_STRATEGY:
                    break
            ranked_lists[strategy] = ordered
            emit("strategy", {"name": strategy, "hits": len(ordered)})
        timings["dense"] = (time.perf_counter() - t0) * 1000

        # ---- sparse strategies
        t0 = time.perf_counter()
        for level_name, strategy in (("passages", "bm25_passage"), ("sentences", "bm25_sentence")):
            if strategy == DROP_STRATEGY:
                ranked_lists[strategy] = []
                continue
            level = getattr(self.store, level_name)
            ordered = []
            for uid, score in level.bm25.search(query, fetch):
                unit = level.by_uid[uid]
                parent = self.store.passages.by_uid.get(unit.passage_id)
                if parent is None:
                    continue
                gid = register(parent.lang, parent.uid, parent.text, parent.query_id, parent.is_selected)
                if gid is None:
                    continue
                if gid not in ordered:
                    ordered.append(gid)
                    raw.setdefault(strategy, {})[gid] = score
                if len(ordered) >= TOP_K_PER_STRATEGY:
                    break
            ranked_lists[strategy] = ordered
            emit("strategy", {"name": strategy, "hits": len(ordered)})
        timings["bm25"] = (time.perf_counter() - t0) * 1000

        # ---- query-cluster strategy
        t0 = time.perf_counter()
        ordered = []
        for pid, score in ([] if DROP_STRATEGY == "cluster" else self._cluster_candidates(qvec, fetch, exclude_qid)):
            parent = self.store.passages.by_uid.get(pid)
            if parent is None:
                continue
            gid = register(parent.lang, parent.uid, parent.text, parent.query_id, parent.is_selected)
            if gid is None or gid in ordered:
                continue
            ordered.append(gid)
            raw.setdefault("cluster", {})[gid] = score
            if len(ordered) >= TOP_K_PER_STRATEGY:
                break
        ranked_lists["cluster"] = ordered
        timings["cluster"] = (time.perf_counter() - t0) * 1000
        emit("strategy", {"name": "cluster", "hits": len(ordered)})

        # ---- reciprocal rank fusion
        t0 = time.perf_counter()
        for strategy, ordered in ranked_lists.items():
            for rank, gid in enumerate(ordered, start=1):
                cand = pool[gid]
                cand.strategies[strategy] = rank
                cand.rrf_score += 1.0 / (RRF_K + rank)
                cand.raw_scores[strategy] = raw.get(strategy, {}).get(gid, 0.0)
        fused = sorted(pool.values(), key=lambda c: -c.rrf_score)
        for rank, cand in enumerate(fused, start=1):
            cand.fusion_rank = rank
            if cand.group_id in best_sentence:
                cand.best_sentence = best_sentence[cand.group_id][1]
        timings["fusion"] = (time.perf_counter() - t0) * 1000
        emit("fusion", {"candidates": len(fused), "ms": timings["fusion"]})

        # ---- rerank
        t0 = time.perf_counter()
        shortlist = fused[:RERANK_CANDIDATES]
        self._rerank(query, qvec, shortlist, lang)
        shortlist.sort(key=lambda c: -c.rerank_score)
        timings["rerank"] = (time.perf_counter() - t0) * 1000
        emit("rerank", {"candidates": len(shortlist), "ms": timings["rerank"]})

        # ---- optional cross-encoder pass ("precision mode")
        if cross_encode and shortlist:
            t0 = time.perf_counter()
            shortlist = self._cross_encode(query, shortlist)
            timings["cross_encoder"] = (time.perf_counter() - t0) * 1000
            emit("cross_encoder", {
                "candidates": min(len(shortlist), CROSS_ENCODER_DEPTH),
                "ms": timings["cross_encoder"],
            })

        final = shortlist[:top_k]
        # Always re-derive the span from the variant actually being shown. A
        # sentence hit may have arrived from a different language's copy of the
        # same passage, and quoting that would answer in the wrong script.
        for cand in final:
            cand.best_sentence = self._best_sentence(cand, qvec)

        timings["total"] = (time.perf_counter() - t_all) * 1000
        cross = any(c.lang != lang for c in final)
        return RetrievalResult(
            query=query,
            lang=lang,
            lang_confidence=confidence,
            candidates=final,
            timings_ms=timings,
            query_vector=qvec,
            cross_lingual=cross,
        )

    # -------------------------------------------------------------- rerank ---
    def _rerank(self, query: str, qvec: np.ndarray, cands: list[Candidate], query_lang: str) -> None:
        """Feature re-scorer over the fused shortlist.

        Deliberately vector-arithmetic only. A cross-encoder costs 200 ms+ on
        CPU for 20 candidates, which would consume the entire Tier 1 budget on
        its own; ``scripts/eval_rerank.py`` measures what that trade costs in
        quality.
        """
        if not cands:
            return
        q_tokens = set(tokenize(query))

        # Score each group by its best-matching language variant, not by the
        # one being displayed. Translations of the same passage do not match a
        # query equally well: asked "कॉर्पोरेशन क्या है", the Hindi text scores
        # 0.861 because it renders the word as निगम, while the Marathi text of
        # the same passage scores 0.923 by keeping the transliteration. Ranking
        # by the displayed variant buried the correct passage at rank four.
        # The asker still reads their own language; only the ranking changes.
        dense = np.empty(len(cands), dtype=np.float32)
        for i, cand in enumerate(cands):
            rows = [
                self.store.passages.by_uid[pid].row
                for pid in cand.variants.values()
                if pid in self.store.passages.by_uid
            ]
            if not rows:
                dense[i] = 0.0
                continue
            scores = self.store.passages.vectors[rows] @ qvec[0]
            best = int(np.argmax(scores))
            dense[i] = float(scores[best])
            cand.raw_scores["dense_best_lang"] = float(scores[best])
            best_pid = list(cand.variants.values())[best]
            best_unit = self.store.passages.by_uid.get(best_pid)
            if best_unit is not None and best_unit.lang != cand.lang:
                # Worth surfacing in the trace: the group won on another
                # language's wording.
                cand.matched_lang = best_unit.lang

        bm25_vals = np.array(
            [max(c.raw_scores.get("bm25_passage", 0.0), c.raw_scores.get("bm25_sentence", 0.0)) for c in cands],
            dtype=np.float32,
        )
        bm25_max = float(bm25_vals.max()) or 1.0
        rrf_vals = np.array([c.rrf_score for c in cands], dtype=np.float32)
        rrf_max = float(rrf_vals.max()) or 1.0

        for i, cand in enumerate(cands):
            tokens = set(tokenize(cand.text))
            overlap = len(q_tokens & tokens) / (len(q_tokens) or 1)
            score = (
                RERANK_WEIGHTS["dense"] * float(dense[i])
                + RERANK_WEIGHTS["bm25"] * float(bm25_vals[i]) / bm25_max
                + RERANK_WEIGHTS["rrf"] * float(rrf_vals[i]) / rrf_max
                + RERANK_WEIGHTS["overlap"] * overlap
                + RERANK_WEIGHTS["lang"] * (1.0 if cand.lang == query_lang else 0.0)
                + RERANK_WEIGHTS["strategies"] * (len(cand.strategies) / 5.0)
            )
            cand.rerank_score = float(score)
            cand.raw_scores["dense_final"] = float(dense[i])
            cand.raw_scores["overlap"] = float(overlap)

    def _cross_encode(self, query: str, cands: list[Candidate]) -> list[Candidate]:
        """Re-score the shortlist head with a joint query-passage model.

        Only the head is scored: cost is linear in depth, and the measurement
        showed the gain flattening after 10 while the latency kept climbing.
        """
        head = cands[:CROSS_ENCODER_DEPTH]
        tail = cands[CROSS_ENCODER_DEPTH:]
        pairs = [(query, c.text[:CROSS_ENCODER_MAX_CHARS]) for c in head]
        scores = self.store.cross_encoder.predict(
            pairs, batch_size=len(pairs), show_progress_bar=False
        )
        for cand, score in zip(head, np.asarray(scores, dtype=np.float32)):
            cand.raw_scores["cross_encoder"] = float(score)
            cand.rerank_score = float(score)
        head.sort(key=lambda c: -c.rerank_score)
        return head + tail

    def _best_sentence(self, cand: Candidate, qvec: np.ndarray) -> str:
        rows = self.store.sentences.rows_by_passage.get(cand.passage_id)
        if not rows:
            return cand.text
        sims = self.store.sentences.vectors[rows] @ qvec[0]
        return self.store.sentences.units[rows[int(np.argmax(sims))]].text
