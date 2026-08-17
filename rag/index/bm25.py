"""Sparse BM25 index backed by a scipy CSR matrix.

Pure-Python BM25 implementations score documents in a loop, which costs tens of
milliseconds on a 50k-document corpus and eats most of the Tier 1 budget. Here
the BM25 term weights are precomputed once at build time, so a query is a
single sparse mat-vec: typically well under 5 ms for this corpus.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from .text import tokenize

K1 = 1.5
B = 0.75


@dataclass
class BM25Index:
    matrix: sparse.csr_matrix  # documents x vocabulary, BM25 term weights
    vocab: dict[str, int]
    idf: np.ndarray
    doc_ids: list[str]

    @classmethod
    def build(cls, doc_ids: list[str], texts: list[str]) -> "BM25Index":
        vocab: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        freqs: list[float] = []
        lengths = np.zeros(len(texts), dtype=np.float32)

        for row, text in enumerate(texts):
            counts: dict[int, int] = {}
            tokens = tokenize(text)
            lengths[row] = len(tokens)
            for token in tokens:
                idx = vocab.get(token)
                if idx is None:
                    idx = len(vocab)
                    vocab[token] = idx
                counts[idx] = counts.get(idx, 0) + 1
            for idx, count in counts.items():
                rows.append(row)
                cols.append(idx)
                freqs.append(count)

        n_docs, n_terms = len(texts), len(vocab)
        tf = sparse.csr_matrix(
            (np.asarray(freqs, dtype=np.float32), (rows, cols)),
            shape=(n_docs, n_terms),
        )

        doc_freq = np.asarray((tf > 0).sum(axis=0)).ravel().astype(np.float32)
        idf = np.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5)).astype(np.float32)

        avgdl = float(lengths.mean()) or 1.0
        # BM25 saturation applied per stored value, so query time is a mat-vec.
        norm = (K1 * (1.0 - B + B * lengths / avgdl)).astype(np.float32)
        weighted = tf.tocoo()
        denom = weighted.data + norm[weighted.row]
        weighted.data = (weighted.data * (K1 + 1.0)) / denom
        matrix = weighted.tocsr()

        return cls(matrix=matrix, vocab=vocab, idf=idf, doc_ids=doc_ids)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        idxs = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        if not idxs:
            return []
        weights = np.zeros(self.matrix.shape[1], dtype=np.float32)
        for i in idxs:
            weights[i] += self.idf[i]
        scores = self.matrix @ weights
        k = min(top_k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.doc_ids[i], float(scores[i])) for i in top if scores[i] > 0.0]

    def save(self, path: Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "matrix": self.matrix,
                    "vocab": self.vocab,
                    "idf": self.idf,
                    "doc_ids": self.doc_ids,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        return cls(**payload)
