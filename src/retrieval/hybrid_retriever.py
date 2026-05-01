"""
Hybrid retrieval pipeline.

Flow
----
Query → [BM25 sparse] ──┐
                         ├─ RRF fusion → top-20 → CrossEncoder rerank → top-5
Query → [Pinecone dense] ┘
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config.settings import get_settings
from src.embeddings.embedding_manager import EmbeddingManager, Namespace

cfg = get_settings()

# Loaded once at module level — model is ~85 MB, cached after first download
_reranker = CrossEncoder(cfg.rerank_model)


@dataclass
class RetrievedChunk:
    content: str
    source: str
    doc_type: str
    page: int | None
    score: float          # CrossEncoder relevance score (0–1 after sigmoid)
    metadata: dict


class HybridRetriever:
    """
    Combines BM25 keyword search with Pinecone semantic search via
    Reciprocal Rank Fusion, then reranks with a local CrossEncoder.

    Changes from paid version
    -------------------------
    - Cohere API reranker replaced with cross-encoder/ms-marco-MiniLM-L6-v2
    - CrossEncoder runs locally — no API key, no per-call cost
    - Scores are normalised to 0–1 via sigmoid for consistency
    """

    def __init__(
        self,
        corpus: list[str],
        corpus_meta: list[dict],
        embed_manager: EmbeddingManager,
        alpha: float = cfg.hybrid_alpha,
    ) -> None:
        self._corpus = corpus
        self._corpus_meta = corpus_meta
        self._embed_mgr = embed_manager
        self._alpha = alpha
        self._bm25 = BM25Okapi([doc.split() for doc in corpus]) if corpus else None

    # ── Public API ─────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = cfg.top_k_retrieval,
        namespace: Namespace | None = None,
        filter: dict | None = None,
    ) -> list[RetrievedChunk]:
        if not self._corpus:
            return []

        bm25_ranks = self._bm25_search(query, top_k)
        query_vec = self._embed_mgr.embed_query(query)
        semantic_ranks = self._semantic_search(query_vec, top_k, namespace, filter)
        fused = self._rrf_fusion(bm25_ranks, semantic_ranks, top_k)
        return self._rerank(query, fused, cfg.top_k_rerank)

    # ── Private ────────────────────────────────────────────────────────

    def _bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(query.split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(idx, float(scores[idx])) for idx in top_indices[:k]]

    def _semantic_search(
        self,
        query_vec: list[float],
        k: int,
        namespace: Namespace | None,
        filter: dict | None,
    ) -> list[tuple[str, float]]:
        matches = self._embed_mgr.semantic_search(query_vec, namespace, k, filter)
        return [(m["id"], m["score"]) for m in matches]

    def _rrf_fusion(
        self,
        bm25_ranks: list[tuple[int, float]],
        semantic_ranks: list[tuple[str, float]],
        top_k: int,
        k: int = 60,
    ) -> list[str]:
        scores: dict[str, float] = {}
        for rank, (idx, _) in enumerate(bm25_ranks):
            doc_id = f"bm25-{idx}"
            scores[doc_id] = scores.get(doc_id, 0) + (1 - self._alpha) / (rank + k)
        for rank, (vec_id, _) in enumerate(semantic_ranks):
            scores[vec_id] = scores.get(vec_id, 0) + self._alpha / (rank + k)
        return sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]

    def _rerank(self, query: str, doc_ids: list[str], top_k: int) -> list[RetrievedChunk]:
        """Local CrossEncoder reranking — replaces Cohere API call."""
        docs_to_rerank: list[tuple[str, dict]] = []

        for doc_id in doc_ids:
            if doc_id.startswith("bm25-"):
                idx = int(doc_id.split("-")[1])
                if idx < len(self._corpus):
                    docs_to_rerank.append((self._corpus[idx], self._corpus_meta[idx]))
            else:
                # semantic result — text stored in Pinecone metadata
                matches = self._embed_mgr.semantic_search(
                    self._embed_mgr.embed_query(query), None, 1
                )
                for m in matches:
                    if m["id"] == doc_id:
                        meta = m.get("metadata", {})
                        docs_to_rerank.append((meta.get("text", ""), meta))

        if not docs_to_rerank:
            return []

        texts = [t for t, _ in docs_to_rerank]
        # CrossEncoder returns raw logit scores; apply sigmoid → 0–1 range
        raw_scores = _reranker.predict([(query, doc) for doc in texts])
        import math
        norm_scores = [1 / (1 + math.exp(-float(s))) for s in raw_scores]

        ranked = sorted(
            zip(norm_scores, docs_to_rerank),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        return [
            RetrievedChunk(
                content=text,
                source=meta.get("source", "unknown"),
                doc_type=meta.get("doc_type", "unknown"),
                page=meta.get("page"),
                score=score,
                metadata=meta,
            )
            for score, (text, meta) in ranked
        ]