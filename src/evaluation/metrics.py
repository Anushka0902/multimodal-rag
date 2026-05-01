"""
Retrieval evaluation module.

Computes standard RAG metrics without ground-truth labels (self-contained):
  - Mean Reciprocal Rank (MRR) — relevance of first good hit
  - Precision@K — fraction of top-K results that are relevant
  - Context utilisation — how much retrieved context the LLM actually used
  - Answer faithfulness — simple keyword overlap proxy
  - Latency breakdown — per-stage timing
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Callable

from src.retrieval.hybrid_retriever import RetrievedChunk


@dataclass
class RetrievalMetrics:
    query: str
    num_chunks_retrieved: int
    num_chunks_used: int
    mean_relevance_score: float
    precision_at_k: float          # @5
    mrr: float
    context_tokens: int
    answer_length: int
    faithfulness_score: float      # overlap proxy
    latency_ms: dict[str, float]   # stage → ms
    timestamp: float = field(default_factory=time.time)


class EvaluationDashboard:
    """
    Records and aggregates retrieval metrics across queries.

    Usage
    -----
        dashboard = EvaluationDashboard()
        metrics = dashboard.evaluate(query, chunks, answer, latencies)
        dashboard.add(metrics)
        summary = dashboard.summary()
    """

    def __init__(self) -> None:
        self._history: list[RetrievalMetrics] = []

    def evaluate(
        self,
        query: str,
        retrieved: list[RetrievedChunk],
        used: list[RetrievedChunk],
        answer: str,
        context_tokens: int,
        latency_ms: dict[str, float],
        relevance_threshold: float = 0.5,
    ) -> RetrievalMetrics:
        scores = [c.score for c in retrieved]
        relevant_mask = [s >= relevance_threshold for s in scores]

        # MRR: 1/rank of first relevant result
        mrr = 0.0
        for rank, is_rel in enumerate(relevant_mask[:10], 1):
            if is_rel:
                mrr = 1.0 / rank
                break

        # Precision@5
        p_at_5 = sum(relevant_mask[:5]) / min(5, len(relevant_mask)) if relevant_mask else 0

        # Faithfulness: fraction of answer sentences that contain retrieved words
        faithfulness = self._faithfulness(answer, retrieved)

        return RetrievalMetrics(
            query=query,
            num_chunks_retrieved=len(retrieved),
            num_chunks_used=len(used),
            mean_relevance_score=mean(scores) if scores else 0.0,
            precision_at_k=p_at_5,
            mrr=mrr,
            context_tokens=context_tokens,
            answer_length=len(answer.split()),
            faithfulness_score=faithfulness,
            latency_ms=latency_ms,
        )

    def add(self, m: RetrievalMetrics) -> None:
        self._history.append(m)

    def summary(self) -> dict:
        if not self._history:
            return {}
        return {
            "total_queries": len(self._history),
            "avg_mrr": round(mean(m.mrr for m in self._history), 3),
            "avg_precision_at_5": round(
                mean(m.precision_at_k for m in self._history), 3
            ),
            "avg_relevance": round(
                mean(m.mean_relevance_score for m in self._history), 3
            ),
            "avg_faithfulness": round(
                mean(m.faithfulness_score for m in self._history), 3
            ),
            "avg_context_tokens": round(
                mean(m.context_tokens for m in self._history)
            ),
            "avg_total_latency_ms": round(
                mean(sum(m.latency_ms.values()) for m in self._history)
            ),
        }

    def history(self) -> list[RetrievalMetrics]:
        return list(self._history)

    # ── Private ────────────────────────────────────────────────────────

    @staticmethod
    def _faithfulness(answer: str, chunks: list[RetrievedChunk]) -> float:
        """
        Proxy: fraction of answer sentences that share ≥3 unique content
        words with at least one retrieved chunk.
        Proper faithfulness needs an LLM judge; this is a cheap proxy.
        """
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on",
            "at", "to", "for", "of", "and", "or", "it", "this", "that",
        }
        corpus_words = set()
        for chunk in chunks:
            corpus_words |= {
                w.lower() for w in chunk.content.split() if w.lower() not in stopwords
            }

        sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
        if not sentences:
            return 0.0

        faithful = 0
        for sent in sentences:
            words = {w.lower() for w in sent.split() if w.lower() not in stopwords}
            if len(words & corpus_words) >= 3:
                faithful += 1

        return faithful / len(sentences)