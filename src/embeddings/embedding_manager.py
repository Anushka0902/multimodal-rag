"""Embedding manager — batched local SentenceTransformer embeds + Pinecone upsert."""
from __future__ import annotations

import time
from typing import Literal

from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

from config.settings import get_settings

cfg = get_settings()

Namespace = Literal["text", "image", "table"]
BATCH_SIZE = 100   # Pinecone upsert batch limit
EMBED_BATCH = 50   # embed batch size


class EmbeddingManager:
    """
    Manages the full embed→store cycle.

    Changes from paid version
    -------------------------
    - OpenAIEmbeddings replaced with local SentenceTransformer (BAAI/bge-base-en-v1.5)
    - Dimension: 3072 → 768  (update pinecone_dimension in config)
    - No API key or cost for embeddings
    - BGE models expect a query prefix for retrieval — applied automatically
    """

    def __init__(self) -> None:
        # Loads model weights from HuggingFace on first run (~440 MB), cached after
        self._model = SentenceTransformer(cfg.embed_model)
        pc = Pinecone(api_key=cfg.pinecone_api_key)
        self._index = self._get_or_create_index(pc)

    # ── Public API ─────────────────────────────────────────────────────

    def embed_and_store(self, docs: list[Document], namespace: Namespace) -> int:
        """Embed *docs* and upsert into *namespace*. Returns vector count."""
        vectors = []
        for i in range(0, len(docs), EMBED_BATCH):
            batch = docs[i : i + EMBED_BATCH]
            texts = [d.page_content for d in batch]
            # BGE passage prefix for document embedding
            prefixed = [f"passage: {t}" for t in texts]
            embeddings = self._model.encode(
                prefixed, normalize_embeddings=True, show_progress_bar=False
            ).tolist()

            for j, (doc, emb) in enumerate(zip(batch, embeddings)):
                vec_id = f"{namespace}-{doc.metadata['doc_hash']}-{i + j}"
                vectors.append(
                    {
                        "id": vec_id,
                        "values": emb,
                        "metadata": {
                            "text": doc.page_content[:1000],
                            **doc.metadata,
                        },
                    }
                )

        for i in range(0, len(vectors), BATCH_SIZE):
            self._index.upsert(
                vectors=vectors[i : i + BATCH_SIZE],
                namespace=namespace,
            )
        return len(vectors)

    def embed_query(self, query: str) -> list[float]:
        # BGE query prefix for retrieval
        prefixed = f"query: {query}"
        return self._model.encode(
            prefixed, normalize_embeddings=True
        ).tolist()

    def semantic_search(
        self,
        query_vec: list[float],
        namespace: Namespace | None,
        top_k: int,
        filter: dict | None = None,
    ) -> list[dict]:
        """Raw Pinecone query — returns list of match dicts."""
        kwargs: dict = {"vector": query_vec, "top_k": top_k, "include_metadata": True}
        if namespace:
            kwargs["namespace"] = namespace
        if filter:
            kwargs["filter"] = filter
        response = self._index.query(**kwargs)
        return response.matches

    def get_index_stats(self) -> dict:
        return self._index.describe_index_stats()

    # ── Private ────────────────────────────────────────────────────────

    def _get_or_create_index(self, pc: Pinecone):
        existing = [i.name for i in pc.list_indexes()]
        if cfg.pinecone_index_name not in existing:
            pc.create_index(
                name=cfg.pinecone_index_name,
                dimension=cfg.pinecone_dimension,  # 768 for bge-base
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=cfg.pinecone_env),
            )
            for _ in range(20):
                status = pc.describe_index(cfg.pinecone_index_name).status
                if status.get("ready"):
                    break
                time.sleep(3)
        return pc.Index(cfg.pinecone_index_name)