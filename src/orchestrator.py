"""RAGOrchestrator — single entry point wiring all components."""
from __future__ import annotations

import time
from pathlib import Path

from config.settings import get_settings
from src.ingestion.pdf_pipeline import PDFIngestionPipeline
from src.ingestion.image_pipeline import ImageIngestionPipeline
from src.ingestion.csv_pipeline import CSVIngestionPipeline
from src.embeddings.embedding_manager import EmbeddingManager
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.rag_chain import RAGChain, RAGResponse
from src.evaluation.metrics import EvaluationDashboard

cfg = get_settings()

SUPPORTED = {
    ".pdf": "pdf", ".png": "image", ".jpg": "image",
    ".jpeg": "image", ".webp": "image", ".csv": "csv", ".tsv": "csv",
}


class RAGOrchestrator:
    def __init__(self):
        self._embed_mgr = EmbeddingManager()
        self._rag_chain = RAGChain()
        self._dashboard = EvaluationDashboard()
        self._corpus: list[str] = []
        self._corpus_meta: list[dict] = []
        self._retriever = None
        self._pdf = PDFIngestionPipeline()
        self._img = ImageIngestionPipeline()
        self._csv = CSVIngestionPipeline()

    def ingest_file(self, data: bytes, filename: str) -> dict:
        ext = Path(filename).suffix.lower()
        ftype = SUPPORTED.get(ext)
        if not ftype:
            return {"success": False, "error": f"Unsupported: {ext}"}
        t0 = time.time()
        try:
            if ftype == "pdf":
                docs = [c.to_langchain_doc() for c in self._pdf.ingest_bytes(data, filename)]
                n = self._embed_mgr.embed_and_store(docs, "text")
            elif ftype == "image":
                docs = [self._img.ingest_bytes(data, filename).to_langchain_doc()]
                n = self._embed_mgr.embed_and_store(docs, "image")
            else:
                docs = [c.to_langchain_doc() for c in self._csv.ingest_bytes(data, filename)]
                n = self._embed_mgr.embed_and_store(docs, "table")

            for d in docs:
                self._corpus.append(d.page_content)
                self._corpus_meta.append(d.metadata)
            self._retriever = HybridRetriever(
                self._corpus, self._corpus_meta, self._embed_mgr
            )
            return {
                "success": True, "filename": filename, "type": ftype,
                "vectors_stored": n, "elapsed_ms": round((time.time() - t0) * 1000),
            }
        except Exception as e:
            return {"success": False, "error": str(e), "filename": filename}

    def query(self, question: str, doc_type_filter=None, year_filter=None, alpha=None):
        if not self._retriever:
            return RAGResponse("Upload documents first.", [], 0.0, 0), None
        pf = {}
        if doc_type_filter:
            pf["doc_type"] = {"$eq": doc_type_filter}
        if year_filter:
            pf["year"] = {"$eq": year_filter}
        if alpha is not None:
            self._retriever._alpha = alpha
        ns_map = {"pdf": "text", "image": "image", "csv": "table"}
        ns = ns_map.get(doc_type_filter or "")
        t0 = time.time()
        chunks = self._retriever.retrieve(question, namespace=ns, filter=pf or None)
        lr = (time.time() - t0) * 1000
        t1 = time.time()
        response = self._rag_chain.answer(question, chunks)
        lg = (time.time() - t1) * 1000
        metrics = self._dashboard.evaluate(
            question, chunks, response.sources, response.answer,
            response.context_tokens,
            {"retrieval_ms": round(lr), "generation_ms": round(lg)},
        )
        self._dashboard.add(metrics)
        return response, metrics

    def eval_summary(self): return self._dashboard.summary()
    def eval_history(self): return self._dashboard.history()
    def index_stats(self): return self._embed_mgr.get_index_stats()