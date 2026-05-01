"""RAG chain — assembles retrieved context and calls Gemini for answers."""
from __future__ import annotations

from dataclasses import dataclass

import google.generativeai as genai

from config.settings import get_settings
from src.retrieval.hybrid_retriever import RetrievedChunk

cfg = get_settings()
genai.configure(api_key=cfg.gemini_api_key)

SYSTEM_PROMPT = """\
You are a knowledgeable, helpful assistant with access to retrieved document chunks.
Your goal is to give the most useful, complete answer possible.

Rules:
1. Primarily use the provided context chunks to answer.
2. If the context partially covers the question, answer what you can from it and
   supplement with your own knowledge — clearly labelling which parts come from
   the documents vs your general knowledge.
3. After each claim drawn from the documents, cite the source using
   [Source: filename, page N] or [Source: filename].
4. Only say "I don't have sufficient information" if the question is completely
   unrelated to anything in the documents AND you have no general knowledge on it.
5. Be thorough but concise. Prefer prose over bullet points unless listing items.
6. If multiple sources agree on a point, note that explicitly.
7. If sources contradict each other, flag the contradiction and present both views.
"""

HUMAN_TEMPLATE = """\
<context>
{context}
</context>

<question>
{question}
</question>

Instructions:
- Answer as completely as possible using the context above.
- If the context is relevant but incomplete, combine it with your general knowledge
  and clearly indicate which parts come from the documents vs general knowledge.
- Cite document sources inline using [Source: filename, page N].
- If the question is entirely outside the scope of the documents, answer from
  general knowledge and note that no relevant documents were found.
"""


@dataclass
class RAGResponse:
    answer: str
    sources: list[RetrievedChunk]
    confidence: float
    context_tokens: int


class RAGChain:
    """
    Assembles retrieved chunks into a prompt and generates a grounded answer.

    Changes from paid version
    -------------------------
    - LangChain ChatOpenAI replaced with google-generativeai GenerativeModel
    - Improved prompt — answers from general knowledge when docs are partial
    - Same RAGResponse dataclass — no downstream changes needed
    """

    def __init__(self) -> None:
        self._model = genai.GenerativeModel(
            model_name=cfg.chat_model,
            system_instruction=SYSTEM_PROMPT,
        )

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> RAGResponse:
        if not chunks:
            # No documents uploaded or no matches — answer from general knowledge
            response = self._model.generate_content(
                f"No documents are available.\n"
                f"Answer from general knowledge.\n\n"
                f"Question: {question}"
            )
            return RAGResponse(
                answer=response.text,
                sources=[],
                confidence=0.0,
                context_tokens=0,
            )

        context_str, used_chunks = self._build_context(chunks)
        prompt = HUMAN_TEMPLATE.format(context=context_str, question=question)

        response = self._model.generate_content(prompt)
        answer_text = response.text

        confidence = (
            sum(c.score for c in used_chunks) / len(used_chunks)
            if used_chunks else 0.0
        )
        context_tokens = len(context_str.split()) * 4 // 3

        return RAGResponse(
            answer=answer_text,
            sources=used_chunks,
            confidence=confidence,
            context_tokens=context_tokens,
        )

    # ── Private ────────────────────────────────────────────────────────

    @staticmethod
    def _build_context(
        chunks: list[RetrievedChunk], max_chars: int = 12_000
    ) -> tuple[str, list[RetrievedChunk]]:
        """
        Build the context string, truncating if needed to stay under
        max_chars. Returns (context_string, chunks_actually_used).
        """
        parts: list[str] = []
        used: list[RetrievedChunk] = []
        total = 0

        for chunk in chunks:
            header = (
                f"[Source: {chunk.source}"
                + (f", page {chunk.page}" if chunk.page else "")
                + f" | type: {chunk.doc_type}"
                + f" | relevance: {chunk.score:.2f}]"
            )
            block = f"{header}\n{chunk.content}\n"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            used.append(chunk)
            total += len(block)

        return "\n---\n".join(parts), used