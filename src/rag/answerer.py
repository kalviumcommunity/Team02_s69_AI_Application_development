"""Generate cited answers from retrieved SchemeLens AI context.

The main public function is ``answer_question(query)``. It returns a
dictionary with the following keys:

- ``answer``: The generated answer containing chunk citations such as [1].
- ``citations``: Citation metadata for the chunks referenced by the answer.
- ``retrieved_chunks``: The top retrieved chunks used as context.
- ``top_score``: The relevance score of the highest-ranked chunk.
"""

from __future__ import annotations

import re
from typing import Any

from openai import OpenAI

from src.config import CHAT_MODEL, LLM_API_KEY, LLM_BASE_URL
from src.indexing.embedder import OpenAIEmbedder
from src.indexing.vector_store import VectorStore
from src.rag.prompt import SYSTEM_PROMPT, build_user_prompt


CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class QuestionAnswerer:
    """Generate answers using only retrieved scheme document chunks."""

    def __init__(
        self,
        vector_store: Any,
        llm_client: Any,
        model: str = CHAT_MODEL,
    ) -> None:
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.model = model

    def answer_question(self, query: str) -> dict:
        """Retrieve context, generate an answer, and build citations."""
        retrieved_chunks = self.vector_store.search(
            query,
            top_k=5,
        )

        if not retrieved_chunks:
            return {
                "answer": (
                    "The provided documents do not contain enough "
                    "information to answer this question."
                ),
                "citations": [],
                "retrieved_chunks": [],
                "top_score": None,
            }

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": build_user_prompt(
                        query,
                        retrieved_chunks,
                    ),
                },
            ],
            temperature=0,
        )

        message = response.choices[0].message

        answer = message.content

        if not answer:
            answer = getattr(message, "reasoning", None)

        if not answer:
            raise ValueError(
                "The LLM returned an empty response."
            )

        answer = answer.strip()

        citations = self._build_citations(
            answer,
            retrieved_chunks,
        )

        return {
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
            "top_score": retrieved_chunks[0]["score"],
        }

    @staticmethod
    def _build_citations(
        answer: str,
        retrieved_chunks: list[dict],
    ) -> list[dict]:
        """Map [n] references in the answer to real chunk metadata."""
        references = CITATION_PATTERN.findall(answer)

        citations: list[dict] = []
        seen: set[int] = set()

        for reference in references:
            chunk_number = int(reference)

            if chunk_number in seen:
                continue

            if not 1 <= chunk_number <= len(retrieved_chunks):
                continue

            seen.add(chunk_number)

            chunk = retrieved_chunks[chunk_number - 1]
            metadata = chunk["metadata"]

            citations.append(
                {
                    "scheme_name": metadata["scheme_name"],
                    "page": metadata["page_number"],
                    "section": metadata["section_title"],
                    "snippet": chunk["chunk"],
                }
            )

        return citations


def answer_question(query: str) -> dict:
    """Answer a question using the persistent SchemeLens AI index."""
    if not LLM_API_KEY:
        raise ValueError(
            "LLM_API_KEY is not configured. "
            "Set it in the local .env file."
        )

    embedder = OpenAIEmbedder()

    vector_store = VectorStore(
        embedder=embedder,
    )

    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )

    answerer = QuestionAnswerer(
        vector_store=vector_store,
        llm_client=client,
    )

    return answerer.answer_question(query)
