"""Generate embeddings using an OpenAI-compatible API."""

from __future__ import annotations

import time

from openai import OpenAI

from src.config import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
)


class OpenAIEmbedder:
    """Create text embeddings using an OpenAI-compatible client.

    Defaults to Google's Gemini embedding model via its OpenAI-compatible
    endpoint (see ``src/config.py``). Whatever model builds the index must
    also be used to embed queries at search time, so this should not be
    changed per-developer the way the chat model can be.
    """

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        batch_size: int = 64,
        max_retries: int = 3,
    ) -> None:
        if not EMBEDDING_API_KEY:
            raise ValueError(
                "EMBEDDING_API_KEY is not configured. "
                "Set it in the local .env file."
            )

        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries

        self.client = OpenAI(
            api_key=EMBEDDING_API_KEY,
            base_url=EMBEDDING_BASE_URL,
        )

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed one batch with simple retry handling."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                )

                ordered_data = sorted(
                    response.data,
                    key=lambda item: item.index,
                )

                return [item.embedding for item in ordered_data]

            except Exception as error:
                last_error = error

                if attempt == self.max_retries - 1:
                    raise

                time.sleep(2**attempt)

        raise RuntimeError("Embedding request failed") from last_error

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches."""
        if not texts:
            return []

        embeddings: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            embeddings.extend(self._embed_batch(batch))

        return embeddings
