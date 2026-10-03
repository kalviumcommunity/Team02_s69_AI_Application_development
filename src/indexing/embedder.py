"""Generate embeddings using an OpenAI-compatible API."""

from __future__ import annotations

import time

from openai import OpenAI, RateLimitError

from src.config import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
)


def _is_daily_limit(error: Exception) -> bool:
    """True when a rate-limit error is a per-day quota, not a per-minute one."""
    text = str(error).lower()
    return any(marker in text for marker in ("per-day", "perday", "per_day"))


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
        max_retries: int = 10,
        min_interval_seconds: float = 40.0,
    ) -> None:
        if not EMBEDDING_API_KEY:
            raise ValueError(
                "EMBEDDING_API_KEY is not configured. "
                "Set it in the local .env file."
            )

        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        # Gemini's free tier allows 100 embedding requests per minute and
        # counts each text in a batch. Keeping at least this much time
        # between batch requests stays under that limit (64 texts / 40s).
        self.min_interval_seconds = min_interval_seconds
        self._last_request_at: float | None = None

        self.client = OpenAI(
            api_key=EMBEDDING_API_KEY,
            base_url=EMBEDDING_BASE_URL,
        )

    def _wait_for_slot(self) -> None:
        """Sleep just long enough to respect the minimum gap between batches."""
        if self._last_request_at is None:
            return

        elapsed = time.monotonic() - self._last_request_at
        remaining = self.min_interval_seconds - elapsed

        if remaining > 0:
            time.sleep(remaining)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed one batch, waiting out rate limits instead of failing fast."""
        last_error = None

        for attempt in range(self.max_retries):
            self._wait_for_slot()

            try:
                self._last_request_at = time.monotonic()
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                )

                # Sort by the API's index when every item carries one. Not
                # every OpenAI-compatible provider does (Gemini returns
                # index=None), so otherwise trust the response order.
                indices = [getattr(item, "index", None) for item in response.data]

                if all(index is not None for index in indices):
                    ordered_data = sorted(response.data, key=lambda item: item.index)
                else:
                    ordered_data = list(response.data)

                return [item.embedding for item in ordered_data]

            except Exception as error:
                last_error = error

                if attempt == self.max_retries - 1:
                    raise

                # A 429 means the per-minute window is full: wait a full
                # minute. Other errors get short exponential backoff.
                if isinstance(error, RateLimitError):
                    # A daily cap will not clear by waiting a minute: fail
                    # now instead of stalling the whole build.
                    if _is_daily_limit(error):
                        raise
                    time.sleep(60)
                else:
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
