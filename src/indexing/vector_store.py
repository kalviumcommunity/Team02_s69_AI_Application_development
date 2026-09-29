"""Pure NumPy vector store for SchemeLens AI.

This was originally built on ChromaDB's native HNSW index. It was replaced
because ChromaDB's compiled `hnswlib` backend segfaulted (silently, with no
Python traceback) on some Windows machines regardless of Python version,
chromadb version, embedding data, or directory state — a native-library
crash that could not be reproduced or fixed remotely.

For a corpus this size (a few thousand chunks at most), brute-force cosine
similarity in NumPy is comfortably fast (single-digit milliseconds per
query) and has no compiled native-library dependency of its own beyond
NumPy, which is far more mature and stable across Python versions than
ChromaDB's hnswlib/onnxruntime dependency chain. It cannot fail the way
ChromaDB did.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import VECTOR_DB_DIR


class VectorStore:
    """Store and search embedded scheme document chunks.

    Persists as two plain files in ``persist_directory``:
    ``vectors.npy`` (an [n_chunks, dim] float32 array) and
    ``metadata.jsonl`` (one JSON object per row, in the same order as the
    array). Both are plain data files with no compiled/native format.
    """

    def __init__(
        self,
        persist_directory: Path = VECTOR_DB_DIR,
        embedder: Any | None = None,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._vectors_path = self.persist_directory / "vectors.npy"
        self._metadata_path = self.persist_directory / "metadata.jsonl"

        self.embedder = embedder
        self._vectors: np.ndarray | None = None
        self._metadata: list[dict] = []

        self._load()

    def _load(self) -> None:
        """Load a previously persisted index from disk, if one exists."""
        if self._vectors_path.exists() and self._metadata_path.exists():
            self._vectors = np.load(self._vectors_path)

            with self._metadata_path.open("r", encoding="utf-8") as file:
                self._metadata = [
                    json.loads(line) for line in file if line.strip()
                ]
        else:
            self._vectors = None
            self._metadata = []

    def _save(self) -> None:
        """Persist the current index to disk."""
        if self._vectors is not None:
            np.save(self._vectors_path, self._vectors)

        with self._metadata_path.open("w", encoding="utf-8") as file:
            for entry in self._metadata:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def reset(self) -> None:
        """Clear the index, in memory and on disk."""
        self._vectors = None
        self._metadata = []

        for path in (self._vectors_path, self._metadata_path):
            path.unlink(missing_ok=True)

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        batch_size: int = 64,
    ) -> None:
        """Add embedded chunks and metadata to the index.

        ``batch_size`` is accepted for interface compatibility with the
        previous ChromaDB-backed implementation (which needed it to keep
        native calls small); it is unused here since a single in-memory
        NumPy concatenation and file write is already fast at this scale.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        if not chunks:
            return

        new_vectors = np.asarray(embeddings, dtype=np.float32)

        if self._vectors is None:
            self._vectors = new_vectors
        else:
            self._vectors = np.vstack([self._vectors, new_vectors])

        for chunk in chunks:
            self._metadata.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "scheme_id": chunk["scheme_id"],
                    "scheme_name": chunk["scheme_name"],
                    "department": chunk["department"],
                    "section_title": chunk["section_title"],
                    "page_number": chunk["page_number"],
                }
            )

        self._save()

    def search(
        self,
        query: str,
        top_k: int = 5,
        scheme_id: str | None = None,
    ) -> list[dict]:
        """Search the index for the most relevant chunks by cosine similarity."""
        if self.embedder is None:
            raise ValueError("An embedder is required for search.")

        if not query.strip():
            return []

        if self._vectors is None or not self._metadata:
            return []

        candidate_indices = list(range(len(self._metadata)))

        if scheme_id:
            candidate_indices = [
                i
                for i in candidate_indices
                if self._metadata[i]["scheme_id"] == scheme_id
            ]

        if not candidate_indices:
            return []

        query_embedding = np.asarray(
            self.embedder.embed([query])[0], dtype=np.float32
        )

        candidate_vectors = self._vectors[candidate_indices]
        similarities = _cosine_similarity(candidate_vectors, query_embedding)

        ranked = sorted(
            zip(candidate_indices, similarities),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_k]

        output: list[dict] = []

        for index, score in ranked:
            entry = self._metadata[index]

            output.append(
                {
                    "chunk": entry["text"],
                    "score": float(score),
                    "metadata": {
                        "scheme_id": entry["scheme_id"],
                        "scheme_name": entry["scheme_name"],
                        "department": entry["department"],
                        "section_title": entry["section_title"],
                        "page_number": entry["page_number"],
                    },
                }
            )

        return output


def _cosine_similarity(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine similarity between each row of ``vectors`` and ``query``."""
    vector_norms = np.linalg.norm(vectors, axis=1)
    query_norm = np.linalg.norm(query)

    # Guard against division by zero for a degenerate all-zero vector.
    # Shouldn't happen with real embeddings, but costs nothing to check.
    safe_vector_norms = np.where(vector_norms == 0, 1, vector_norms)
    safe_query_norm = query_norm if query_norm != 0 else 1

    dot_products = vectors @ query

    return dot_products / (safe_vector_norms * safe_query_norm)
