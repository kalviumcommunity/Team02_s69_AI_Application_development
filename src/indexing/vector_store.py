"""Persistent ChromaDB vector store for SchemeLens AI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from src.config import VECTOR_DB_DIR


COLLECTION_NAME = "scheme_chunks"


class VectorStore:
    """Store and search embedded scheme document chunks."""

    def __init__(
        self,
        persist_directory: Path = VECTOR_DB_DIR,
        embedder: Any | None = None,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self.embedder = embedder
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Delete and recreate the collection."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        batch_size: int = 64,
    ) -> None:
        """Add embedded chunks and metadata to ChromaDB."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        for start in range(0, len(chunks), batch_size):
            chunk_batch = chunks[start:start + batch_size]
            embedding_batch = embeddings[start:start + batch_size]

            self.collection.add(
                ids=[
                    chunk["chunk_id"]
                    for chunk in chunk_batch
                ],
                embeddings=embedding_batch,
                documents=[
                    chunk["text"]
                    for chunk in chunk_batch
                ],
                metadatas=[
                    {
                        "scheme_id": chunk["scheme_id"],
                        "scheme_name": chunk["scheme_name"],
                        "department": chunk["department"],
                        "section_title": chunk["section_title"],
                        "page_number": chunk["page_number"],
                    }
                    for chunk in chunk_batch
                ],
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        scheme_id: str | None = None,
    ) -> list[dict]:
        """Search the vector store for relevant chunks."""
        if self.embedder is None:
            raise ValueError("An embedder is required for search.")

        if not query.strip():
            return []

        query_embedding = self.embedder.embed([query])[0]

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }

        if scheme_id:
            query_kwargs["where"] = {
                "scheme_id": scheme_id,
            }

        results = self.collection.query(**query_kwargs)

        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        output: list[dict] = []

        for document, distance, metadata in zip(
            documents,
            distances,
            metadatas,
        ):
            output.append(
                {
                    "chunk": document,
                    "score": 1.0 - distance,
                    "metadata": metadata,
                }
            )

        return output
