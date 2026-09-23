"""Build the complete SchemeLens AI vector index."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import DATA_DIR
from src.indexing.embedder import OpenAIEmbedder
from src.indexing.vector_store import VectorStore


CHUNKS_FILE = DATA_DIR / "processed" / "chunks.jsonl"
SCHEMES_FILE = DATA_DIR / "schemes.json"


def load_chunks(
    chunks_file: Path = CHUNKS_FILE,
    schemes_file: Path = SCHEMES_FILE,
) -> list[dict]:
    """Load chunks and attach scheme metadata."""
    with schemes_file.open("r", encoding="utf-8") as file:
        schemes = json.load(file)

    scheme_lookup = {
        scheme["scheme_id"]: scheme
        for scheme in schemes
    }

    chunks: list[dict] = []

    with chunks_file.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            chunk = json.loads(line)

            scheme = scheme_lookup.get(chunk["scheme_id"])

            if scheme is None:
                raise ValueError(
                    f"No scheme metadata found for "
                    f"{chunk['scheme_id']}"
                )

            chunk["scheme_name"] = scheme["scheme_name"]
            chunk["department"] = scheme["department"]

            chunks.append(chunk)

    return chunks


def main() -> None:
    """Build the complete vector index from all chunks."""
    print("Loading chunks...")

    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks.")

    embedder = OpenAIEmbedder()
    vector_store = VectorStore(embedder=embedder)

    print("Resetting vector collection...")
    vector_store.reset()

    print("Generating embeddings...")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed(texts)

    print(f"Generated {len(embeddings)} embeddings.")

    print("Adding chunks to ChromaDB...")

    vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    print("Index build complete.")
    print(f"Indexed {len(chunks)} chunks.")
    print("Vector database: data/chroma")


if __name__ == "__main__":
    main()
