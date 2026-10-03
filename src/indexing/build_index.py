"""Build the SchemeLens AI vector index.

Full rebuild (default):
    python -m src.indexing.build_index

Re-index one scheme only, leaving every other scheme's vectors untouched:
    python -m src.indexing.build_index --scheme pmkisan
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.config import DATA_DIR
from src.indexing.embedder import OpenAIEmbedder
from src.indexing.vector_store import VectorStore


CHUNKS_FILE = DATA_DIR / "processed" / "chunks.jsonl"
SCHEMES_FILE = DATA_DIR / "schemes.json"


def _content_key(text: str) -> str:
    """A fingerprint of a chunk's text, ignoring case and whitespace."""
    normalised = " ".join(text.lower().split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def dedupe_chunks(chunks: list[dict]) -> tuple[list[dict], int]:
    """Drop chunks whose text duplicates an earlier chunk in the same scheme.

    Re-ingesting an updated PDF can produce chunks identical to ones already
    indexed, or repeated boilerplate within one document. Keeps the first
    occurrence and returns the count removed.
    """
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    removed = 0

    for chunk in chunks:
        key = (chunk["scheme_id"], _content_key(chunk["text"]))

        if key in seen:
            removed += 1
            continue

        seen.add(key)
        unique.append(chunk)

    return unique, removed


def load_chunks(
    chunks_file: Path = CHUNKS_FILE,
    schemes_file: Path = SCHEMES_FILE,
    scheme_id: str | None = None,
) -> list[dict]:
    """Load chunks, attach scheme metadata, and drop duplicates.

    If ``scheme_id`` is given, only that scheme's chunks are returned.
    """
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

            if scheme_id is not None and chunk["scheme_id"] != scheme_id:
                continue

            scheme = scheme_lookup.get(chunk["scheme_id"])

            if scheme is None:
                raise ValueError(
                    f"No scheme metadata found for "
                    f"{chunk['scheme_id']}"
                )

            chunk["scheme_name"] = scheme["scheme_name"]
            chunk["department"] = scheme["department"]

            chunks.append(chunk)

    chunks, removed = dedupe_chunks(chunks)

    if removed:
        print(f"Removed {removed} duplicate chunks.")

    return chunks


def main(argv: list[str] | None = None) -> None:
    """Build the vector index, or re-index a single scheme."""
    parser = argparse.ArgumentParser(description="Build the vector index.")
    parser.add_argument(
        "--scheme",
        help="Re-index only this scheme_id (e.g. pmkisan). "
        "Other schemes' vectors are left untouched.",
    )
    args = parser.parse_args(argv)

    print("Loading chunks...")

    chunks = load_chunks(scheme_id=args.scheme)

    print(f"Loaded {len(chunks)} chunks.")

    embedder = OpenAIEmbedder()
    vector_store = VectorStore(embedder=embedder)

    # Embed first. Only once every embedding exists is the old index
    # touched, so a failed or rate-limited run never leaves it empty.
    print("Generating embeddings...")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed(texts)

    print(f"Generated {len(embeddings)} embeddings.")

    if args.scheme:
        print(f"Removing existing vectors for {args.scheme}...")
        vector_store.delete_scheme(args.scheme)
    else:
        print("Resetting vector collection...")
        vector_store.reset()

    print("Adding chunks to the vector index...")

    vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    print("Index build complete.")
    print(f"Indexed {len(chunks)} chunks.")
    print("Vector database: data/chroma")


if __name__ == "__main__":
    main()
