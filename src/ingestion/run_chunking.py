"""Run the full document chunking pipeline."""

from __future__ import annotations

from src.ingestion.chunker import build_chunks


def main() -> None:
    """Build chunks and print a summary."""
    counts = build_chunks()

    print("\nChunking Summary")
    print("----------------")

    total = 0

    for scheme_id, count in sorted(counts.items()):
        print(f"{scheme_id}: {count} chunks")
        total += count

    print(f"\nTotal chunks: {total}")
    print("\nOutput: data/processed/chunks.jsonl")


if __name__ == "__main__":
    main()