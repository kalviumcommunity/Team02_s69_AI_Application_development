"""End-to-end data pipeline: ingest PDFs, chunk them, build the vector index.

This is the one command a new teammate (or CI) needs to stand up the
knowledge base, instead of running ingestion, chunking and indexing as three
separate manual steps:

    python -m src.pipeline

Ingestion and chunking are free and deterministic: they only read local PDFs
and slice text, with no external calls and no credentials required. Building
the vector index is different: it calls the embeddings API once per chunk,
which costs money and requires a real ``LLM_API_KEY``. Use ``--skip-index``
to run only the free steps:

    python -m src.pipeline --skip-index

This is what CI uses as a pipeline smoke test on every push, since CI has no
API key. It is also useful locally when iterating on ingestion or chunking
logic without wanting to re-embed (and re-pay for) the full corpus each run.
Once you are ready for a working, queryable index, run the full command
(or ``python -m src.indexing.build_index`` on its own).
"""

from __future__ import annotations

import argparse

from src.indexing.build_index import main as run_build_index
from src.ingestion.run_chunking import main as run_chunking
from src.ingestion.run_ingestion import main as run_ingestion


def _print_step(step: str, title: str) -> None:
    print()
    print("=" * 60)
    print(f"{step}  {title}")
    print("=" * 60)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse pipeline command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the SchemeLens AI data pipeline: ingest scheme PDFs, "
            "chunk them, and build the vector index."
        )
    )

    parser.add_argument(
        "--skip-index",
        action="store_true",
        help=(
            "Skip the embedding and vector index build step. "
            "No API key is needed and no API calls are made."
        ),
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run ingestion, chunking, and (unless skipped) index building."""
    args = parse_args(argv)

    _print_step("Step 1/3", "PDF ingestion")
    run_ingestion()

    _print_step("Step 2/3", "Chunking")
    run_chunking()

    if args.skip_index:
        _print_step("Step 3/3", "Skipped (--skip-index)")
        print(
            "Run `python -m src.indexing.build_index` later once "
            "LLM_API_KEY is set in .env."
        )
        return

    _print_step("Step 3/3", "Embeddings and vector index")
    run_build_index()

    print()
    print("Pipeline complete. The vector index is ready to query, e.g.:")
    print('  python -m src.rag.ask "Who is eligible for PM-KISAN?"')


if __name__ == "__main__":
    main()
