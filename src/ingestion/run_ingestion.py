"""Command-line entry point for PDF ingestion."""

from __future__ import annotations

import json
from pathlib import Path

from src.ingestion.pdf_loader import (
    load_pdf,
    load_scheme_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENTS_DIR = PROJECT_ROOT / "documents"
METADATA_PATH = PROJECT_ROOT / "data" / "schemes.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "pages"
LOG_PATH = PROJECT_ROOT / "logs" / "ingestion_errors.log"


def main() -> None:
    """Process all PDFs in the documents directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    metadata = load_scheme_metadata(METADATA_PATH)

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    documents_processed = 0
    total_pages_read = 0
    total_pages_skipped = 0

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        records, pages_read, pages_skipped = load_pdf(
            pdf_path=pdf_path,
            metadata=metadata,
            log_path=LOG_PATH,
        )

        documents_processed += 1
        total_pages_read += pages_read
        total_pages_skipped += pages_skipped

        if records:
            scheme_id = records[0]["scheme_id"]
            output_path = OUTPUT_DIR / f"{scheme_id}.jsonl"

            with output_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                for record in records:
                    file.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    print("\nIngestion Summary")
    print("-----------------")
    print(f"Documents: {documents_processed}")
    print(f"Pages read: {total_pages_read}")
    print(f"Pages skipped: {total_pages_skipped}")


if __name__ == "__main__":
    main()
