"""Utilities for extracting text from scheme PDF documents."""

from __future__ import annotations

import json
from pathlib import Path

import fitz


MIN_TEXT_LENGTH = 20


def load_scheme_metadata(metadata_path: Path) -> dict[str, dict]:
    """Load scheme metadata and index it by PDF filename."""
    with metadata_path.open("r", encoding="utf-8") as file:
        schemes = json.load(file)

    return {
        scheme["file_name"]: scheme
        for scheme in schemes
    }


def log_ingestion_error(
    log_path: Path,
    scheme_id: str,
    page_number: int,
    reason: str,
) -> None:
    """Append an ingestion error to the log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as file:
        file.write(
            f"{scheme_id}, page {page_number}, {reason}\n"
        )


def load_pdf(
    pdf_path: Path,
    metadata: dict[str, dict],
    log_path: Path,
) -> tuple[list[dict], int, int]:
    """
    Extract usable text from a PDF page by page.

    Returns:
        records: Valid extracted page records.
        pages_read: Total number of pages processed.
        pages_skipped: Number of pages skipped.
    """
    file_name = pdf_path.name

    if file_name not in metadata:
        raise ValueError(
            f"No metadata found for PDF: {file_name}"
        )

    scheme_id = metadata[file_name]["scheme_id"]

    records: list[dict] = []
    pages_read = 0
    pages_skipped = 0

    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        log_ingestion_error(
            log_path,
            scheme_id,
            0,
            f"unreadable PDF: {exc}",
        )
        return records, pages_read, pages_skipped

    try:
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            pages_read += 1

            try:
                text = page.get_text("text").strip()
            except Exception as exc:
                pages_skipped += 1
                log_ingestion_error(
                    log_path,
                    scheme_id,
                    page_number,
                    f"unreadable page: {exc}",
                )
                continue

            if len(text) < MIN_TEXT_LENGTH:
                pages_skipped += 1
                log_ingestion_error(
                    log_path,
                    scheme_id,
                    page_number,
                    (
                        f"text length below "
                        f"{MIN_TEXT_LENGTH} characters"
                    ),
                )
                continue

            records.append(
                {
                    "scheme_id": scheme_id,
                    "page_number": page_number,
                    "text": text,
                }
            )
    finally:
        document.close()

    return records, pages_read, pages_skipped
