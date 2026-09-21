from pathlib import Path

import fitz

from src.ingestion.pdf_loader import load_pdf


def create_test_pdf(pdf_path: Path) -> None:
    """Create a PDF containing one valid and one blank page."""
    document = fitz.open()

    page = document.new_page()
    page.insert_text(
        (72, 72),
        "This page contains enough text to be ingested successfully.",
    )

    document.new_page()

    document.save(pdf_path)
    document.close()


def test_blank_page_is_skipped_and_logged(tmp_path):
    """Blank pages should be skipped and written to the error log."""
    pdf_path = tmp_path / "test.pdf"
    log_path = tmp_path / "errors.log"

    create_test_pdf(pdf_path)

    metadata = {
        "test.pdf": {
            "scheme_id": "test_scheme",
            "scheme_name": "Test Scheme",
            "department": "Test Department",
            "source_url": "unknown",
            "doc_version": "unknown",
            "published_date": "unknown",
            "file_name": "test.pdf",
        }
    }

    records, pages_read, pages_skipped = load_pdf(
        pdf_path=pdf_path,
        metadata=metadata,
        log_path=log_path,
    )

    assert pages_read == 2
    assert pages_skipped == 1
    assert len(records) == 1

    assert records[0]["scheme_id"] == "test_scheme"
    assert records[0]["page_number"] == 1

    log_contents = log_path.read_text(
        encoding="utf-8"
    )

    assert "test_scheme" in log_contents
    assert "page 2" in log_contents


def test_page_numbers_start_at_one(tmp_path):
    """Extracted page numbers should start at 1."""
    pdf_path = tmp_path / "numbering.pdf"
    log_path = tmp_path / "errors.log"

    document = fitz.open()

    for number in range(1, 4):
        page = document.new_page()
        page.insert_text(
            (72, 72),
            f"This is test page {number} with enough text.",
        )

    document.save(pdf_path)
    document.close()

    metadata = {
        "numbering.pdf": {
            "scheme_id": "numbering_test",
            "scheme_name": "Numbering Test",
            "department": "Test Department",
            "source_url": "unknown",
            "doc_version": "unknown",
            "published_date": "unknown",
            "file_name": "numbering.pdf",
        }
    }

    records, pages_read, pages_skipped = load_pdf(
        pdf_path=pdf_path,
        metadata=metadata,
        log_path=log_path,
    )

    assert pages_read == 3
    assert pages_skipped == 0
    assert [record["page_number"] for record in records] == [1, 2, 3]
