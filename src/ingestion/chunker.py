"""Chunk processed PDF pages into traceable text chunks."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR


PAGES_DIR = DATA_DIR / "processed" / "pages"
CHUNKS_FILE = DATA_DIR / "processed" / "chunks.jsonl"


def _clean_text(text: str) -> str:
    """Normalize whitespace while preserving readable text."""
    lines = []

    for line in text.splitlines():
        cleaned = " ".join(line.split())
        if cleaned:
            lines.append(cleaned)

    return "\n".join(lines)


def _is_heading(line: str) -> bool:
    """Return True when a line looks like a simple section heading."""
    line = line.strip()

    if not line or len(line) > 100:
        return False

    # Numbered headings such as:
    # 3. Eligibility
    # 3.1 Application Process
    if re.match(r"^\d+(?:\.\d+)*[\.\)]?\s+\S+", line):
        return True

    # ALL CAPS headings such as:
    # ELIGIBILITY
    # APPLICATION PROCESS
    letters = re.sub(r"[^A-Za-z]", "", line)

    if not letters:
        return False

    return letters.isupper() and len(letters) >= 3


def _find_section_titles(text: str) -> list[tuple[int, str]]:
    """Return character positions paired with the active section title."""
    current_section = ""
    result: list[tuple[int, str]] = []
    position = 0

    for line in text.splitlines():
        cleaned = " ".join(line.split())

        if not cleaned:
            position += len(line) + 1
            continue

        if _is_heading(cleaned):
            current_section = cleaned

        result.append((position, current_section))

        position += len(line) + 1

    return result


def chunk_page(
    scheme_id: str,
    page_number: int,
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Create traceable fixed-size overlapping chunks from one page."""

    text = _clean_text(text)

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    section_titles = _find_section_titles(text)

    chunks: list[dict] = []
    start = 0
    chunk_number = 1

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Prefer a sentence boundary when it is reasonably close
        # to the configured chunk size.
        if end < len(text):
            sentence_end = text.rfind(".", start, end)

            if sentence_end > start + (chunk_size // 2):
                end = sentence_end + 1

        chunk_text_value = text[start:end]

        # Remove only surrounding whitespace for stored text.
        chunk_text_value = chunk_text_value.strip()

        if chunk_text_value:
            section_title = ""

            for title_start, title in section_titles:
                if title_start <= start:
                    section_title = title
                else:
                    break

            chunks.append(
                {
                    "chunk_id": f"{scheme_id}-p{page_number}-c{chunk_number}",
                    "scheme_id": scheme_id,
                    "section_title": section_title,
                    "page_number": page_number,
                    "text": chunk_text_value,
                }
            )

            chunk_number += 1

        if end >= len(text):
            break

        # Move forward while retaining the requested overlap.
        start = end - chunk_overlap

    return chunks


def load_page_records(pages_dir: Path = PAGES_DIR) -> list[dict]:
    """Load page records from all processed page JSONL files."""
    records: list[dict] = []

    for file_path in sorted(pages_dir.glob("*.jsonl")):
        with file_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                records.append(json.loads(line))

    return records


def build_chunks(
    pages_dir: Path = PAGES_DIR,
    output_file: Path = CHUNKS_FILE,
) -> dict[str, int]:
    """Chunk all processed pages and write chunks.jsonl."""
    pages = load_page_records(pages_dir)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}

    with output_file.open("w", encoding="utf-8") as output:
        for page in pages:
            scheme_id = page["scheme_id"]
            page_number = page["page_number"]
            text = page["text"]

            chunks = chunk_page(
                scheme_id=scheme_id,
                page_number=page_number,
                text=text,
            )

            counts.setdefault(scheme_id, 0)

            for chunk in chunks:
                output.write(
                    json.dumps(
                        chunk,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                counts[scheme_id] += 1

    return counts
