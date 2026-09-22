from src.ingestion.chunker import chunk_page


def test_chunk_overlap_works():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 20

    chunks = chunk_page(
        scheme_id="test_scheme",
        page_number=1,
        text=text,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1

    first_text = chunks[0]["text"]
    second_text = chunks[1]["text"]

    assert first_text[-20:] == second_text[:20]


def test_chunk_id_is_unique():
    text = "This is some sample text. " * 100

    chunks = chunk_page(
        scheme_id="test_scheme",
        page_number=1,
        text=text,
        chunk_size=100,
        chunk_overlap=20,
    )

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_chunks_have_required_metadata():
    text = "This is some sample text. " * 100

    chunks = chunk_page(
        scheme_id="test_scheme",
        page_number=3,
        text=text,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert chunks

    for chunk in chunks:
        assert chunk["scheme_id"] == "test_scheme"
        assert chunk["page_number"] == 3
        assert chunk["chunk_id"]
        assert chunk["text"]


def test_chunk_output_contains_expected_fields():
    text = "This is sample content for testing chunk output."

    chunks = chunk_page(
        scheme_id="pmegp",
        page_number=5,
        text=text,
        chunk_size=900,
        chunk_overlap=150,
    )

    assert chunks

    expected_fields = {
        "chunk_id",
        "scheme_id",
        "section_title",
        "page_number",
        "text",
    }

    for chunk in chunks:
        assert set(chunk.keys()) == expected_fields
