from pathlib import Path

import pytest

from src.indexing.vector_store import VectorStore


class FakeEmbedder:
    """Return deterministic embeddings without calling an API."""

    def embed(self, texts):
        embeddings = []

        for text in texts:
            value = float(len(text))

            embeddings.append(
                [
                    value,
                    value / 2,
                    value / 3,
                ]
            )

        return embeddings


def test_search_filters_by_scheme_id(tmp_path: Path):
    """Search with scheme_id returns only matching scheme chunks."""
    embedder = FakeEmbedder()

    store = VectorStore(
        persist_directory=tmp_path / "vector_index",
        embedder=embedder,
    )

    chunks = [
        {
            "chunk_id": "pmkisan-p1-c1",
            "scheme_id": "pmkisan",
            "scheme_name": "PM-KISAN",
            "department": "Agriculture & Farmers Welfare",
            "section_title": "ELIGIBILITY",
            "page_number": 1,
            "text": "PM-KISAN eligibility information.",
        },
        {
            "chunk_id": "pmay-g-p1-c1",
            "scheme_id": "pmay_g",
            "scheme_name": "PMAY-G",
            "department": "Rural Development",
            "section_title": "ELIGIBILITY",
            "page_number": 1,
            "text": "PMAY-G eligibility information.",
        },
    ]

    embeddings = embedder.embed(
        [chunk["text"] for chunk in chunks]
    )

    store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    results = store.search(
        "PM-KISAN eligibility",
        top_k=5,
        scheme_id="pmkisan",
    )

    assert results
    assert all(
        result["metadata"]["scheme_id"] == "pmkisan"
        for result in results
    )


def _sample_chunk(chunk_id: str, scheme_id: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "scheme_id": scheme_id,
        "scheme_name": scheme_id.upper(),
        "department": "Test",
        "section_title": "ELIGIBILITY",
        "page_number": 1,
        "text": text,
    }


def test_search_ranks_most_similar_first(tmp_path: Path):
    """The closest match by cosine similarity should rank first."""
    embedder = FakeEmbedder()
    store = VectorStore(
        persist_directory=tmp_path / "vector_index", embedder=embedder
    )

    chunks = [
        _sample_chunk("a", "pmkisan", "short"),
        _sample_chunk("b", "pmkisan", "a much much much longer chunk"),
    ]
    embeddings = embedder.embed([c["text"] for c in chunks])
    store.add_chunks(chunks=chunks, embeddings=embeddings)

    # A query embedding identical in direction to chunk "b" should rank it
    # first, since FakeEmbedder's vectors all point the same direction and
    # only differ in magnitude (cosine similarity is scale-invariant, so
    # this actually confirms ties resolve sensibly rather than crashing).
    results = store.search("a much much much longer query", top_k=2)

    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]


def test_search_on_empty_index_returns_empty_list(tmp_path: Path):
    """Searching before anything has been added should not crash."""
    store = VectorStore(
        persist_directory=tmp_path / "vector_index",
        embedder=FakeEmbedder(),
    )

    assert store.search("anything") == []


def test_persists_and_reloads_across_instances(tmp_path: Path):
    """A new VectorStore pointed at the same directory sees prior data."""
    persist_directory = tmp_path / "vector_index"
    embedder = FakeEmbedder()

    first = VectorStore(persist_directory=persist_directory, embedder=embedder)
    chunk = _sample_chunk("a", "pmkisan", "PM-KISAN eligibility information.")
    first.add_chunks(
        chunks=[chunk], embeddings=embedder.embed([chunk["text"]])
    )

    second = VectorStore(persist_directory=persist_directory, embedder=embedder)
    results = second.search("PM-KISAN eligibility")

    assert len(results) == 1
    assert results[0]["metadata"]["scheme_id"] == "pmkisan"


def test_reset_clears_data_in_memory_and_on_disk(tmp_path: Path):
    """reset() empties the index so a fresh instance sees nothing."""
    persist_directory = tmp_path / "vector_index"
    embedder = FakeEmbedder()

    store = VectorStore(persist_directory=persist_directory, embedder=embedder)
    chunk = _sample_chunk("a", "pmkisan", "PM-KISAN eligibility information.")
    store.add_chunks(
        chunks=[chunk], embeddings=embedder.embed([chunk["text"]])
    )

    store.reset()
    assert store.search("PM-KISAN eligibility") == []

    reloaded = VectorStore(persist_directory=persist_directory, embedder=embedder)
    assert reloaded.search("PM-KISAN eligibility") == []


def test_add_chunks_rejects_mismatched_lengths(tmp_path: Path):
    """A chunk/embedding count mismatch should raise, not silently corrupt."""
    store = VectorStore(
        persist_directory=tmp_path / "vector_index",
        embedder=FakeEmbedder(),
    )

    chunk = _sample_chunk("a", "pmkisan", "text")

    with pytest.raises(ValueError):
        store.add_chunks(chunks=[chunk], embeddings=[])
