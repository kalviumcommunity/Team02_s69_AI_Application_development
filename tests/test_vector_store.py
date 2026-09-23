from pathlib import Path

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
        persist_directory=tmp_path / "chroma",
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
