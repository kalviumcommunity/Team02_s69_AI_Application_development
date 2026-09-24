from src.rag.answerer import QuestionAnswerer


class FakeMessage:
    """Fake LLM message."""

    def __init__(self, content):
        self.content = content


class FakeChoice:
    """Fake LLM choice."""

    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    """Fake LLM response."""

    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    """Fake chat completions endpoint."""

    def create(self, **kwargs):
        return FakeResponse(
            "PM-KISAN provides support to eligible farmers [1]."
        )


class FakeChat:
    """Fake chat endpoint."""

    def __init__(self):
        self.completions = FakeCompletions()


class FakeLLMClient:
    """Fake OpenAI-compatible client."""

    def __init__(self):
        self.chat = FakeChat()


class FakeVectorStore:
    """Return deterministic retrieved chunks."""

    def search(self, query, top_k=5):
        assert query == "Who is eligible for PM-KISAN?"
        assert top_k == 5

        return [
            {
                "chunk": (
                    "PM-KISAN provides support to eligible farmers."
                ),
                "score": 0.91,
                "metadata": {
                    "scheme_name": "PM-KISAN",
                    "page_number": 3,
                    "section_title": "ELIGIBILITY",
                },
            },
            {
                "chunk": "Additional PM-KISAN information.",
                "score": 0.82,
                "metadata": {
                    "scheme_name": "PM-KISAN",
                    "page_number": 4,
                    "section_title": "BENEFITS",
                },
            },
        ]


def test_answer_question_maps_citation_to_metadata():
    """Citations should map [n] to real retrieved chunk metadata."""
    answerer = QuestionAnswerer(
        vector_store=FakeVectorStore(),
        llm_client=FakeLLMClient(),
        model="fake-model",
    )

    result = answerer.answer_question(
        "Who is eligible for PM-KISAN?"
    )

    assert set(result.keys()) == {
        "answer",
        "citations",
        "retrieved_chunks",
        "top_score",
    }

    assert result["answer"] == (
        "PM-KISAN provides support to eligible farmers [1]."
    )

    assert result["top_score"] == 0.91

    assert len(result["retrieved_chunks"]) == 2

    assert result["citations"] == [
        {
            "scheme_name": "PM-KISAN",
            "page": 3,
            "section": "ELIGIBILITY",
            "snippet": (
                "PM-KISAN provides support to eligible farmers."
            ),
        }
    ]
