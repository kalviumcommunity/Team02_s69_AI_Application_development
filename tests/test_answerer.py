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


class FailingLLMClient:
    """LLM client that fails if called."""

    def __init__(self):
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        raise AssertionError(
            "LLM should not be called for low-confidence retrieval."
        )


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


class LowConfidenceVectorStore:
    """Return weakly relevant retrieved chunks."""

    def search(self, query, top_k=5):
        return [
            {
                "chunk": "Unrelated information.",
                "score": 0.10,
                "metadata": {
                    "scheme_name": "Unrelated Scheme",
                    "page_number": 1,
                    "section_title": "OTHER",
                },
            }
        ]


class NoCitationCompletions:
    """Return an answer without citations."""

    def create(self, **kwargs):
        return FakeResponse(
            "PM-KISAN provides support to eligible farmers."
        )


class NoCitationChat:
    """Fake chat endpoint returning an uncited answer."""

    def __init__(self):
        self.completions = NoCitationCompletions()


class NoCitationLLMClient:
    """Fake LLM client returning an answer without citations."""

    def __init__(self):
        self.chat = NoCitationChat()


def test_low_confidence_returns_insufficient_information():
    """Weak retrieval should return a refusal without calling the LLM."""
    answerer = QuestionAnswerer(
        vector_store=LowConfidenceVectorStore(),
        llm_client=FailingLLMClient(),
        model="fake-model",
    )

    result = answerer.answer_question("What is the weather in Delhi?")

    assert result["status"] == "insufficient_information"
    assert result["citations"] == []
    assert result["top_score"] == 0.10
    assert "not contain enough information" in result["answer"]


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
        "status",
        "answer",
        "citations",
        "retrieved_chunks",
        "top_score",
    }

    assert result["status"] == "answered"

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


def test_answer_without_citation_becomes_refusal():
    """Answers without citations should be converted to refusal."""
    answerer = QuestionAnswerer(
        vector_store=FakeVectorStore(),
        llm_client=NoCitationLLMClient(),
        model="fake-model",
    )

    result = answerer.answer_question(
        "Who is eligible for PM-KISAN?"
    )

    assert result["status"] == "insufficient_information"
    assert result["citations"] == []
    assert "not contain enough information" in result["answer"]
