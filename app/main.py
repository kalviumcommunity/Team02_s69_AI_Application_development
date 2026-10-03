
import json

import streamlit as st
from openai import OpenAI

from src.config import CHAT_MODEL, CHAT_API_KEY, CHAT_BASE_URL, DATA_DIR
from src.indexing.embedder import OpenAIEmbedder
from src.indexing.vector_store import VectorStore
from src.query_log import log_feedback, log_unanswered
from src.rag.answer import QuestionAnswerer
from src.rag.citations import display_section, unique_citations


st.set_page_config(
    page_title="SchemeLens AI Chat",
    page_icon="💬",
    layout="centered",
)

st.title("SchemeLens AI")
st.caption(
    "Ask questions about government schemes "
    "and get answers with document citations."
)


# Cache the answerer and its vector store between reruns.
@st.cache_resource
def get_answerer():
    if not CHAT_API_KEY:
        raise ValueError("API key is not configured.")

    embedder = OpenAIEmbedder()
    vector_store = VectorStore(embedder=embedder)

    client = OpenAI(
        api_key=CHAT_API_KEY,
        base_url=CHAT_BASE_URL,
    )

    return QuestionAnswerer(
        vector_store=vector_store,
        llm_client=client,
        model=CHAT_MODEL,
    )


@st.cache_data
def scheme_source_urls() -> dict[str, str]:
    """Map scheme name to the official PDF URL, for clickable citations."""
    with (DATA_DIR / "schemes.json").open("r", encoding="utf-8") as file:
        schemes = json.load(file)

    return {scheme["scheme_name"]: scheme["source_url"] for scheme in schemes}


def _citation_markdown(citation: dict) -> str:
    """One citation as a link to the source PDF at the cited page."""
    label = f"{citation['scheme_name']}, page {citation['page']}"

    section = display_section(citation.get("section"))

    if section:
        label += f", {section}"

    url = scheme_source_urls().get(citation["scheme_name"])

    if url:
        return f"- [{label}]({url}#page={citation['page']})"

    return f"- {label}"


def _render_citations(citations: list[dict]) -> None:
    citations = unique_citations(citations)

    if not citations:
        return

    st.markdown("**Sources**")

    for citation in citations:
        st.markdown(_citation_markdown(citation))


def _record_feedback(question: str, rating: str) -> None:
    """on_click handler: log a thumbs-up or thumbs-down for an answer."""
    log_feedback(question, rating)


def _conversation_history() -> list[dict]:
    """Prior turns as role/content pairs, for resolving follow-up questions."""
    return [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.get("messages", [])
    ]


def _answer(question: str, history: list[dict] | None = None) -> dict:
    """Run one question through the RAG backend and log refusals."""
    response = get_answerer().answer_question(question, history=history)

    if response["status"] == "insufficient_information":
        top_score = (
            response["retrieved_chunks"][0]["score"]
            if response.get("retrieved_chunks")
            else None
        )
        log_unanswered(question, top_score)

    return response


def render_citizen_chat() -> None:
    """The citizen-facing chat: follow-ups, citations, and feedback."""
    st.session_state.setdefault("messages", [])

    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message.get("insufficient"):
                st.warning(message["content"])
            else:
                st.markdown(message["content"])

            if message["role"] == "assistant":
                _render_citations(message.get("citations", []))

                if not message.get("insufficient"):
                    st.button(
                        "👍",
                        key=f"up_{index}",
                        on_click=_record_feedback,
                        args=(message["question"], "up"),
                    )
                    st.button(
                        "👎",
                        key=f"down_{index}",
                        on_click=_record_feedback,
                        args=(message["question"], "down"),
                    )

    question = st.chat_input("Ask about a government scheme...")

    if not question:
        return

    history = _conversation_history()

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Checking the scheme documents..."):
                response = _answer(question, history=history)
        except ValueError as error:
            if "API key" in str(error):
                st.error(
                    "The AI service is not configured. "
                    "Please check your API key."
                )
            else:
                st.error(
                    "Unable to process your question. "
                    "Please check the app configuration."
                )
            return
        except Exception:
            st.error(
                "Something went wrong while answering "
                "your question. Please try again."
            )
            return

        insufficient = response["status"] == "insufficient_information"

        if insufficient:
            st.warning(response["answer"])
        else:
            st.markdown(response["answer"])

        _render_citations(response["citations"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "citations": response["citations"],
            "insufficient": insufficient,
            "question": question,
        })


def _is_cited(chunk: dict, citations: list[dict]) -> bool:
    """A retrieved chunk counts as cited if a citation was built from it.

    Citations are derived from retrieved_chunks in the answerer, so an
    exact match on chunk text and page number reliably identifies which
    retrieved chunks the answer actually referenced.
    """
    metadata = chunk["metadata"]

    return any(
        citation["snippet"] == chunk["chunk"]
        and citation["page"] == metadata["page_number"]
        for citation in citations
    )


def render_helpdesk_view() -> None:
    """Same Q&A as the citizen chat, plus every retrieved source passage
    — for staff to verify an answer before relaying it to a citizen.
    """
    st.caption(
        "Ask the same question a citizen would, and see the exact "
        "source passages behind the answer before sharing it."
    )

    question = st.text_input(
        "Question",
        key="helpdesk_question",
        placeholder="e.g. Who is eligible for PMAY-G?",
    )

    if not st.button("Search", key="helpdesk_search"):
        return

    if not question.strip():
        st.warning("Enter a question first.")
        return

    try:
        with st.spinner("Retrieving and answering..."):
            response = _answer(question)
    except ValueError as error:
        if "API key" in str(error):
            st.error("The AI service is not configured. Please check your API key.")
        else:
            st.error("Unable to process your question. Please check the app configuration.")
        return
    except Exception:
        st.error("Something went wrong while answering your question. Please try again.")
        return

    if response["status"] == "insufficient_information":
        st.warning(response["answer"])
    else:
        st.markdown("**Answer:**")
        st.markdown(response["answer"])
        _render_citations(response["citations"])

    st.divider()

    retrieved_chunks = response.get("retrieved_chunks", [])
    st.markdown(f"**Retrieved passages** ({len(retrieved_chunks)})")

    if not retrieved_chunks:
        st.caption("No chunks were retrieved for this question.")
        return

    citations = response.get("citations", [])

    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk["metadata"]
        cited = _is_cited(chunk, citations)
        label = (
            f"{'✓ Cited — ' if cited else ''}"
            f"[{index}] {metadata['scheme_name']} — "
            f"page {metadata['page_number']} — score {chunk['score']:.2f}"
        )

        with st.expander(label, expanded=cited):
            st.caption(
                f"Department: {metadata['department']} | "
                f"Section: {metadata['section_title'] or '(none)'}"
            )
            st.text(chunk["chunk"])


citizen_tab, helpdesk_tab = st.tabs(["Citizen Chat", "Helpdesk View"])

with citizen_tab:
    render_citizen_chat()

with helpdesk_tab:
    render_helpdesk_view()
