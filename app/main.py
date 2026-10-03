
import json

import streamlit as st
from openai import OpenAI

from src.config import CHAT_MODEL, CHAT_API_KEY, CHAT_BASE_URL, DATA_DIR
from src.indexing.embedder import OpenAIEmbedder
from src.indexing.vector_store import VectorStore
from src.query_log import log_feedback, log_unanswered
from src.rag.answer import QuestionAnswerer
from src.eligibility.checker import check_all
from src.eligibility.models import UserProfile
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


def _optional_number(value: str, label: str) -> float | None:
    """Parse an optional form value without turning blanks into zero."""
    if not value.strip():
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a number or left blank.") from error
    if parsed < 0:
        raise ValueError(f"{label} cannot be negative.")
    return parsed


def render_eligibility_checker() -> None:
    """Collect optional profile facts and display the rule-based results."""
    st.caption(
        "Enter what you know. Leave any field blank when you are unsure; "
        "the checker will report Unclear when that information is needed."
    )

    with st.form("eligibility_profile"):
        income_text = st.text_input(
            "Annual income (INR)",
            placeholder="Leave blank if unknown",
            help="Enter annual income in rupees. Do not enter a monthly amount.",
        )
        land_text = st.text_input(
            "Land size (acres)", placeholder="Leave blank if unknown"
        )
        category = st.selectbox(
            "Social category",
            ["", "General", "OBC", "SC", "ST", "Other"],
            format_func=lambda value: value or "Select or leave blank",
        )
        state = st.text_input("State / Union Territory", placeholder="Optional")
        age_text = st.text_input("Age (years)", placeholder="Leave blank if unknown")
        submitted = st.form_submit_button("Check eligibility")

    if not submitted:
        return

    try:
        income = _optional_number(income_text, "Annual income")
        land_acres = _optional_number(land_text, "Land size")
        age_value = _optional_number(age_text, "Age")
        if age_value is not None and not age_value.is_integer():
            raise ValueError("Age must be a whole number or left blank.")
        profile = UserProfile(
            income=income,
            land_acres=land_acres,
            category=category or None,
            state=state.strip() or None,
            age=int(age_value) if age_value is not None else None,
        )
    except ValueError as error:
        st.error(str(error))
        return

    results = check_all(profile)
    rows = []
    for result in results:
        cited = result.failing_rule or (result.cited_rules[0] if result.cited_rules else None)
        rows.append({
            "Scheme": result.scheme_name,
            "Verdict": result.verdict.value,
            "Rule cited": (
                f"{cited['rule_id']}: {cited['condition']}" if cited else "—"
            ),
            "Missing fields": ", ".join(result.missing_fields) or "—",
            "Citation/page": result.citation or "—",
        })

    import pandas as pd

    frame = pd.DataFrame(rows)

    def verdict_style(value: str) -> str:
        colors = {
            "Eligible": "background-color: #d1fae5; color: #065f46; font-weight: bold",
            "Unclear": "background-color: #fef3c7; color: #92400e; font-weight: bold",
            "Not Eligible": "background-color: #fee2e2; color: #991b1b; font-weight: bold",
        }
        return colors.get(value, "")

    st.dataframe(
        frame.style.map(verdict_style, subset=["Verdict"]),
        width="stretch",
        hide_index=True,
    )

    for result in results:
        if result.unresolved_rules:
            with st.expander(f"Why {result.scheme_name} is unclear"):
                st.write("; ".join(result.unresolved_rules))


citizen_tab, helpdesk_tab, eligibility_tab = st.tabs([
    "Citizen Chat", "Helpdesk View", "Eligibility Checker"
])

with citizen_tab:
    render_citizen_chat()

with helpdesk_tab:
    render_helpdesk_view()

with eligibility_tab:
    render_eligibility_checker()
