
import streamlit as st
from openai import OpenAI

from src.config import CHAT_MODEL, CHAT_API_KEY, CHAT_BASE_URL
from src.indexing.embedder import OpenAIEmbedder
from src.indexing.vector_store import VectorStore
from src.rag.answer import QuestionAnswerer
from src.eligibility.checker import check_all
from src.eligibility.models import UserProfile


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


def render_citizen_chat() -> None:
    """The existing citizen-facing chat, unchanged from issue #11."""
    # Initialize the chat history.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages and their citations.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("insufficient"):
                st.warning(message["content"])
            else:
                st.markdown(message["content"])

            if message.get("citations"):
                st.markdown("**Sources**")

                for citation in message["citations"]:
                    st.markdown(
                        f"- **{citation['scheme_name']}** — "
                        f"Page {citation['page']}, "
                        f"{citation['section']}"
                    )

    # Receive the user's question.
    question = st.chat_input("Ask about a government scheme...")

    if question:
        # Show and save the user's question.
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        # Generate the answer using the existing RAG backend.
        with st.chat_message("assistant"):
            try:
                answerer = get_answerer()
                response = answerer.answer_question(question)

                answer = response["answer"]
                citations = response.get("citations", [])
                retrieved_chunks = response.get(
                    "retrieved_chunks", []
                )

                # The current backend returns no chunks when it
                # cannot find sufficient document information.
                insufficient = not retrieved_chunks

                if insufficient:
                    st.warning(answer)
                else:
                    st.markdown(answer)

                if citations:
                    st.markdown("**Sources**")

                    for citation in citations:
                        st.markdown(
                            f"- **{citation['scheme_name']}** — "
                            f"Page {citation['page']}, "
                            f"{citation['section']}"
                        )

                # Save the assistant's response for future reruns.
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "insufficient": insufficient,
                })

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

            except Exception:
                st.error(
                    "Something went wrong while answering "
                    "your question. Please try again."
                )


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
            response = get_answerer().answer_question(question)
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
