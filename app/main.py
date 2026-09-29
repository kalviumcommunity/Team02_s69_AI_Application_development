
import streamlit as st
from openai import OpenAI

from src.config import CHAT_MODEL, CHAT_API_KEY, CHAT_BASE_URL
from src.indexing.embedder import OpenAIEmbedder
from src.indexing.vector_store import VectorStore
from src.rag.answer import QuestionAnswerer


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


citizen_tab, helpdesk_tab = st.tabs(["Citizen Chat", "Helpdesk View"])

with citizen_tab:
    render_citizen_chat()

with helpdesk_tab:
    render_helpdesk_view()
