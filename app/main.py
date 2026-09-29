
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
