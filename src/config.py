"""Central configuration for SchemeLens AI.

Values come from environment variables (loaded from a local ``.env`` file if
present) and are read once, when this module is first imported. Real
environment variables take precedence over ``.env``. Relative paths are
resolved from the project root, so the app behaves the same whichever
directory it is started from.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _resolve_path(value: str) -> Path:
    """Return ``value`` as an absolute path, anchored at the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


# Embeddings: used to build the vector index and to embed each query at
# search time. These MUST match — a query embedded with a different model
# than the one that built the index will not retrieve correctly. Everyone
# on the team should therefore use the same embedding settings.
#
# Default: Google's Gemini embedding model via its OpenAI-compatible
# endpoint, which has a free tier. Get a key from https://aistudio.google.com/
# The key has no default and is never committed.
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_BASE_URL = os.getenv(
    "EMBEDDING_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# Chat LLM: used only to generate the final answer text from retrieved
# context. Unlike embeddings, this does not need to match across teammates
# — any OpenAI-compatible provider, model and API key works here.
# The key has no default and is never committed.
CHAT_API_KEY = os.getenv("CHAT_API_KEY", "")
CHAT_BASE_URL = os.getenv("CHAT_BASE_URL", "https://api.openai.com/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Chunking configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# RAG confidence configuration
RAG_CONFIDENCE_THRESHOLD = float(
    os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.30")
)

# Paths
DOCS_DIR = _resolve_path(os.getenv("DOCS_DIR", "documents"))
VECTOR_DB_DIR = _resolve_path(
    os.getenv("VECTOR_DB_DIR", "data/vector_index")
)
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
