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


# OpenAI-compatible API. The key has no default and is never committed.
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Paths
DOCS_DIR = _resolve_path(os.getenv("DOCS_DIR", "documents"))
VECTOR_DB_DIR = _resolve_path(os.getenv("VECTOR_DB_DIR", "data/chroma"))
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
