"""Placeholder tests so CI has something to run; also guards the config module."""
from pathlib import Path

from src import config


def test_paths_are_absolute():
    assert config.DOCS_DIR.is_absolute()
    assert config.VECTOR_DB_DIR.is_absolute()


def test_settings_have_expected_types():
    assert isinstance(config.PROJECT_ROOT, Path)
    assert isinstance(config.LLM_API_KEY, str)
    assert config.LLM_BASE_URL
    assert config.CHAT_MODEL
    assert config.EMBEDDING_MODEL


def test_project_layout_exists():
    for name in ("src", "app", "evaluation", "tests", "data", "logs"):
        assert (config.PROJECT_ROOT / name).is_dir(), name
