# SchemeLens AI

A RAG assistant that answers welfare-scheme eligibility and application questions from official scheme PDFs, with citations back to the scheme, page and section.

Team 02 · Squad 69 · Alliance campus · Sem 5, Sprint 2

## Setup

Requires Python 3.10 or newer.

1. Clone the repository and enter it:
   ```bash
   git clone https://github.com/kalviumcommunity/Team02_s69_AI_Application_development.git
   cd Team02_s69_AI_Application_development
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   # macOS / Linux
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create your local settings file and add your API key:
   ```bash
   cp .env.example .env        # Windows: copy .env.example .env
   ```
   `.env` is git-ignored. Never commit keys.
5. Check everything works:
   ```bash
   pytest
   flake8 src app tests
   ```

Run the Streamlit app (once it exists) from the project root with `python -m streamlit run app/main.py`. Using `python -m` puts the project root on the import path so `from src...` imports work.

## Project layout

| Path | Purpose |
|---|---|
| `documents/` | Source scheme PDFs |
| `src/config.py` | Settings read from environment variables |
| `src/ingestion/` | PDF text extraction and chunking |
| `src/indexing/` | Embeddings and vector store |
| `src/rag/` | Retrieval, prompting and answer generation |
| `src/eligibility/` | Eligibility rules and checker |
| `app/` | Streamlit interface |
| `evaluation/` | Evaluation questions and scripts |
| `data/` | Generated data (git-ignored, except `schemes.json` and `rules/`) |
| `logs/` | Runtime logs (git-ignored) |
| `tests/` | Unit tests |

## CI

GitHub Actions runs `flake8` and `pytest` on every push and pull request to `main` and `develop` (`.github/workflows/ci.yml`).
