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
6. Build the knowledge base (one command runs ingestion, chunking and
   indexing in sequence):
   ```bash
   python -m src.pipeline
   ```
   This calls the embeddings API once per chunk, so it needs a real
   `LLM_API_KEY` in `.env` and will use API quota. If you just want to work
   on ingestion or chunking without an API key or without re-embedding
   everything, skip the indexing step:
   ```bash
   python -m src.pipeline --skip-index
   ```
   CI runs this flag automatically as a smoke test, since it has no API key.
   Each step can also be run on its own: `python -m src.ingestion.run_ingestion`,
   `python -m src.ingestion.run_chunking`, `python -m src.indexing.build_index`.
7. Ask a question from the command line:
   ```bash
   python -m src.rag.ask "Who is eligible for PM-KISAN?"
   ```

Run the Streamlit app (once it exists) from the project root with `python -m streamlit run app/main.py`. Using `python -m` puts the project root on the import path so `from src...` imports work.

### Known limitation: scanned pages

A handful of source PDFs include scanned annexures or form templates with no
extractable text (for example, PMAY-G's later pages). These are detected and
skipped during ingestion — logged to `logs/ingestion_errors.log` — rather
than indexed as empty content. OCR support to recover them is not part of
the current scope.

## Project layout

| Path | Purpose |
|---|---|
| `documents/` | Source scheme PDFs |
| `src/config.py` | Settings read from environment variables |
| `src/pipeline.py` | One-command pipeline: ingest, chunk, build the index |
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

## RAG Confidence Guardrails

SchemeLens AI uses a configurable retrieval-confidence threshold to avoid
sending weakly related questions to the language model.

The initial threshold is set to `0.30` in `src/config.py`.

This starting value was selected by testing representative in-scope and
off-topic queries. PM-KISAN and PMAY-G queries produced scores above the
threshold, while clearly unrelated questions such as weather and general
knowledge queries produced lower scores.

If the highest retrieved chunk score is below the threshold, the system
returns an `insufficient_information` response without calling the LLM.

The system also checks the generated response for citations. If no valid
citation is present, the response is converted to the same
`insufficient_information` response.

Threshold calibration against a larger evaluation set is deferred to a
future iteration.
