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
4. Create your local settings file:
   ```bash
   cp .env.example .env        # Windows: copy .env.example .env
   ```
   `.env` is git-ignored. Never commit keys. Fill in two separate keys —
   see [API keys](#api-keys) below for why there are two and where to get
   each one free:
   - `EMBEDDING_API_KEY` — a Google AI Studio key (free tier). Everyone on
     the team must use this same provider/model, since the index can only
     be searched correctly with the model that built it.
   - `CHAT_API_KEY` — any OpenAI-compatible chat provider/key you have.
     This one does not need to match your teammates'.
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
   `EMBEDDING_API_KEY` in `.env` and will use API quota. If you just want to
   work on ingestion or chunking without an API key or without re-embedding
   everything, skip the indexing step:
   ```bash
   python -m src.pipeline --skip-index
   ```
   CI runs this flag automatically as a smoke test, since it has no API key.
   Each step can also be run on its own: `python -m src.ingestion.run_ingestion`,
   `python -m src.ingestion.run_chunking`, `python -m src.indexing.build_index`.
7. Ask a question from the command line (needs `CHAT_API_KEY` too):
   ```bash
   python -m src.rag.ask "Who is eligible for PM-KISAN?"
   ```
8. Run the app (has a Citizen Chat tab and a Helpdesk View tab that also
   shows the retrieved source passages behind each answer):
   ```bash
   python -m streamlit run app/main.py
   ```
   Using `python -m` puts the project root on the import path so
   `from src...` imports work regardless of the working directory.

### Current status

**Done:** PDF ingestion, chunking, embeddings and vector index (with
duplicate removal, and single-scheme re-indexing via
`python -m src.indexing.build_index --scheme <scheme_id>`), cited Q&A with
an insufficient-information refusal path, follow-up questions using
conversation context, the Streamlit app (citizen chat with clickable
sources and thumbs feedback, helpdesk source view), unanswered-query and
feedback logs (`logs/`), verified eligibility rules for PM-KISAN and
PMAY-G (`data/rules/`), and an evaluation question set covering all 8
schemes.

**Evaluation (latest run, 11 questions):**
- Retrieval hit rate: 9 of 9 in-corpus questions retrieved the expected
  scheme in the top 5 (100%, PRD target 85%).
- Refusal rate: 2 of 2 out-of-corpus questions refused (100%, PRD target 100%).
- Results are saved to `evaluation/results.json`.

**Known gaps:** the eligibility checker itself (matching a citizen's
profile against `data/rules/`) hasn't been built yet. Eligibility rules
exist for 2 of the 8 schemes. The evaluation set is small (11 questions),
so these numbers are a baseline rather than a full accuracy claim.

**Next:** the eligibility checker, then rules for the remaining schemes.

### API keys

SchemeLens AI uses two independent OpenAI-compatible API credentials,
configured separately in `.env`:

| | Used for | Must match across the team? | Default |
|---|---|---|---|
| `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL` | Building the vector index and embedding each query at search time | **Yes** | Google Gemini's `gemini-embedding-001`, via its OpenAI-compatible endpoint (free tier at [aistudio.google.com](https://aistudio.google.com/)) |
| `CHAT_API_KEY` / `CHAT_BASE_URL` / `CHAT_MODEL` | Generating the final answer text from retrieved context | No — any provider works | OpenAI `gpt-4o-mini` (change freely, e.g. to Groq, OpenRouter, or another free/available key) |

The embedding side must be consistent because a query embedded with a
different model than the one that built the index will not retrieve
correctly — vectors from different models are not comparable. The chat side
has no such constraint: each call is independent, so any teammate can point
it at whatever OpenAI-compatible chat provider and key they have.

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

SchemeLens AI uses a retrieval-confidence threshold to avoid sending weakly
related questions to the language model. If the best-matching chunk scores
below it, the system returns an `insufficient_information` response without
calling the LLM. The response is also refused if the generated answer has
no valid citation.

The threshold is `RAG_CONFIDENCE_THRESHOLD`, default **0.40**
(`src/config.py`).

**How it was set.** The top retrieval score was measured for every
evaluation question, using the current embedding model:

| Question type | Top scores |
|---|---|
| Answerable (in-corpus, 9 questions) | 0.51 to 0.66 |
| Off-topic (weather, cricket, cake, and others) | 0.09 to 0.24 |

0.40 sits between the two groups, with a margin of about 0.16 on each side.
The earlier default of 0.30 was chosen before the current embedding model and
would have left only about 0.06 of margin on the off-topic side.

**Re-check it when the embedding model changes.** Scores depend on the
embedding model. `python -m evaluation.run_eval` records each question's
`top_score` in `evaluation/results.json`, so the same check can be repeated
after a model change before trusting the threshold.

## Eligibility Checker

### Implementation Status

Implemented the Eligibility Checker feature for the citizen-facing application.

The checker allows a citizen to provide their profile information and receive an eligibility result based on the currently available verified scheme rules.

### Implemented

* Added `UserProfile` dataclass with optional fields:

  * Income
  * Land size in acres
  * Category
  * State
  * Age
* Added generic eligibility checker logic in `src/eligibility/checker.py`.
* Added `SchemeResult` and verdict handling:

  * **Eligible**
  * **Unclear**
  * **Not Eligible**
* Implemented rule evaluation using supported operators and threshold values.
* Missing profile information required by a rule produces **Unclear** instead of making an assumption.
* Eligibility results are ranked in the following order:

  1. Eligible
  2. Unclear
  3. Not Eligible
* Rule citations and relevant missing fields are preserved in the result.
* The checker discovers rule files dynamically from `data/rules/`, so additional scheme rule files can be added without changing the checker implementation.
* Added an **Eligibility Checker** tab to the Streamlit application.
* Added a structured form for:

  * Income
  * Land size
  * Category
  * State
  * Age
* Added a colour-coded results table displaying:

  * Scheme
  * Verdict
  * Rule/citation
  * Missing fields
* Added automated tests for:

  * Eligibility threshold evaluation
  * Missing profile fields
  * Unclear results
  * Rule citations
  * Result ranking
  * Dynamic discovery of available scheme rule files
  * Protection against treating missing values as zero/false

### Current Rule-Data Status

The Eligibility Checker implementation is designed to support all scheme rule files placed in `data/rules/`.

At the time of this implementation, only the existing rule files were available. The remaining scheme-specific eligibility rule data is being prepared separately and will be added through a separate pull request.

The checker therefore does not hard-code the number of schemes and will automatically include additional rule files when they are added.

### Testing

The following checks were completed:

```text
python -m pytest
58 passed, 1 warning
```

The warning is an existing ChromaDB deprecation warning and is unrelated to the Eligibility Checker implementation.

Additional validation:

```text
flake8 — passed
git diff --check — passed
Streamlit AppTest — passed
```

The Streamlit application was tested by submitting the Eligibility Checker form and verifying that the results table rendered without application exceptions.
