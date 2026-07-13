# Support Doc Chatbot (RAG)

Upload a document (PDF/TXT/MD), then ask questions about it in plain English.
The app finds the relevant part of the document and answers using only that
content, with the source snippet shown so answers are verifiable, not
hallucinated.

Built as a demonstration project: FastAPI backend, FAISS for retrieval, a
Streamlit chat UI, and a swappable LLM layer (Gemini or OpenAI).

## Architecture

```
Streamlit UI  --->  FastAPI backend  --->  LLMProvider (Gemini or OpenAI)
                          |
                          v
                  FAISS vector store
                  (one index per uploaded doc, saved to disk)
```

Flow:

1. **Upload**: file text is extracted, split into overlapping chunks, each
   chunk is embedded, and the embeddings are stored in a FAISS index on disk
   keyed by a generated `document_id`.
2. **Chat**: the question is embedded the same way, FAISS returns the
   closest-matching chunks, and those chunks are handed to the LLM as context
   so it answers from the document instead of guessing.

## Why a swappable LLM provider

`app/llm/base.py` defines one interface (`embed`, `embed_query`, `generate`).
`app/llm/gemini_provider.py` and `app/llm/openai_provider.py` both implement
it; `app/llm/factory.py` picks one based on the `LLM_PROVIDER` env var. The
rest of the codebase (`app/rag/pipeline.py`, `app/main.py`) never imports a
vendor SDK directly — it only depends on the interface.

**Interview talking point:** *"I used Gemini's free tier while building this
to keep costs at zero, but the LLM layer is provider-agnostic — swapping to
OpenAI is a one-line env var change, no code changes. I designed it that way
deliberately so the app isn't locked to one vendor's pricing or uptime."*
That's a stronger answer than defending a single hardcoded vendor, and it's
true — try it: set `LLM_PROVIDER=openai` and add an `OPENAI_API_KEY`, nothing
else changes.

## Project structure

```
backend/
  app/
    main.py           FastAPI app: /upload, /chat, /health
    config.py          Settings from env vars
    models.py           Request/response schemas
    llm/
      base.py            Provider interface
      gemini_provider.py
      openai_provider.py
      factory.py          Picks provider from config
    rag/
      extraction.py      PDF/text extraction
      chunking.py         Overlapping word-window chunking
      vector_store.py     FAISS index read/write per document
      pipeline.py          Orchestrates ingest + query
  tests/
    test_chunking.py
    test_api.py           Uses a fake provider — no API key needed to run tests
frontend/
  streamlit_app.py       Upload + chat UI
docker-compose.yml        Runs both services together
.github/workflows/ci.yml  Runs pytest on every push
```

## Local setup

### 1. Get a free API key

- Gemini (recommended, free tier, no card required to start):
  https://aistudio.google.com/apikey
- OpenAI (optional, needed only if you want to demo the OpenAI path):
  https://platform.openai.com/api-keys

### 2. Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # then edit .env and paste your Gemini API key
uvicorn app.main:app --reload
```

Backend runs at http://localhost:8000 — check http://localhost:8000/docs for
the interactive API explorer FastAPI generates automatically.

### 3. Frontend (separate terminal)

```bash
cd frontend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens at http://localhost:8501. Upload a `.txt` or `.pdf`, then ask questions.

### 4. Run tests

```bash
cd backend
pytest
```

Tests use a fake in-memory provider, so they run without any API key or
network call — this is what the CI workflow runs on every push too.

## Docker (run both services together)

```bash
docker compose up --build
```

Backend on `:8000`, frontend on `:8501`.

## Deploying for free (to put a live link on your resume)

1. **Backend** → Render.com (free tier): connect the repo, point it at
   `backend/`, set the start command to
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, and add
   `LLM_PROVIDER` + `GEMINI_API_KEY` as environment variables in Render's
   dashboard.
2. **Frontend** → Streamlit Community Cloud (free): connect the repo, point
   it at `frontend/streamlit_app.py`, and set `BACKEND_URL` to your Render
   backend's public URL under "Secrets".

## Switching to OpenAI

Edit `.env`:

```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

No code changes needed — restart the backend.
