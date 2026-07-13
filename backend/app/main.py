from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request

from app.config import settings
from app.llm.factory import get_provider
from app.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SourceChunk,
    UploadResponse,
)
from app.rag.pipeline import answer_question, ingest_document

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Support RAG Chatbot", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", llm_provider=settings.llm_provider)


@app.post("/upload", response_model=UploadResponse)
@limiter.limit(settings.rate_limit)
async def upload(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    content = await file.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        provider = get_provider()
        document_id, chunk_count = ingest_document(file.filename, content, provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return UploadResponse(document_id=document_id, filename=file.filename, chunk_count=chunk_count)


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.rate_limit)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    try:
        provider = get_provider()
        answer, sources = answer_question(body.document_id, body.question, provider)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        answer=answer,
        sources=[SourceChunk(text=text, score=score) for text, score in sources],
    )
