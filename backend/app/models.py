from pydantic import BaseModel


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int


class ChatRequest(BaseModel):
    document_id: str
    question: str


class SourceChunk(BaseModel):
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
