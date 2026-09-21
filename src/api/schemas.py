from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    user_id: str
    session_id: str
    include_shared: bool = True
    top_k: int = 4


class ChatResponse(BaseModel):
    answer: str
    steps: list[dict]
    latency_ms: float


class IngestStatus(BaseModel):
    doc_id: str
    status: str  # pending | processing | ready | failed
    chunk_count: int = 0
    error: str | None = None
