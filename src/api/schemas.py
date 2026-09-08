from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    user_id: str
    session_id: str
    include_shared: bool = True
    top_k: int = 4
    history: list[dict] = []


class ChatResponse(BaseModel):
    answer: str
    steps: list[dict]
    latency_ms: float


class IngestStatus(BaseModel):
    doc_id: str
    status: str
    chunk_count: int = 0
    error: str | None = None