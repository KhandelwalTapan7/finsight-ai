import time
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.ingestion.parser import parse_pdf
from src.ingestion.chunker import chunk_raw
from src.rag import vector_store
from src.agents.graph import run_agent
from src.telemetry import logger as telemetry
from src.api.schemas import ChatRequest, ChatResponse, IngestStatus

app = FastAPI(title="FinSight AI", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_ingestion_status: dict[str, IngestStatus] = {}

_query_cache: dict[tuple, tuple[dict, float]] = {}
CACHE_TTL_SECONDS = 3600


def _cache_key(query: str, user_id: str, session_id: str, include_shared: bool, top_k: int, history: list[dict]) -> tuple:
    history_sig = tuple((h.get("role"), h.get("content")) for h in history)
    return (query.strip().lower(), user_id, session_id, include_shared, top_k, history_sig)


def _ingest_job(doc_id: str, file_path: Path, filename: str, user_id: str, session_id: str) -> None:
    _ingestion_status[doc_id] = IngestStatus(doc_id=doc_id, status="processing")
    try:
        raw = parse_pdf(file_path, doc_id=doc_id)
        chunks = chunk_raw(raw)
        count = vector_store.upsert_chunks(
            chunks, source="user", user_id=user_id, session_id=session_id
        )
        _ingestion_status[doc_id] = IngestStatus(doc_id=doc_id, status="ready", chunk_count=count)
        telemetry.log_ingestion(
            user_id=user_id, session_id=session_id, doc_id=doc_id,
            filename=filename, status="ready", chunk_count=count,
        )
    except Exception as e:
        _ingestion_status[doc_id] = IngestStatus(doc_id=doc_id, status="failed", error=str(e))
        telemetry.log_ingestion(
            user_id=user_id, session_id=session_id, doc_id=doc_id,
            filename=filename, status="failed",
        )


@app.post("/upload", response_model=IngestStatus)
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = "demo-user",
    session_id: str = "demo-session",
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF uploads are supported right now.")

    doc_id = str(uuid.uuid4())[:8]
    user_dir = settings.LOCAL_UPLOAD_DIR / user_id / session_id
    user_dir.mkdir(parents=True, exist_ok=True)
    dest = user_dir / f"{doc_id}.pdf"
    dest.write_bytes(await file.read())

    _ingestion_status[doc_id] = IngestStatus(doc_id=doc_id, status="pending")
    background_tasks.add_task(_ingest_job, doc_id, dest, file.filename, user_id, session_id)
    return _ingestion_status[doc_id]


@app.get("/ingest/status/{doc_id}", response_model=IngestStatus)
async def ingest_status(doc_id: str):
    if doc_id not in _ingestion_status:
        raise HTTPException(404, "Unknown doc_id")
    return _ingestion_status[doc_id]


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.time()

    key = _cache_key(req.query, req.user_id, req.session_id, req.include_shared, req.top_k, req.history)
    cached = _query_cache.get(key)
    if cached and (time.time() - cached[1]) < CACHE_TTL_SECONDS:
        result = cached[0]
    else:
        result = run_agent(
            req.query, user_id=req.user_id, session_id=req.session_id,
            include_shared=req.include_shared, top_k=req.top_k, history=req.history,
        )
        _query_cache[key] = (result, time.time())

    latency_ms = (time.time() - start) * 1000
    telemetry.log_query(
        user_id=req.user_id, session_id=req.session_id, query=req.query,
        answer=result["answer"], latency_ms=latency_ms,
        tool_calls=sum(1 for s in result["steps"] if "tool" in s),
    )
    return ChatResponse(answer=result["answer"], steps=result["steps"], latency_ms=latency_ms)


@app.get("/admin/session-stats/{user_id}/{session_id}")
async def session_stats(user_id: str, session_id: str):
    return telemetry.get_session_stats(user_id, session_id)


@app.post("/admin/cleanup/{user_id}/{session_id}")
async def cleanup_session(user_id: str, session_id: str):
    vector_store.delete_user_data(user_id, session_id)
    user_dir = settings.LOCAL_UPLOAD_DIR / user_id / session_id
    if user_dir.exists():
        for f in user_dir.glob("*"):
            f.unlink()
        user_dir.rmdir()
    return {"status": "cleaned", "user_id": user_id, "session_id": session_id}


@app.post("/admin/export")
async def export_telemetry():
    q_path, i_path = telemetry.export_to_csv()
    return {"queries_csv": q_path, "ingestions_csv": i_path}


@app.get("/health")
async def health():
    return {"status": "ok"}