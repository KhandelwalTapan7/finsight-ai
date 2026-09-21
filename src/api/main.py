"""
FastAPI backend.

Local mode (STORAGE_MODE=local, the default): uploads are written straight
to disk and ingested in a FastAPI BackgroundTask — no AWS account needed,
runs entirely free on your laptop.

AWS mode (STORAGE_MODE=aws): swap in presigned S3 uploads + a Lambda
triggered by the S3 event instead of the BackgroundTask. The ingestion
logic itself (parser -> chunker -> embedder -> vector_store) doesn't
change at all — only how the file arrives and who triggers ingestion does.
See infra/aws/README.md for the exact swap.
"""
import os
import re
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

app = FastAPI(title=settings.APP_NAME, version="0.2.0")
# Tighten this before going public: "*" lets any site on the internet call
# your API from a user's browser. Set ALLOWED_ORIGINS to your Render URL.
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware, allow_origins=_origins, allow_methods=["*"], allow_headers=["*"]
)

# In-memory status board for local mode. In AWS mode this is a DynamoDB
# table instead (see infra/aws/README.md) so status survives across
# Lambda invocations / multiple instances.
_ingestion_status: dict[str, IngestStatus] = {}

# Allowed shape for client-supplied identifiers used in filesystem paths.
_SAFE_ID = re.compile(r"[A-Za-z0-9_\-]{1,64}")

# Exact-match query cache: repeat questions (extremely common while
# testing/demoing) are answered instantly with zero extra LLM calls
# instead of re-spending scarce free-tier token quota on identical work.
# Keyed by (normalized query, user_id, session_id); TTL keeps stale
# answers from lingering if the underlying documents change.
_query_cache: dict[tuple[str, str, str], tuple[dict, float]] = {}
CACHE_TTL_SECONDS = 3600


def _cache_key(query: str, user_id: str, session_id: str, include_shared: bool, top_k: int) -> tuple:
    return (query.strip().lower(), user_id, session_id, include_shared, top_k)


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

    body = await file.read()
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(body) > max_bytes:
        raise HTTPException(413, f"File too large — the limit is {settings.MAX_UPLOAD_MB} MB.")

    # user_id/session_id arrive from the client, so they must never be
    # trusted as path components — a crafted value like "../../etc" would
    # otherwise write outside the upload directory.
    if not all(_SAFE_ID.fullmatch(v or "") for v in (user_id, session_id)):
        raise HTTPException(400, "Invalid user_id or session_id.")

    doc_id = str(uuid.uuid4())[:8]
    user_dir = settings.LOCAL_UPLOAD_DIR / user_id / session_id
    user_dir.mkdir(parents=True, exist_ok=True)
    dest = user_dir / f"{doc_id}.pdf"
    dest.write_bytes(body)

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

    key = _cache_key(req.query, req.user_id, req.session_id, req.include_shared, req.top_k)
    cached = _query_cache.get(key)
    if cached and (time.time() - cached[1]) < CACHE_TTL_SECONDS:
        result = cached[0]
    else:
        result = run_agent(
            req.query, user_id=req.user_id, session_id=req.session_id,
            include_shared=req.include_shared, top_k=req.top_k,
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
    """
    TTL cleanup: deletes this user's vectors and uploaded files. Call this
    from a cron job locally, or an AWS EventBridge scheduled rule /
    DynamoDB TTL trigger in production (see infra/aws/README.md).
    """
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
    return {"status": "ok", "app": settings.APP_NAME}


# --- Mount the Chainlit UI onto this same FastAPI app -------------------
# One process, one port, one Qdrant connection. Running the API and the UI
# as two separate Render services would mean paying for two instances and,
# in embedded-Qdrant mode, fighting over an exclusive directory lock.
# The chat UI is served at /ui; the API keeps /upload, /chat (POST), etc.
from chainlit.utils import mount_chainlit

mount_chainlit(app=app, target="ui/chainlit_app.py", path="/ui")
