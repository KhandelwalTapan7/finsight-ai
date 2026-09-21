"""
Vector store wrapper around Qdrant.

Runs in embedded/local mode by default (QdrantClient(path=...)) — this
needs no server, no Docker, and no account, so the whole project runs for
free with zero external infra. Swap to Qdrant Cloud's free tier or a
self-hosted instance on AWS by changing QDRANT_URL/QDRANT_API_KEY (see
infra/aws/README.md) without touching any other code.

Isolation model: every point carries `source` ("shared" | "user"),
`user_id`, and `session_id` in its payload. Retrieval always filters on
these so one user's uploads are never visible to another user, and the
shared/pre-loaded corpus is queryable by everyone.
"""
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.config import settings
from src.ingestion.chunker import Chunk
from src.ingestion.embedder import embed_texts, embedding_dim

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """
    Remote Qdrant (Cloud or self-hosted) when QDRANT_URL is set, otherwise
    embedded on-disk mode for local dev.

    On a platform like Render the embedded mode is not an option: the
    filesystem is wiped on every deploy, and embedded Qdrant holds an
    exclusive lock on its directory, so a second process (or a second
    instance after a scale-up) fails to start. Qdrant Cloud has a free
    1 GB tier that removes both problems.
    """
    global _client
    if _client is None:
        if settings.QDRANT_URL:
            _client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY or None,
                timeout=30,
            )
        else:
            _client = QdrantClient(path=str(settings.LOCAL_QDRANT_PATH))
        _ensure_collection(_client)
    return _client


def _ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if settings.COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=qm.VectorParams(
                size=embedding_dim(), distance=qm.Distance.COSINE
            ),
        )


def upsert_chunks(
    chunks: list[Chunk],
    *,
    source: str,          # "shared" | "user"
    user_id: str | None = None,
    session_id: str | None = None,
) -> int:
    if not chunks:
        return 0
    client = get_client()
    vectors = embed_texts([c.text for c in chunks])

    points = []
    for chunk, vector in zip(chunks, vectors):
        payload = {
            "doc_id": chunk.doc_id,
            "page": chunk.page,
            "source_type": chunk.source_type,  # text | table | chart
            "text": chunk.text,
            "source": source,                   # shared | user
            "user_id": user_id,
            "session_id": session_id,
        }
        points.append(qm.PointStruct(id=chunk.id, vector=vector, payload=payload))

    client.upsert(collection_name=settings.COLLECTION_NAME, points=points)
    return len(points)


def search(
    query: str,
    *,
    user_id: str | None,
    session_id: str | None,
    top_k: int = 6,
    include_shared: bool = True,
) -> list[dict]:
    """
    Returns chunks visible to this user: their own uploads (matched on
    user_id + session_id) plus the shared corpus if include_shared=True.

    User uploads and the shared corpus are searched separately, not as
    one blended similarity search. A small personal upload (a handful of
    chunks) would otherwise be drowned out by a much larger shared corpus
    in a single ranked search, even when it's exactly what the user is
    asking about — so the user's own documents always get a guaranteed
    slot instead of competing purely on similarity score.
    """
    client = get_client()
    query_vector = embed_texts([query])[0]

    hits = []

    if user_id:
        user_filter = qm.Filter(
            must=[
                qm.FieldCondition(key="source", match=qm.MatchValue(value="user")),
                qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
                qm.FieldCondition(key="session_id", match=qm.MatchValue(value=session_id)),
            ]
        )
        hits.extend(
            client.search(
                collection_name=settings.COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=user_filter,
                limit=top_k,
            )
        )

    if include_shared and settings.ENABLE_SHARED_CORPUS:
        remaining = max(top_k - len(hits), top_k // 2)
        shared_filter = qm.Filter(
            must=[qm.FieldCondition(key="source", match=qm.MatchValue(value="shared"))]
        )
        hits.extend(
            client.search(
                collection_name=settings.COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=shared_filter,
                limit=remaining,
            )
        )

    return [
        {
            "text": h.payload["text"],
            "doc_id": h.payload["doc_id"],
            "page": h.payload["page"],
            "source_type": h.payload["source_type"],
            "source": h.payload["source"],
            "score": h.score,
        }
        for h in hits
    ]


def delete_user_data(user_id: str, session_id: str) -> None:
    """Called by the TTL cleanup job when a session expires."""
    client = get_client()
    client.delete(
        collection_name=settings.COLLECTION_NAME,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(
                must=[
                    qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
                    qm.FieldCondition(key="session_id", match=qm.MatchValue(value=session_id)),
                ]
            )
        ),
    )


def doc_exists(doc_id: str) -> bool:
    """
    True if this doc_id already has vectors stored. Used to make bulk
    ingestion idempotent — without this, re-running the ingestion script
    silently re-parses and re-embeds everything, which on a free-tier LLM
    API means wastefully burning scarce daily token quota on documents
    that were already indexed correctly the first time.
    """
    client = get_client()
    result = client.count(
        collection_name=settings.COLLECTION_NAME,
        count_filter=qm.Filter(
            must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
        ),
    )
    return result.count > 0
