"""Thin, agent-facing wrapper over vector_store.search with citation formatting."""
from src.rag import vector_store

# Free-tier LLM APIs enforce a tokens-per-minute cap. A single large table
# chunk can blow past that on its own, so every retrieved chunk is capped
# here at retrieval time — no re-ingestion needed, it just bounds what
# gets stuffed into the prompt.
MAX_CHARS_PER_CHUNK = 500
MAX_TOTAL_CONTEXT_CHARS = 2500


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " …[truncated]"


def retrieve(
    query: str,
    *,
    user_id: str | None,
    session_id: str | None,
    top_k: int = 4,
    include_shared: bool = True,
) -> str:
    """
    Returns retrieved chunks formatted as a citation-ready context block.
    Each chunk is tagged with its origin so the model can cite "[doc:page]"
    and the agent can distinguish the shared corpus from this user's own
    upload. Both per-chunk and total context length are capped to stay
    under free-tier token limits.
    """
    hits = vector_store.search(
        query,
        user_id=user_id,
        session_id=session_id,
        top_k=top_k,
        include_shared=include_shared,
    )
    if not hits:
        return "No relevant context found."

    blocks, total_chars = [], 0
    for h in hits:
        tag = f"[{h['doc_id']} p.{h['page']} · {h['source_type']} · {h['source']}]"
        chunk_text = _truncate(h["text"], MAX_CHARS_PER_CHUNK)
        block = f"{tag}\n{chunk_text}"
        if total_chars + len(block) > MAX_TOTAL_CONTEXT_CHARS:
            break
        blocks.append(block)
        total_chars += len(block)
    return "\n\n".join(blocks)
