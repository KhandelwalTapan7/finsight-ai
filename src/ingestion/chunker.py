"""
Splits RawChunks into embedding-sized pieces. Tables and chart descriptions
are kept whole (splitting a table mid-row destroys its meaning); only long
prose "text" chunks get split further.
"""
from dataclasses import dataclass, field
import uuid

from src.ingestion.parser import RawChunk

CHUNK_SIZE = 800       # characters
CHUNK_OVERLAP = 120
MAX_TABLE_CHARS = 1200  # large tables get truncated, not split mid-row


@dataclass
class Chunk:
    id: str
    doc_id: str
    page: int
    source_type: str
    text: str
    meta: dict = field(default_factory=dict)


def _split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= size:
        return [text]
    pieces, start = [], 0
    while start < len(text):
        end = start + size
        pieces.append(text[start:end])
        start = end - overlap
    return pieces


def chunk_raw(raw_chunks: list[RawChunk]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for raw in raw_chunks:
        if raw.source_type == "text":
            pieces = _split_text(raw.content)
        elif raw.source_type == "table" and len(raw.content) > MAX_TABLE_CHARS:
            # Splitting a table mid-row destroys it, but an unbounded table
            # can also blow a free-tier LLM's per-request token budget on
            # its own — truncate rather than split.
            pieces = [raw.content[:MAX_TABLE_CHARS] + "\n…[table truncated]"]
        else:
            # short tables / chart descriptions stay intact
            pieces = [raw.content]

        for piece in pieces:
            if not piece.strip():
                continue
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    doc_id=raw.doc_id,
                    page=raw.page,
                    source_type=raw.source_type,
                    text=piece.strip(),
                    meta=raw.meta,
                )
            )
    return chunks
