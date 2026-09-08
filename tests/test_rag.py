import uuid

import pytest

from src.ingestion.chunker import Chunk, chunk_raw
from src.ingestion.parser import RawChunk
from src.agents.tools import calculate


def test_chunker_splits_long_text():
    long_text = "word " * 500  # well over CHUNK_SIZE
    raw = [RawChunk(doc_id="d1", page=1, source_type="text", content=long_text)]
    chunks = chunk_raw(raw)
    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.doc_id == "d1" for c in chunks)


def test_chunker_keeps_tables_intact():
    table_md = "| a | b |\n|---|---|\n| 1 | 2 |\n"
    raw = [RawChunk(doc_id="d1", page=2, source_type="table", content=table_md)]
    chunks = chunk_raw(raw)
    assert len(chunks) == 1
    assert chunks[0].text == table_md


def test_calculate_basic_arithmetic():
    assert calculate("(1250 - 980) / 980 * 100") == str((1250 - 980) / 980 * 100)


def test_calculate_blocks_unsafe_expressions():
    result = calculate("__import__('os').system('echo hi')")
    assert "Could not evaluate" in result


@pytest.mark.slow
def test_vector_store_roundtrip(tmp_path, monkeypatch):
    """Requires downloading the embedding model — runs in CI, skip locally with -m 'not slow'."""
    monkeypatch.setenv("LOCAL_QDRANT_PATH", str(tmp_path / "qdrant"))
    from src.rag import vector_store  # imported here so the monkeypatched path is picked up

    chunk = Chunk(id=str(uuid.uuid4()), doc_id="d1", page=1, source_type="text", text="Revenue grew 12% year over year.")
    vector_store.upsert_chunks([chunk], source="shared")
    hits = vector_store.search("revenue growth", user_id=None, session_id=None)
    assert any("Revenue" in h["text"] for h in hits)
