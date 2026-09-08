"""
Logs every query and ingestion event so the ops dashboard (Streamlit) and
the Power BI dashboard (via export_to_csv) have real usage data to show,
not a mocked demo. SQLite locally; the AWS deployment path swaps this for
DynamoDB behind the same function signatures (see infra/aws/README.md).
"""
import sqlite3
import time
import uuid
from contextlib import contextmanager

from src.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id TEXT PRIMARY KEY,
    ts REAL,
    user_id TEXT,
    session_id TEXT,
    query TEXT,
    answer TEXT,
    latency_ms REAL,
    tool_calls INTEGER,
    ragas_faithfulness REAL,
    ragas_relevance REAL
);

CREATE TABLE IF NOT EXISTS ingestions (
    id TEXT PRIMARY KEY,
    ts REAL,
    user_id TEXT,
    session_id TEXT,
    doc_id TEXT,
    filename TEXT,
    status TEXT,
    chunk_count INTEGER
);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(settings.TELEMETRY_DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.executescript(SCHEMA)


def log_query(
    *, user_id, session_id, query, answer, latency_ms, tool_calls,
    ragas_faithfulness=None, ragas_relevance=None,
) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO queries VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                str(uuid.uuid4()), time.time(), user_id, session_id, query,
                answer, latency_ms, tool_calls, ragas_faithfulness, ragas_relevance,
            ),
        )


def log_ingestion(*, user_id, session_id, doc_id, filename, status, chunk_count=0) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO ingestions VALUES (?,?,?,?,?,?,?,?)",
            (
                str(uuid.uuid4()), time.time(), user_id, session_id, doc_id,
                filename, status, chunk_count,
            ),
        )


def export_to_csv(out_dir: str = "./data/exports") -> tuple[str, str]:
    """Dumps both tables to CSV for Power BI Desktop / Tableau to import."""
    import csv
    import pathlib

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    q_path = f"{out_dir}/queries.csv"
    i_path = f"{out_dir}/ingestions.csv"

    with _conn() as conn:
        for table, path in [("queries", q_path), ("ingestions", i_path)]:
            cur = conn.execute(f"SELECT * FROM {table}")
            cols = [d[0] for d in cur.description]
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(cols)
                writer.writerows(cur.fetchall())

    return q_path, i_path


def get_session_stats(user_id: str, session_id: str) -> dict:
    with _conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*), AVG(latency_ms), AVG(tool_calls) FROM queries "
            "WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        ).fetchone()
        doc_row = conn.execute(
            "SELECT COUNT(*) FROM ingestions WHERE user_id = ? AND session_id = ? AND status = 'ready'",
            (user_id, session_id),
        ).fetchone()
    count, avg_latency, avg_tools = row
    return {
        "query_count": count or 0,
        "avg_latency_ms": round(avg_latency, 1) if avg_latency else 0,
        "avg_tool_calls": round(avg_tools, 1) if avg_tools else 0,
        "documents_uploaded": doc_row[0] or 0,
    }


init_db()
