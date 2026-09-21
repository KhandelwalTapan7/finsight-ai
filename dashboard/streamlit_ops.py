"""
Lightweight in-app analytics dashboard. This is the quick, free way to see
usage while developing. For the polished, resume-facing version, export
to CSV (button below) and open it in Power BI Desktop (free) — see
README "BI dashboard" section for the exact steps and suggested visuals.

Run:  streamlit run dashboard/streamlit_ops.py
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit adds this script's own folder (dashboard/) to sys.path, not the
# repo root — so "from src.config import settings" fails with
# "ModuleNotFoundError: No module named 'src'" even when launched from the
# root. Add the repo root explicitly before importing anything under src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.telemetry.logger import export_to_csv

st.set_page_config(page_title="FinSight — Ops dashboard", layout="wide")
st.title("FinSight — usage & quality dashboard")


@st.cache_data(ttl=10)
def load_data():
    conn = sqlite3.connect(settings.TELEMETRY_DB_PATH)
    queries = pd.read_sql("SELECT * FROM queries", conn)
    ingestions = pd.read_sql("SELECT * FROM ingestions", conn)
    conn.close()
    if not queries.empty:
        queries["ts"] = pd.to_datetime(queries["ts"], unit="s")
    if not ingestions.empty:
        ingestions["ts"] = pd.to_datetime(ingestions["ts"], unit="s")
    return queries, ingestions


queries, ingestions = load_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total queries", len(queries))
col2.metric("Documents ingested", len(ingestions))
col3.metric(
    "Avg latency (ms)",
    f"{queries['latency_ms'].mean():.0f}" if not queries.empty else "—",
)
col4.metric(
    "Avg RAGAS faithfulness",
    f"{queries['ragas_faithfulness'].mean():.2f}"
    if not queries.empty and queries["ragas_faithfulness"].notna().any()
    else "—",
)

st.divider()

if not queries.empty:
    left, right = st.columns(2)
    with left:
        st.subheader("Queries over time")
        by_day = queries.set_index("ts").resample("D").size()
        st.line_chart(by_day)
    with right:
        st.subheader("Latency distribution (ms)")
        st.bar_chart(queries["latency_ms"])

    st.subheader("Recent queries")
    st.dataframe(
        queries[["ts", "user_id", "query", "latency_ms", "tool_calls"]]
        .sort_values("ts", ascending=False)
        .head(20),
        use_container_width=True,
    )
else:
    st.info("No queries logged yet — run some chats in the Chainlit app first.")

st.divider()
st.subheader("Export for Power BI / Tableau")
st.caption(
    "Exports both tables to CSV. In Power BI Desktop: Get Data → Text/CSV → "
    "point at these files → build visuals on top (usage trend, latency, "
    "RAGAS score over time, top query topics)."
)
if st.button("Export to CSV"):
    q_path, i_path = export_to_csv()
    st.success(f"Exported to `{q_path}` and `{i_path}`")
