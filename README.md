# FinSight AI — multimodal agentic RAG over financial reports

A research assistant over financial reports (annual reports, investor
decks) that understands **text, tables, and charts**, answers through a
**multi-tool LangGraph agent**, exposes its capabilities over **MCP** so
other MCP clients can use them, evaluates and monitors itself with an
**MLOps loop** (MLflow + RAGAS + CI), supports **users uploading their own
documents** with proper isolation, and ships a **usage/quality dashboard**
for Power BI. Runs entirely free.

## Why this project

Most portfolio RAG projects are "chat with a PDF" with no eval, no tests,
and a static pre-loaded corpus. This one is built to answer the three
questions that actually separate a demo from a production-minded project:
*does it stay correct over time* (RAGAS + CI), *is one user's data safe
from another's* (per-session vector isolation), and *can anyone see how
it's doing* (the BI dashboard). See `infra/aws/README.md` for the
deployment story.

## Architecture

```
Chat UI (Chainlit)
      │
      ▼
Agent orchestrator (LangGraph, tool-calling: search / chart-search / calc)
      │                                    │
      ▼                                    ▼
Multimodal RAG (Qdrant + sentence-       MCP server (same tools, exposed
transformers + Groq vision for charts)   to Claude Desktop / other clients)
      │
      ▼
AWS deployment (local by default; see infra/aws/README.md for the
S3 / Lambda / DynamoDB / Cognito production path)
      │
      ├──► MLOps loop: MLflow experiment tracking + RAGAS eval, run in CI
      └──► Telemetry → CSV export → Power BI dashboard
```

## What's genuinely multimodal, agentic, etc. — and why

- **Multimodal**: chart/graph-heavy pages are rendered to an image and
  described by a free-tier vision LLM (Groq), then embedded as text
  alongside prose and tables — see `src/ingestion/parser.py`.
- **Agentic**: a LangGraph ReAct agent decides which tool to call
  (`search_documents`, `extract_chart_data`, `calculate`) and can chain
  multiple calls before answering — see `src/agents/graph.py`.
- **RAG with real isolation**: every vector is tagged `source`, `user_id`,
  `session_id`; retrieval always filters on these, so an uploaded document
  is never visible to another user — see `src/rag/vector_store.py`.
- **MCP**: the exact same tool functions are exposed as an MCP server, so
  Claude Desktop (or anything else speaking MCP) can use FinSight's
  retrieval — see `src/mcp_server/server.py`.
- **MLOps**: RAGAS scores (faithfulness, answer relevancy) are logged to
  MLflow on every CI run, not just eyeballed once — see
  `src/eval/ragas_eval.py` and `.github/workflows/ci.yml`.
- **BI dashboard**: every query and ingestion event is logged; export to
  CSV and open in Power BI Desktop (free) for a usage/quality dashboard —
  see `dashboard/streamlit_ops.py` for the quick in-app version.

## Quickstart (100% free, no AWS account needed)

```bash
git clone <your-repo-url> && cd finsight-ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste a free Groq key from https://console.groq.com

# terminal 1
uvicorn src.api.main:app --reload

# terminal 2
chainlit run ui/chainlit_app.py -w

# terminal 3 (optional — usage dashboard)
streamlit run dashboard/streamlit_ops.py
```

Or with Docker: `docker compose up --build`.

### Loading the shared corpus

Drop a few real annual-report PDFs into `data/shared_corpus/` and run:

```bash
python -m scripts.ingest_shared_corpus   # see "what's next" below
```

## Running the eval

```bash
python -m src.eval.ragas_eval
mlflow ui   # view the score trend at http://localhost:5000
```

## Running tests

```bash
pytest -m "not slow"   # fast, no model download
pytest                 # full suite (downloads the embedding model)
```

## What's next (honest roadmap, not finished-and-perfect)

- Split the single ReAct agent into explicit sub-agent graphs (retrieval
  specialist, numeric-reasoning specialist, writer) behind a supervisor
  node — the current single-agent version is simpler to debug and a
  reasonable place to start; multi-graph is the natural v2.
  Fill in `src/eval/testset.json` with real question/answer pairs once
  you've loaded actual filings — the placeholders there are structural,
  not real ground truth.
- AWS deployment per `infra/aws/README.md`.

## Cost

Everything above runs for $0: Groq's free tier for LLM calls, local
embedded Qdrant, local SQLite, GitHub Actions' free minutes for CI. The
only signup required is a free Groq API key (no credit card).
