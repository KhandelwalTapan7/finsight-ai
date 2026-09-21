"""
Chainlit chat UI — settings panel, action buttons (clear/list documents,
session stats), and sources rendered as proper side-panel citations
instead of buried inline brackets.

Run:  chainlit run ui/chainlit_app.py -w
"""
import asyncio
import re
import uuid

import os

import chainlit as cl
import httpx

# Deliberately no "from src.config import settings" here: chainlit's own
# module loader (both standalone `chainlit run` and when mounted via
# mount_chainlit) does not add the repo root to sys.path the way running
# uvicorn from the root does, so that import fails with
# "ModuleNotFoundError: No module named 'src'". Read the same env vars
# directly instead — it's just two display strings.
APP_NAME = os.getenv("APP_NAME", "DocSight AI")
SESSION_TTL_HOURS = os.getenv("SESSION_TTL_HOURS", "24")

# Chainlit is mounted onto the same FastAPI app in production, so it talks
# to the API over loopback on whatever port Render assigned.
API_URL = os.getenv("API_URL", f"http://127.0.0.1:{os.getenv('PORT', '8000')}")

_CITATION_RE = re.compile(r"\[([\w\-\.]+)\s+p\.(\d+)\s*·\s*(\w+)\s*·\s*(\w+)\]\n(.+?)(?=\n\[|\Z)", re.DOTALL)


@cl.set_starters
async def starters():
    return [
        cl.Starter(
            label="Summarize my document",
            message="Summarize the document I uploaded — main points, structure, and anything notable.",
        ),
        cl.Starter(
            label="Explain a figure",
            message="Explain the charts, diagrams or figures in this document and what they show.",
        ),
        cl.Starter(
            label="Find something specific",
            message="Find every mention of deadlines, obligations or numbers in this document and cite the page.",
        ),
    ]


@cl.set_chat_profiles
async def chat_profiles():
    return [
        cl.ChatProfile(
            name="Standard",
            markdown_description="Searches your uploads **and** the shared corpus (if one is configured).",
            default=True,
        ),
        cl.ChatProfile(
            name="My Documents Only",
            markdown_description="Searches **only** documents you've uploaded this session — the shared corpus is excluded.",
        ),
    ]


@cl.on_chat_start
async def start():
    cl.user_session.set("session_id", str(uuid.uuid4())[:8])
    # One isolated user_id per browser session. Never a fixed "demo-user":
    # on a public deployment that would put every visitor's uploads into
    # the same namespace, so anyone could query anyone else's documents.
    cl.user_session.set("user_id", str(uuid.uuid4()))
    cl.user_session.set("documents", [])  # [{filename, doc_id, chunk_count}]

    profile = cl.user_session.get("chat_profile")
    cl.user_session.set("include_shared", profile != "My Documents Only")

    settings = await cl.ChatSettings(
        [
            cl.input_widget.Switch(
                id="include_shared",
                label="Include shared corpus",
                initial=cl.user_session.get("include_shared"),
            ),
            cl.input_widget.Slider(
                id="top_k",
                label="Number of sources to retrieve",
                initial=4,
                min=2,
                max=8,
                step=1,
            ),
        ]
    ).send()
    cl.user_session.set("top_k", settings["top_k"])

    await cl.Message(
        content=(
            f"**Welcome to {APP_NAME}** — drop any PDF into the chat "
            "and ask questions about it. Contracts, research papers, manuals, "
            "reports, whatever you have. Your upload is private to this "
            f"session and auto-deleted after {SESSION_TTL_HOURS} hours.\n\n"
            "Use the ⚙️ settings icon to control retrieval, and the actions "
            "below for quick tools."
        ),
        actions=[
            cl.Action(name="list_documents", value="click", label="📄 My documents"),
            cl.Action(name="session_stats", value="click", label="📊 My session stats"),
            cl.Action(name="clear_documents", value="click", label="🗑️ Clear my documents"),
        ],
    ).send()


@cl.on_settings_update
async def on_settings_update(settings):
    cl.user_session.set("include_shared", settings["include_shared"])
    cl.user_session.set("top_k", settings["top_k"])


@cl.action_callback("list_documents")
async def list_documents(action: cl.Action):
    docs = cl.user_session.get("documents") or []
    if not docs:
        await cl.Message(content="You haven't uploaded any documents this session yet.").send()
        return
    lines = [f"- `{d['filename']}` — {d['chunk_count']} chunks indexed" for d in docs]
    await cl.Message(content="**Your uploaded documents:**\n" + "\n".join(lines)).send()


@cl.action_callback("session_stats")
async def session_stats(action: cl.Action):
    user_id = cl.user_session.get("user_id")
    session_id = cl.user_session.get("session_id")
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{API_URL}/admin/session-stats/{user_id}/{session_id}")
        resp.raise_for_status()
        stats = resp.json()
    await cl.Message(
        content=(
            f"**Session stats**\n"
            f"- Queries asked: {stats['query_count']}\n"
            f"- Avg latency: {stats['avg_latency_ms']:.0f} ms\n"
            f"- Avg tool calls per query: {stats['avg_tool_calls']}\n"
            f"- Documents uploaded: {stats['documents_uploaded']}"
        )
    ).send()


@cl.action_callback("clear_documents")
async def clear_documents(action: cl.Action):
    user_id = cl.user_session.get("user_id")
    session_id = cl.user_session.get("session_id")
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{API_URL}/admin/cleanup/{user_id}/{session_id}")
        resp.raise_for_status()
    cl.user_session.set("documents", [])
    await cl.Message(content="Cleared — your uploaded documents and vectors have been deleted from this session.").send()


async def _upload_and_wait(file_element: cl.File, user_id: str, session_id: str) -> None:
    async with httpx.AsyncClient(timeout=60) as client:
        with open(file_element.path, "rb") as f:
            resp = await client.post(
                f"{API_URL}/upload",
                files={"file": (file_element.name, f, "application/pdf")},
                params={"user_id": user_id, "session_id": session_id},
            )
        resp.raise_for_status()
        doc_id = resp.json()["doc_id"]

        status_msg = cl.Message(content=f"Processing `{file_element.name}`…")
        await status_msg.send()

        for _ in range(60):
            status = (await client.get(f"{API_URL}/ingest/status/{doc_id}")).json()
            if status["status"] == "ready":
                status_msg.content = (
                    f"`{file_element.name}` ready — indexed {status['chunk_count']} chunks. "
                    "You can now ask questions about it."
                )
                await status_msg.update()
                docs = cl.user_session.get("documents") or []
                docs.append({
                    "filename": file_element.name,
                    "doc_id": doc_id,
                    "chunk_count": status["chunk_count"],
                })
                cl.user_session.set("documents", docs)
                return
            if status["status"] == "failed":
                status_msg.content = f"Failed to process `{file_element.name}`: {status.get('error')}"
                await status_msg.update()
                return
            await asyncio.sleep(1)

        status_msg.content = f"Still processing `{file_element.name}` — try asking your question shortly."
        await status_msg.update()


def _parse_sources(steps: list[dict]) -> list[cl.Text]:
    """Turns retrieved tool_result strings (already tagged '[doc_id p.page · type · source]')
    into individual Chainlit Text elements, so citations show as an expandable, clickable
    side panel instead of a wall of text buried in the answer."""
    elements = []
    seen = set()
    for step in steps:
        raw = step.get("output") or step.get("tool_result", "")
        for match in _CITATION_RE.finditer(raw + "\n["):  # sentinel to close the last match
            doc_id, page, source_type, source, text = match.groups()
            key = (doc_id, page, source_type)
            if key in seen:
                continue
            seen.add(key)
            tag = "📤 your upload" if source == "user" else "📚 shared corpus"
            elements.append(
                cl.Text(
                    name=f"{doc_id} · p.{page} ({source_type})",
                    content=f"{tag}\n\n{text.strip()}",
                    display="side",
                )
            )
    return elements


@cl.on_message
async def main(message: cl.Message):
    user_id = cl.user_session.get("user_id")
    session_id = cl.user_session.get("session_id")

    pdf_files = [e for e in message.elements if getattr(e, "name", "").lower().endswith(".pdf")]
    for f in pdf_files:
        await _upload_and_wait(f, user_id, session_id)

    if not message.content.strip():
        return

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{API_URL}/chat",
            json={
                "query": message.content,
                "user_id": user_id,
                "session_id": session_id,
                "include_shared": cl.user_session.get("include_shared"),
                "top_k": cl.user_session.get("top_k") or 4,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    for step in data["steps"]:
        if "tool" in step:
            async with cl.Step(name=step["tool"], type="tool") as s:
                s.input = step.get("input", {})
                s.output = step.get("output", "")

    elements = _parse_sources(data["steps"])
    await cl.Message(
        content=data["answer"],
        elements=elements,
        actions=[
            cl.Action(name="session_stats", value="click", label="📊 My session stats"),
        ],
    ).send()
