from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from src.config import settings
from src.agents import tools as tool_impls

SYSTEM_PROMPT = """You are FinSight, a financial-document research assistant.

Rules:
- Always ground answers in retrieved context — call search_documents (and
  extract_chart_data for anything about a chart, graph, or trend) before
  answering ANY question about a document's content, including
  summaries, overviews, or vague references like "this document," "it,"
  or "the file." Never ask the user to re-share something they already
  uploaded — search for it instead.
- Be economical with tool calls: call search_documents at most 3 times
  total for the entire response, no matter how many companies or topics
  are involved — construct one broad query covering multiple entities
  together (e.g. "TCS Infosys HDFC Bank revenue growth") rather than a
  separate narrow query per entity. One well-constructed search usually
  retrieves relevant chunks across every indexed company at once, since
  it's a similarity search over the whole corpus.
- Use the calculate tool for any arithmetic instead of doing it in your head.
- Cite sources inline like [doc_id p.page] for every factual claim.
- If the retrieved context doesn't support an answer, say so plainly
  instead of guessing.
- When the user's own uploaded document and the shared corpus disagree or
  can be compared, point that out explicitly.
- Prior turns from this conversation may be included above the current
  question — use them for context (e.g. "this document" may refer to
  something discussed earlier), but still call search_documents again if
  you need to re-confirm or find supporting facts for a new question.
"""


def _build_tools(
    user_id: str | None, session_id: str | None, *, include_shared: bool = True, top_k: int = 4
):
    @tool
    def search_documents(query: str = "") -> str:
        """Semantic search over the shared corpus and this user's uploaded documents.
        For a vague or general request (e.g. "summarize this document"), still call
        this with your best guess at a search phrase, or leave it blank to fall back
        to a general overview search."""
        q = query.strip() or "document summary overview key points main topics"
        return tool_impls.search_documents(
            q, user_id=user_id, session_id=session_id,
            include_shared=include_shared, top_k=top_k,
        )

    @tool
    def extract_chart_data(query: str = "") -> str:
        """Search specifically within chart/graph descriptions rather than plain text."""
        q = query.strip() or "chart graph trend"
        return tool_impls.extract_chart_data(
            q, user_id=user_id, session_id=session_id, include_shared=include_shared,
        )

    @tool
    def calculate(expression: str) -> str:
        """Evaluate an arithmetic expression, e.g. growth or margin calculations."""
        return tool_impls.calculate(expression)

    return [search_documents, extract_chart_data, calculate]


def build_agent(
    user_id: str | None = None,
    session_id: str | None = None,
    *,
    include_shared: bool = True,
    top_k: int = 4,
):
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_TEXT_MODEL,
        temperature=0.1,
        timeout=90,
        max_retries=3,
        max_tokens=1500,
    )
    tools = _build_tools(user_id, session_id, include_shared=include_shared, top_k=top_k)
    return create_react_agent(llm, tools=tools)


def run_agent(
    query: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    include_shared: bool = True,
    top_k: int = 4,
    history: list[dict] | None = None,
) -> dict:
    """
    Returns {"answer": str, "steps": [...]}. `steps` lists each tool call
    made along the way — the Chainlit UI renders these so users can see
    the agent's reasoning transparently instead of a black box.

    `history` is prior user/assistant turns from the same chat (already
    bounded to the last few exchanges by the caller) — only final answers
    are kept, not intermediate tool calls/results, so memory stays compact
    instead of re-sending every retrieved chunk from earlier turns.
    """
    agent = build_agent(user_id, session_id, include_shared=include_shared, top_k=top_k)
    history_messages = [(h["role"], h["content"]) for h in (history or [])]
    messages = [("system", SYSTEM_PROMPT), *history_messages, ("user", query)]
    steps = []
    final_answer = ""

    messages_out = []
    try:
        for state in agent.stream(
            {"messages": messages}, config={"recursion_limit": 20}, stream_mode="values"
        ):
            messages_out = state["messages"]
    except GraphRecursionError:
        pass
    except Exception:
        pass

    for msg in messages_out:
        msg_type = msg.__class__.__name__
        if msg_type == "AIMessage" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                steps.append({"tool": tc["name"], "input": tc["args"]})
        elif msg_type == "ToolMessage":
            steps.append({"tool_result": str(msg.content)[:500]})
        elif msg_type == "AIMessage" and msg.content:
            final_answer = msg.content

    if not final_answer:
        tool_results = [s["tool_result"] for s in steps if "tool_result" in s]
        if tool_results:
            final_answer = (
                "I found some relevant context but couldn't finish forming a complete "
                "answer in time. Here's what I found — try rephrasing your question to "
                "be more specific:\n\n" + "\n\n".join(tool_results[-2:])
            )
        else:
            final_answer = (
                "I wasn't able to find relevant information for that question. Try "
                "rephrasing it, or check that the document/company you're asking about "
                "has actually been ingested."
            )

    return {"answer": final_answer, "steps": steps}