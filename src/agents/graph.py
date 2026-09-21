"""
Agent orchestration layer.

Domain-agnostic: the system prompt is built from config, so the same
graph serves finance, legal, research or any other document type.

Uses LangGraph's prebuilt ReAct agent as the supervisor: given a question,
the LLM decides which tool(s) to call (search_documents, extract_chart_data,
calculate), observes the results, and can call more tools before writing
a final, cited answer. Tools are rebuilt per request so user_id/session_id
are captured in a closure — that's what keeps one user's uploaded
documents invisible to another user's queries.

Roadmap note (see README "what's next"): splitting this into separate
sub-agent graphs (a retrieval specialist, a numeric-reasoning specialist,
a writer) behind an explicit supervisor node is the natural next
iteration once the single-agent version is solid — starting simple and
correct beats a fragile multi-graph setup that's hard to debug.
"""
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from src.config import settings
from src.agents import tools as tool_impls


def _system_prompt() -> str:
    """Domain-neutral by default; DOMAIN_HINT specialises it without a code change."""
    scope = (
        f"The documents you work with are typically {settings.DOMAIN_HINT}."
        if settings.DOMAIN_HINT
        else "Documents can be about anything — contracts, research papers, "
             "manuals, reports, policies, invoices, notes."
    )
    return f"""You are {settings.APP_NAME}, a research assistant that answers questions strictly from the documents the user has uploaded.

{scope} Never assume a subject area the documents do not support, and never fall back on general world knowledge when asked about "this document".

Rules:
- Always ground answers in retrieved context — call search_documents (and
  extract_chart_data for anything about a chart, graph, diagram or figure)
  before answering ANY question about a document's content, including
  summaries, overviews, or vague references like "this document," "it," or
  "the file." Never ask the user to re-share something they already
  uploaded — search for it instead.
- Be economical with tool calls: call search_documents at most 3 times for
  the entire response, no matter how many topics or entities are involved.
  Construct one broad query covering them together rather than a separate
  narrow query per entity — it is a similarity search over the whole
  corpus, so one good query usually retrieves across every indexed
  document at once.
- Use the calculate tool for any arithmetic instead of doing it in your head.
- Cite sources inline like [doc_id p.page] for every factual claim.
- If the retrieved context doesn't support an answer, say so plainly
  instead of guessing.
- Match the register of the source material: summarise a legal document in
  legal terms, a scientific paper in scientific terms, and so on.
"""


def _build_tools(
    user_id: str | None, session_id: str | None, *, include_shared: bool = True, top_k: int = 4
):
    @tool
    def search_documents(query: str) -> str:
        """Semantic search over this user's uploaded documents (and the shared corpus, if enabled)."""
        return tool_impls.search_documents(
            query, user_id=user_id, session_id=session_id,
            include_shared=include_shared, top_k=top_k,
        )

    @tool
    def extract_chart_data(query: str) -> str:
        """Search within descriptions of charts, diagrams and figures rather than plain text."""
        return tool_impls.extract_chart_data(
            query, user_id=user_id, session_id=session_id, include_shared=include_shared,
        )

    @tool
    def calculate(expression: str) -> str:
        """Evaluate an arithmetic expression."""
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
        timeout=90,       # generous read timeout — free-tier API calls can be slow under load
        max_retries=3,    # transient network blips shouldn't fail the whole request
        max_tokens=1500,  # gpt-oss models spend some of this budget on internal
                          # reasoning before writing the visible answer — 800 was too
                          # tight and could produce an empty response if reasoning
                          # consumed the whole budget before any answer was written
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
) -> dict:
    """
    Returns {"answer": str, "steps": [...]}. `steps` lists each tool call
    made along the way — the Chainlit UI renders these so users can see
    the agent's reasoning transparently instead of a black box.
    """
    agent = build_agent(user_id, session_id, include_shared=include_shared, top_k=top_k)
    messages = [("system", _system_prompt()), ("user", query)]
    steps = []
    final_answer = ""

    try:
        # Hard cap on agent steps: without this, nothing stops a confused
        # agent from looping through extra tool calls it doesn't need,
        # silently burning through a scarce daily token quota.
        result = agent.invoke({"messages": messages}, config={"recursion_limit": 20})
        messages_out = result["messages"]
    except GraphRecursionError as e:
        # The agent ran out of steps without reaching a final answer —
        # degrade gracefully with whatever it found instead of crashing
        # into a generic error. e.args[1] holds the partial state in
        # recent langgraph versions; fall back to an empty list if not.
        messages_out = getattr(e, "state", {}).get("messages", []) if hasattr(e, "state") else []

    for msg in messages_out:
        msg_type = msg.__class__.__name__
        if msg_type == "AIMessage" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                steps.append({"tool": tc["name"], "input": tc["args"], "tool_call_id": tc.get("id")})
        elif msg_type == "ToolMessage":
            # Attach this result to the step that made the matching call,
            # instead of appending a separate, unlinked entry — otherwise
            # the UI has no way to show a tool call next to what it returned.
            result_text = str(msg.content)[:500]
            call_id = getattr(msg, "tool_call_id", None)
            matched = next(
                (s for s in steps if s.get("tool_call_id") == call_id and "output" not in s),
                None,
            ) if call_id else None
            if matched is not None:
                matched["output"] = result_text
            else:
                steps.append({"tool_result": result_text})
        elif msg_type == "AIMessage" and msg.content:
            final_answer = msg.content

    if not final_answer:
        tool_results = [s.get("output") or s.get("tool_result") for s in steps if s.get("output") or s.get("tool_result")]
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
                "has actually been uploaded."
            )

    return {"answer": final_answer, "steps": steps}
