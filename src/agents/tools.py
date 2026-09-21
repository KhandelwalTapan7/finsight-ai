"""
Tool functions used by the LangGraph agents. These are plain Python
functions on purpose — src/mcp_server/server.py wraps the exact same
functions as MCP tools, so the logic only lives in one place and both the
in-app agent and any external MCP client (e.g. Claude Desktop) get
identical behavior.
"""
import ast
import operator

from src.rag.retriever import retrieve
from src.rag import vector_store


def search_documents(
    query: str,
    user_id: str | None = None,
    session_id: str | None = None,
    include_shared: bool = True,
    top_k: int = 4,
) -> str:
    """Semantic search over this user's uploaded documents, plus the shared corpus if enabled."""
    return retrieve(
        query, user_id=user_id, session_id=session_id,
        top_k=top_k, include_shared=include_shared,
    )


def extract_chart_data(
    query: str,
    user_id: str | None = None,
    session_id: str | None = None,
    include_shared: bool = True,
) -> str:
    """Search within vision-derived descriptions of charts, diagrams and figures."""
    hits = vector_store.search(
        query, user_id=user_id, session_id=session_id, top_k=6, include_shared=include_shared,
    )
    chart_hits = [h for h in hits if h["source_type"] == "chart"]
    if not chart_hits:
        return "No chart, diagram or figure data found matching that query."
    return "\n\n".join(f"[{h['doc_id']} p.{h['page']}]\n{h['text'][:500]}" for h in chart_hits)


# --- Safe calculator: no eval(), only arithmetic AST nodes allowed ---
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Disallowed expression")


def calculate(expression: str) -> str:
    """Evaluate a plain arithmetic expression, e.g. '(1250 - 980) / 980 * 100'."""
    try:
        tree = ast.parse(expression, mode="eval").body
        return str(_safe_eval(tree))
    except Exception as e:
        return f"Could not evaluate expression: {e}"


TOOL_REGISTRY = {
    "search_documents": search_documents,
    "extract_chart_data": extract_chart_data,
    "calculate": calculate,
}
