"""
Exposes FinSight's retrieval and calculation tools over MCP (Model Context
Protocol) so any MCP-compatible client — Claude Desktop, another agent,
your own future app — can call them the standard way, without knowing
anything about this project's internals.

Run standalone for local testing:
    python -m src.mcp_server.server

Note: MCP tool calls here run in "shared corpus only" mode (no user_id/
session_id) since an external MCP client isn't authenticated into a
FinSight session — it can search the shared financial corpus, not a
particular user's private uploads. That boundary is enforced in
tools.py's default args, not just left implicit.
"""
import sys

if sys.platform == "win32":
    # Windows defaults stdio to the legacy cp1252 codepage, which can't
    # encode some characters that show up in real document text (stray
    # glyphs from OCR/PDF extraction, etc.) — this crashed the MCP
    # server's stdout writer mid-response, which looked like an
    # indefinite hang to the client waiting for a reply that could
    # never arrive. Forcing UTF-8 here fixes it at the source.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from mcp.server.fastmcp import FastMCP

from src.agents import tools as tool_impls

mcp = FastMCP("finsight-financial-research")


@mcp.tool()
def search_documents(query: str) -> str:
    """Semantic search over FinSight's shared financial document corpus."""
    return tool_impls.search_documents(query, user_id=None, session_id=None)


@mcp.tool()
def extract_chart_data(query: str) -> str:
    """Search specifically within chart/graph descriptions in the shared corpus."""
    return tool_impls.extract_chart_data(query, user_id=None, session_id=None)


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. a growth or margin calculation."""
    return tool_impls.calculate(expression)


if __name__ == "__main__":
    mcp.run(transport="stdio")