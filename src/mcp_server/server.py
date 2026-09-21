"""
Exposes the app's retrieval and calculation tools over MCP (Model Context
Protocol) so any MCP-compatible client — Claude Desktop, another agent,
your own future app — can call them the standard way, without knowing
anything about this project's internals.

Run standalone for local testing:
    python -m src.mcp_server.server

Wire into Claude Desktop by adding to its config:
    {
      "mcpServers": {
        "docsight": {
          "command": "python",
          "args": ["-m", "src.mcp_server.server"],
          "cwd": "/absolute/path/to/your-repo"
        }
      }
    }

Note: MCP tool calls here run in "shared corpus only" mode (no user_id/
session_id) since an external MCP client isn't authenticated into a
session — it can search the shared document corpus, not a
particular user's private uploads. That boundary is enforced in
tools.py's default args, not just left implicit.
"""
from mcp.server.fastmcp import FastMCP

from src.agents import tools as tool_impls

mcp = FastMCP("docsight-research")


@mcp.tool()
def search_documents(query: str) -> str:
    """Semantic search over the shared document corpus."""
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
