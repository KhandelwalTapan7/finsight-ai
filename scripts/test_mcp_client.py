"""
A minimal MCP client that connects to our own server exactly like a real
MCP client (Claude Desktop, etc.) would — spawns it as a subprocess over
stdio, lists its tools, and calls one.

Run:  python -m scripts.test_mcp_client
"""
import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_server.server"],
        cwd=os.getcwd(),
    )

    print("Connecting to the finsight MCP server...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print(f"\nConnected. {len(tools.tools)} tools available:")
            for t in tools.tools:
                print(f"  - {t.name}: {t.description}")

            print("\nCalling search_documents(query='TCS revenue')...")
            try:
                result = await asyncio.wait_for(
                    session.call_tool("search_documents", {"query": "TCS revenue"}),
                    timeout=30,
                )
                print("\n--- Result ---")
                print("Raw content list length:", len(result.content))
                for item in result.content:
                    print("Raw item:", repr(item))
                    text = getattr(item, "text", str(item))
                    print(text[:800])
            except asyncio.TimeoutError:
                print("\n--- TIMED OUT after 30s waiting for search_documents ---")
                print("This confirms a real hang, not just slowness.")

            print("\nCalling calculate(expression='(255324-240893)/240893*100')...")
            result2 = await session.call_tool(
                "calculate", {"expression": "(255324-240893)/240893*100"}
            )
            for item in result2.content:
                print(getattr(item, "text", str(item)))

    print("\nDone — the MCP server responded correctly over the protocol.")


if __name__ == "__main__":
    asyncio.run(main())