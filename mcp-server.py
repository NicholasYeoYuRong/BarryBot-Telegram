from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="MCP-server",
    host="0.0.0.0",
    port="8000",
    )

@mcp.tool()
def list_task(max_results: int) -> list[str]:
    """List all tasks"""
    return[
        "East breakfast",
        "Go to the gym",
        "Read a book",
    ][:max_results]

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

if __name__ == "__main__":
    mcp.run(transport="stdio")
