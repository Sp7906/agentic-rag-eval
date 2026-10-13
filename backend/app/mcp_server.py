"""MCP server exposing the handbook RAG as tools for any MCP client (Claude Desktop, IDEs, agents).

Run:  python -m app.mcp_server      (needs: pip install -r requirements-mcp.txt)
"""
from mcp.server.fastmcp import FastMCP

from . import agent

mcp = FastMCP("northwind-handbook")


@mcp.tool()
def search_handbook(query: str, k: int = 4) -> list[dict]:
    """Hybrid (BM25 + dense) search over the Northwind employee handbook."""
    return [{"source": h.chunk.source, "heading": h.chunk.heading, "text": h.chunk.text, "score": h.score}
            for h in agent.get_retriever().search(query, k)]


@mcp.tool()
def ask_handbook(question: str) -> dict:
    """Answer a policy question with citations, or abstain if the handbook doesn't cover it."""
    out = agent.answer(question)
    return {"answer": out["answer"], "abstained": out["abstained"],
            "citations": [{"source": c["source"], "heading": c["heading"]} for c in out["citations"]]}


if __name__ == "__main__":
    mcp.run()
