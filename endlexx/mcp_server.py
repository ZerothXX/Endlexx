"""Read-only MCP bridge to the same scoped knowledge service (stdio transport)."""

from mcp.server.fastmcp import FastMCP
from sqlalchemy import select

from .db import Character, Session, init_db
from .knowledge import retrieve

mcp = FastMCP("Endlexx Knowledge")


@mcp.tool()
def list_characters() -> list[dict]:
    """List characters in this single-owner local workspace. Does not expose files or secrets."""
    with Session() as db:
        return [{"id": char.id, "name": char.name} for char in db.scalars(select(Character))]


@mcp.tool()
def search_character_knowledge(character_id: str, query: str, version: str | None = None) -> dict:
    """Retrieve cited source passages for one explicit character; read-only, no image execution."""
    if not 1 <= len(query) <= 2000:
        raise ValueError("Query must contain 1 to 2000 characters")
    with Session() as db:
        if not db.get(Character, character_id):
            raise ValueError("Unknown character")
    return retrieve(character_id, query, version)


if __name__ == "__main__":
    init_db()
    mcp.run(transport="stdio")
