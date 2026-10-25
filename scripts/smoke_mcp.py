"""Verify actual MCP stdio handshake and read-only tool discovery."""

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


async def main():
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "endlexx.mcp_server"],
        cwd=str(ROOT),
        # MCP intentionally inherits only a small OS environment by default.
        # Explicitly forward the isolated test DB instead of reading production.
        env={
            name: value
            for name, value in os.environ.items()
            if name in {"ENDLEXX_DATABASE_URL", "ENDLEXX_DATA_DIR"}
        },
    )
    async with stdio_client(parameters) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        names = {tool.name for tool in tools.tools}
        assert names == {"list_characters", "search_character_knowledge"}
        result = await session.call_tool("list_characters", {})
        assert not result.isError, result.content
        report = {"transport": "stdio", "tools": sorted(names), "handshake": "passed", "list_call": "passed"}
        rows = (result.structuredContent or {}).get("result", [])
        if rows:
            search = await session.call_tool(
                "search_character_knowledge", {"character_id": rows[0]["id"], "query": "月牙徽记在哪边肩膀？"}
            )
            assert not search.isError, search.content
            data = search.structuredContent or json.loads(search.content[0].text)
            data = data.get("result", data)
            assert "citations" in data, "MCP search must return the knowledge service payload"
            report["search_call"] = "passed"
            report["citation_count"] = len(data.get("citations", []))
            report["warnings"] = data.get("warnings", [])
        unknown = await session.call_tool(
            "search_character_knowledge", {"character_id": "nonexistent-acceptance-role", "query": "test"}
        )
        assert unknown.isError, "An unknown role must never search other roles' documents"
        report["unknown_role_rejected"] = True
        target = ROOT / "data/verification/mcp.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))


if __name__ == "__main__":
    asyncio.run(main())
