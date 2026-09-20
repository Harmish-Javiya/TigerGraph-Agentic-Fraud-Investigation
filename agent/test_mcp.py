# test_mcp.py
import asyncio
from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession

async def test():
    async with streamable_http_client("http://localhost:8000/mcp/") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            for t in tools.tools:
                print(t.name)

asyncio.run(test())