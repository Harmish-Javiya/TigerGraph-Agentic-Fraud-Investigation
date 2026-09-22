"""
Standalone MCP/TigerGraph diagnostic script. Run this from agent/ directly:

    python test_mcp.py

It does three things, in order, so you can see exactly where the chain breaks:
  1. Raw MCP connectivity + lists every tool the MCP server actually exposes
  2. REST-level check of whether each query your code calls is installed
  3. Calls each graph_client.py function one at a time with full tracebacks
"""

import asyncio
import sys
import time
import requests
import graph_client as gc

MCP_URL = "http://localhost:8000/mcp/"
TG_REST_URL = "http://localhost:9000"
GRAPH_NAME = "fraud_investigation"

# Every query name your code currently calls
QUERIES_USED = [
    "get_transaction", "card_baseline", "card_window", "region_neighbors",
    "device_neighbors", "similar_closed_cases", "customer_cards",
    "write_investigation_case", "write_investigation_event",
]


# ── 1. Raw MCP connectivity + tool listing ───────────────────────────────────

async def list_mcp_tools():
    from mcp.client.streamable_http import streamable_http_client
    from mcp import ClientSession
    import httpx

    print("=" * 70)
    print("1. MCP connectivity + available tools")
    print("=" * 70)
    try:
        async with httpx.AsyncClient(timeout=15.0) as http_client:
            async with streamable_http_client(MCP_URL, http_client=http_client) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    print(f"✅ Connected to MCP server at {MCP_URL}\n")
                    tools = await session.list_tools()
                    print(f"Server exposes {len(tools.tools)} tool(s):")
                    for t in tools.tools:
                        print(f"  - {t.name}")
                        if t.description:
                            print(f"      {t.description[:120]}")
    except Exception as e:
        print(f"❌ Could not connect to MCP server at {MCP_URL}: {e}")
        print("   Is the TigerGraph MCP bridge process actually running?")
        sys.exit(1)


# ── 2. REST-level check: is each query even installed? ──────────────────────

def check_installed_queries():
    print("\n" + "=" * 70)
    print("2. Is each query installed on the graph? (RESTPP-level check)")
    print("=" * 70)
    for q in QUERIES_USED:
        url = f"{TG_REST_URL}/query/{GRAPH_NAME}/{q}"
        try:
            # empty POST — we only care about 404 (not installed) vs anything else
            resp = requests.post(url, json={}, timeout=5)
            if resp.status_code == 404:
                print(f"  ❌ {q:<28} NOT INSTALLED (404) — run INSTALL QUERY {q} in gsql")
            elif resp.status_code in (200, 400, 500):
                # 400/500 here usually just means "missing params", which is fine —
                # it proves the query IS installed and reachable.
                print(f"  ✅ {q:<28} installed (HTTP {resp.status_code})")
            else:
                print(f"  ⚠️  {q:<28} unexpected HTTP {resp.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"  ❌ {q:<28} REST call failed: {e}")


# ── 3. Call each graph_client function directly, one at a time ──────────────

def test_functions():
    print("\n" + "=" * 70)
    print("3. Calling each graph_client.py function directly")
    print("=" * 70)
    print("(replace the sample IDs below with real ones from your case pack)\n")

    tests = [
        ("get_transaction", lambda: gc.get_transaction("3514030")),
        ("card_baseline", lambda: gc.card_baseline("C12382")),
        ("card_window", lambda: gc.card_window("C12382", "2016-12-05 01:55:28", 48)),
        ("region_neighbors", lambda: gc.region_neighbors("220", "2016-12-05 01:55:28", 7)),
        ("customer_cards", lambda: gc.customer_cards("C12382")),
        ("similar_closed_cases", lambda: gc.similar_closed_cases("card_not_present_fraud", "C12382")),
        # device_neighbors needs a real device_key — pull one from the lookup first
        ("device_neighbors", lambda: gc.device_neighbors(
            gc.get_device_key("3514030") or "UNKNOWN", "2016-12-05 01:55:28", 30
        )),
        ("write_investigation_case", lambda: gc.write_investigation_case(
            "TEST-MCP-001", "C12382", "CARD-1", "2016-12-05 01:55:28",
            "open", "uncertain", 0.5, "none", 0.0, "mcp diagnostic test"
        )),
        ("write_investigation_event", lambda: gc.write_investigation_event(
            "TEST-MCP-001", "evt-1", "test", "{}", "2016-12-05 01:55:28"
        )),
    ]

    for name, fn in tests:
        gc.TOOL_CALLS = 0
        start = time.time()
        try:
            result = fn()
            elapsed = time.time() - start
            if isinstance(result, bool):
                status = "✅" if result else "❌"
                print(f"  {status} {name:<26} {elapsed:5.1f}s  returned={result}")
            else:
                size = len(result) if hasattr(result, "__len__") else "n/a"
                print(f"  ✅ {name:<26} {elapsed:5.1f}s  result size={size}")
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ❌ {name:<26} {elapsed:5.1f}s  RAISED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(list_mcp_tools())
    check_installed_queries()
    test_functions()
    print("\nDone. Anything marked ❌ above is your next thing to fix.")