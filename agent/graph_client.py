import os
import csv
import json
import asyncio
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MCP_URL = "http://localhost:8000/mcp/"

# Bug fix: honest, centralized tool-call counting.
# Every call that reaches the MCP server — success or failure — increments this
# exactly once. investigator.py should reset gc.TOOL_CALLS = 0 at the start of
# each case and read it once at the end, instead of manually incrementing a
# separate counter that can drift out of sync with real query attempts.
TOOL_CALLS = 0

DEVICE_LOOKUP_PATH = os.getenv(
    "TXN_DEVICE_CSV",
    str(Path(__file__).parent.parent / "HHGOA_IEEE_DATASETS" / "txn_device.csv")
)
_device_lookup = None  # lazy-loaded txn_id -> device_key


def _parse_mcp_payload(raw: str) -> list:
    if not raw or not raw.strip():
        return []
    start = raw.find("{")
    if start == -1:
        return []
    obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    if not obj.get("success", True):
        raise RuntimeError(f"MCP error: {obj.get('error') or obj.get('message')}")
    data = obj.get("data", {})
    return data.get("result", data.get("results", []))


async def _query_mcp(query_name: str, params: dict) -> list:
    from mcp.client.streamable_http import streamable_http_client
    from mcp import ClientSession
    import httpx  # bug fix: was "httpx2", which doesn't exist

    timeout = 10.0
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        async with streamable_http_client(MCP_URL, http_client=http_client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "tigergraph__run_installed_query",
                    arguments={
                        "graph_name": "fraud_investigation",
                        "query_name": query_name,
                        "params": params
                    }
                )
                raw = result.content[0].text if result.content else ""
                return _parse_mcp_payload(raw)


def _describe_exception(e: BaseException) -> str:
    """asyncio.TaskGroup wraps real failures in an (Base)ExceptionGroup whose
    default str() is just "unhandled errors in a TaskGroup (N sub-exception)"
    — useless for debugging. Unwrap and describe the actual sub-exception(s)
    instead, recursively, so a connection-refused, timeout, or protocol error
    is visible in the logs rather than hidden behind that summary line.
    """
    sub_exceptions = getattr(e, "exceptions", None)
    if sub_exceptions:
        return "; ".join(_describe_exception(sub) for sub in sub_exceptions)
    return f"{type(e).__name__}: {e}"


def _run_with_retry(query_name: str, params: dict):
    """Up to 2 retries (3 attempts total) with increasing backoff. Returns (success: bool, data: list).
    Bumped from 1 retry after observing the MCP bridge occasionally drop the
    connection outright (not just individual queries erroring), especially on
    the first call after being idle."""
    global TOOL_CALLS
    TOOL_CALLS += 1
    last_err = None
    for attempt in range(3):
        try:
            return True, asyncio.run(_query_mcp(query_name, params))
        except Exception as e:
            last_err = e
            if attempt < 2:
                wait = 2 * (attempt + 1)
                print(f"  [MCP retry {attempt + 1}/2] {query_name}: {_describe_exception(e)} — waiting {wait}s")
                time.sleep(wait)
    print(f"  [MCP FAIL] {query_name}: {_describe_exception(last_err)}")
    return False, []


def _get(query_name: str, params: dict) -> list:
    """For read queries. Always returns a list — [] on failure."""
    ok, data = _run_with_retry(query_name, params)
    return data if ok else []


def _execute(query_name: str, params: dict) -> bool:
    """For write/mutation queries. Returns True only if the call actually succeeded."""
    ok, _ = _run_with_retry(query_name, params)
    return ok


# ── Reads ───────────────────────────────────────────────────────────────────

def get_transaction(txn_id: str) -> dict:
    results = _get("get_transaction", {"txn_id": txn_id})
    if results and results[0].get("Result"):
        return results[0]["Result"][0]["attributes"]
    return {}

def card_baseline(customer_id: str) -> dict:
    results = _get("card_baseline", {"customer_id": customer_id})
    out = {}
    for block in results:
        out.update(block)
    total = int(out.get("total_txns", 0) or 0)
    sum_amt = float(out.get("sum_amount", 0) or 0)
    regions = [str(int(float(r))) for r in (out.get("regions") or []) if r]
    channels = [c for c in (out.get("channels") or []) if c]
    return {
        "total_txns": total,
        "avg_amount": round(sum_amt / total, 2) if total else 0,
        "max_amount": round(float(out.get("max_amount", 0) or 0), 2),
        "usual_channels": channels,
        "usual_regions": regions,
    }

def card_window(customer_id: str, center_ts: str, hours: int) -> list:
    results = _get("card_window", {
        "customer_id": customer_id,
        "center_ts": center_ts,
        "hours": hours
    })
    if results and results[0].get("Txns"):
        return [t["attributes"] for t in results[0]["Txns"]]
    return []

def region_neighbors(region_code: str, from_ts: str, days: int) -> dict:
    """
    Transactions billed in this region within `days` of from_ts, plus the
    cards that made them and any closed cases on those cards — needed so
    "confirmed fraud on a card sharing this region" can actually be checked
    against real ClosedCase.outcome data, not fields that don't exist on
    Transaction.
    """
    empty = {"transactions": [], "cards": [], "cases": []}
    results = _get("region_neighbors", {
        "region_code": region_code,
        "from_ts": from_ts,
        "days": days
    })
    out = dict(empty)
    for block in results:
        if "Txns" in block:
            out["transactions"] = [t["attributes"] for t in block["Txns"]]
        if "Cards" in block:
            out["cards"] = [c["attributes"] for c in block["Cards"]]
        if "Cases" in block:
            out["cases"] = [c["attributes"] for c in block["Cases"]]
    return out

def device_neighbors(device_key: str, from_ts: str, days: int) -> dict:
    """
    Everything connected to a device within `days` of from_ts: the
    transactions made from it, the cards that made them (via the Card
    -(MADE)-> Transaction edge — Transaction itself has no card_id
    attribute), the customers who own those cards, and any closed cases on
    those cards. Used to ground connected_card_ids / connected_device_profiles
    / shared_origin / coordinated_across_customers instead of trusting the
    LLM's say-so or reading fields that don't exist on Transaction.
    """
    empty = {"transactions": [], "cards": [], "customers": [], "cases": []}
    if not device_key:
        return empty
    results = _get("device_neighbors", {
        "device_key": device_key,
        "from_ts": from_ts,
        "days": days
    })
    out = dict(empty)
    for block in results:
        if "Txns" in block:
            out["transactions"] = [t["attributes"] for t in block["Txns"]]
        if "Cards" in block:
            out["cards"] = [c["attributes"] for c in block["Cards"]]
        if "Customers" in block:
            out["customers"] = [c["attributes"] for c in block["Customers"]]
        if "Cases" in block:
            out["cases"] = [c["attributes"] for c in block["Cases"]]
    return out

def similar_closed_cases(pattern_name: str, customer_id: str) -> dict:
    results = _get("similar_closed_cases", {
        "pattern_name": pattern_name,
        "customer_id": customer_id
    })
    out = {"by_pattern": [], "by_customer": []}
    for block in results:
        if "ByPattern" in block:
            out["by_pattern"] = [cc["attributes"] for cc in block["ByPattern"]]
        if "ByCustomer" in block:
            out["by_customer"] = [cc["attributes"] for cc in block["ByCustomer"]]
    return out

def customer_cards(customer_id: str) -> list:
    results = _get("customer_cards", {"customer_id": customer_id})
    if results and results[0].get("Cards"):
        return [c["attributes"] for c in results[0]["Cards"]]
    return []


# ── Local device lookup (txn_id -> device_key) ───────────────────────────────

def _load_device_lookup() -> dict:
    global _device_lookup
    if _device_lookup is not None:
        return _device_lookup
    _device_lookup = {}
    path = Path(DEVICE_LOOKUP_PATH)
    if not path.exists():
        print(f"  [WARN] txn_device lookup not found at {path} — device grounding disabled")
        return _device_lookup
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txn_id = str(row.get("txn_id") or row.get("TransactionID") or "").strip()
            device_key = str(row.get("device_key") or row.get("DeviceInfo") or "").strip()
            if txn_id and device_key:
                _device_lookup[txn_id] = device_key
    print(f"  [device_lookup] Loaded {len(_device_lookup)} txn->device mappings")
    return _device_lookup

def get_device_key(txn_id: str) -> str:
    """Returns the device fingerprint for a transaction, or '' if unknown."""
    if not txn_id:
        return ""
    return _load_device_lookup().get(str(txn_id), "")


# ── Writes ────────────────────────────────────────────────────────────────────

def write_investigation_case(
    case_id: str,
    customer_id: str,
    card_id: str,
    opened_at: str,
    status: str,
    verdict: str,
    fraud_probability: float,
    pattern: str,
    exposure_usd: float,
    summary: str,
) -> bool:
    """
    Writes the investigation result back to the graph as a Case vertex/edges.
    Requires a `write_investigation_case` installed query on the graph.
    Returns True only if the write actually succeeded (was previously missing
    entirely, which crashed every investigate_case() run at Step 8).
    """
    return _execute("write_investigation_case", {
        "case_id": case_id,
        "customer_id": customer_id,
        "card_id": card_id,
        "opened_at": opened_at,
        "status": status,
        "verdict": verdict,
        "fraud_probability": fraud_probability,
        "pattern": pattern,
        "exposure_usd": exposure_usd,
        "summary": summary,
    })


def write_investigation_event(
    case_id: str,
    event_id: str,
    event_type: str,
    payload: str,
    created_at: str,
) -> bool:
    """Persist an append-only audit event attached to an InvestigationCase.

    The accompanying `gsql/investigation_lifecycle.gsql` file defines this
    installed query.  Events make evidence, recommendation changes, and the
    final decision queryable in TigerGraph instead of leaving them only in the
    exported JSON case file.
    """
    return _execute("write_investigation_event", {
        "case_id": case_id,
        "event_id": event_id,
        "event_type": event_type,
        "payload": payload,
        "created_at": created_at,
    })


if __name__ == "__main__":
    print("=== MCP Quick Test ===")
    txn = get_transaction("3514030")
    print(f"  txn={txn.get('amount')} risk={txn.get('risk_score')}")
    baseline = card_baseline("C12382")
    print(f"  baseline: {baseline['total_txns']} txns, avg=${baseline['avg_amount']}")
    window = card_window("C12382", "2016-12-05 01:55:28", 48)
    print(f"  window: {len(window)} txns")
    cards = customer_cards("C12382")
    print(f"  cards: {[c.get('card_id') for c in cards]}")
    cases = similar_closed_cases("card_not_present_fraud", "C12382")
    print(f"  similar: {len(cases['by_pattern'])} pattern, {len(cases['by_customer'])} customer")
    ok = write_investigation_case(
        "TEST-001", "C12382", "CARD-1", "2016-12-05 01:55:28",
        "open", "uncertain", 0.5, "none", 0.0, "smoke test"
    )
    print(f"  write_investigation_case ok={ok}")
    print(f"  total tool calls this run: {TOOL_CALLS}")
    print("✅ graph_client.py OK")