import os
import requests
from dotenv import load_dotenv

load_dotenv()

TG_HOST = os.getenv("TG_HOST", "localhost")
TG_PORT = os.getenv("TG_PORT", "9000")
TG_GRAPH = os.getenv("TG_GRAPH", "fraud_investigation")
TG_USERNAME = os.getenv("TG_USERNAME", "tigergraph")
TG_PASSWORD = os.getenv("TG_PASSWORD", "tigergraph")

BASE_URL = f"http://{TG_HOST}:{TG_PORT}/query/{TG_GRAPH}"


def _get(query_name: str, params: dict) -> dict:
    """Call a GSQL installed query via REST API."""
    url = f"{BASE_URL}/{query_name}"
    response = requests.get(
        url,
        params=params,
        auth=(TG_USERNAME, TG_PASSWORD),
        timeout=30
    )
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise RuntimeError(f"TigerGraph error: {data.get('message')}")
    return data.get("results", [])


def get_transaction(txn_id: str) -> dict:
    """
    Fetch a single transaction and all its attributes.
    Returns the transaction dict or empty dict if not found.
    """
    results = _get("get_transaction", {"txn_id": txn_id})
    if results and results[0].get("Result"):
        return results[0]["Result"][0]["attributes"]
    return {}


def card_history(customer_id: str) -> list:
    """
    Fetch all transactions for a customer (baseline behavior).
    Returns list of transaction dicts sorted by ts ASC.
    """
    results = _get("card_history", {"customer_id": customer_id})
    if results and results[0].get("Txns"):
        return [t["attributes"] for t in results[0]["Txns"]]
    return []


def card_window(customer_id: str, center_ts: str, hours: int) -> list:
    """
    Fetch transactions within ±hours of center_ts for a customer.
    center_ts format: 'YYYY-MM-DD HH:MM:SS'
    Returns list of transaction dicts sorted by ts ASC.
    """
    results = _get("card_window", {
        "customer_id": customer_id,
        "center_ts": center_ts,
        "hours": hours
    })
    if results and results[0].get("Txns"):
        return [t["attributes"] for t in results[0]["Txns"]]
    return []


def device_neighbors(device_key: str) -> dict:
    """
    Find other cards and closed cases sharing the same device profile.
    Returns dict with keys: transactions, cards, cases
    """
    results = _get("device_neighbors", {"device_key": device_key})
    out = {"transactions": [], "cards": [], "cases": []}
    if not results:
        return out
    for block in results:
        if "Txns" in block:
            out["transactions"] = [t["attributes"] for t in block["Txns"]]
        if "Cards" in block:
            out["cards"] = [c["attributes"] for c in block["Cards"]]
        if "Cases" in block:
            out["cases"] = [cc["attributes"] for cc in block["Cases"]]
    return out


def region_neighbors(region_code: str, from_ts: str, days: int) -> list:
    """
    Find transactions in the same billing region within a time window.
    Useful for detecting out-of-region clusters.
    Returns list of transaction dicts.
    """
    results = _get("region_neighbors", {
        "region_code": region_code,
        "from_ts": from_ts,
        "days": days
    })
    if results and results[0].get("Txns"):
        return [t["attributes"] for t in results[0]["Txns"]]
    return []


def similar_closed_cases(pattern_name: str, customer_id: str) -> dict:
    """
    Find closed cases matching a pattern or the same customer.
    Returns dict with keys: by_pattern (up to 10), by_customer (up to 5)
    """
    results = _get("similar_closed_cases", {
        "pattern_name": pattern_name,
        "customer_id": customer_id
    })
    out = {"by_pattern": [], "by_customer": []}
    if not results:
        return out
    for block in results:
        if "ByPattern" in block:
            out["by_pattern"] = [cc["attributes"] for cc in block["ByPattern"]]
        if "ByCustomer" in block:
            out["by_customer"] = [cc["attributes"] for cc in block["ByCustomer"]]
    return out


def customer_cards(customer_id: str) -> list:
    """
    Fetch all cards owned by a customer.
    Returns list of card dicts.
    """
    results = _get("customer_cards", {"customer_id": customer_id})
    if results and results[0].get("Cards"):
        return [c["attributes"] for c in results[0]["Cards"]]
    return []


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
    summary: str
) -> bool:
    """
    Write the agent's findings back to the graph as an InvestigationCase vertex.
    Returns True on success.
    """
    url = f"{BASE_URL}/write_investigation_case"
    params = {
        "case_id": case_id,
        "customer_id": customer_id,
        "card_id": card_id,
        "opened_at": opened_at,
        "status": status,
        "verdict": verdict,
        "fraud_probability": fraud_probability,
        "pattern": pattern,
        "exposure_usd": exposure_usd,
        "summary": summary
    }
    try:
        response = requests.get(
            url,
            params=params,
            auth=(TG_USERNAME, TG_PASSWORD),
            timeout=30
        )
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Warning: could not write case to graph: {e}")
        return False


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing graph_client.py against HHG-001...\n")

    # Test 1 — flagged transaction
    print("=== get_transaction(3514030) ===")
    txn = get_transaction("3514030")
    print(f"  amount: ${txn.get('amount')} | channel: {txn.get('channel')} | risk: {txn.get('risk_score')}")

    # Test 2 — card history
    print("\n=== card_history(C12382) ===")
    history = card_history("C12382")
    print(f"  Total transactions: {len(history)}")
    if history:
        amounts = [t["amount"] for t in history]
        print(f"  Amount range: ${min(amounts):.2f} - ${max(amounts):.2f}")
        print(f"  Avg amount: ${sum(amounts)/len(amounts):.2f}")

    # Test 3 — burst window
    print("\n=== card_window(C12382, 2016-12-05 01:55:28, 48h) ===")
    window = card_window("C12382", "2016-12-05 01:55:28", 48)
    print(f"  Transactions in 48h window: {len(window)}")
    for t in window:
        print(f"    {t.get('ts')} | ${t.get('amount')} | {t.get('channel')}")

    # Test 4 — customer cards
    print("\n=== customer_cards(C12382) ===")
    cards = customer_cards("C12382")
    print(f"  Cards: {[c.get('card_id') for c in cards]}")

    # Test 5 — similar closed cases
    print("\n=== similar_closed_cases(card_not_present_fraud, C12382) ===")
    cases = similar_closed_cases("card_not_present_fraud", "C12382")
    print(f"  By pattern: {len(cases['by_pattern'])} cases")
    print(f"  By customer: {len(cases['by_customer'])} cases")
    if cases["by_pattern"]:
        ex = cases["by_pattern"][0]
        print(f"  Example: {ex.get('case_id')} | {ex.get('outcome')} | ${ex.get('exposure_usd')}")

    print("\n✅ graph_client.py working correctly")