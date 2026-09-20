"""
GraphRAG module — local vector store implementation.
TigerGraph CE 4.3 does not support vector attributes,
so we embed and search locally using numpy + sentence-transformers.

Knowledge sources:
1. Closed case analyst notes (5565 examples with outcomes)
2. Fraud policy document (R1-R10 rules + pattern descriptions)
3. Fraud typology descriptions
"""

import os
import json
import pickle
import numpy as np
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TG_HOST = os.getenv("TG_HOST", "http://localhost")
TG_PORT = os.getenv("TG_RESTPP_PORT", "9000")
TG_GRAPH = os.getenv("TG_GRAPHNAME", "fraud_investigation")
TG_USER = os.getenv("TG_USERNAME", "tigergraph")
TG_PASS = os.getenv("TG_PASSWORD", "tigergraph")

# Local vector store path
VECTOR_STORE_PATH = Path(__file__).parent / "vector_store.pkl"

# ── Embedding model ────────────────────────────────────────────────────────────

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print("[GraphRAG] Loading embedding model...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("[GraphRAG] Model ready.")
    return _embedder

def embed(text: str) -> np.ndarray:
    return get_embedder().encode(text, normalize_embeddings=True)

def embed_batch(texts: list) -> np.ndarray:
    return get_embedder().encode(
        texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True
    )


# ── Fraud policy knowledge base ───────────────────────────────────────────────

FRAUD_POLICY = {
    "card_not_present_fraud": """
Pattern: Card Not Present Fraud
Fraudulent online transactions where physical card is not required.
Key signals: channel=online, unusual merchant, P_emaildomain doesn't match history,
M5=F, C1 elevated, risk_score > 0.6.
Policy: VERIFY_WITH_CUSTOMER (R1). If denied + exposure > $1000: BLOCK_CARD + FILE_REPORT (R2).
""",
    "card_not_present_new_device": """
Pattern: Card Not Present — New Device
Online transaction from a device never seen before for this customer.
Key signals: channel=online, new DeviceProfile, different OS/browser, new screen resolution.
Policy: VERIFY_WITH_CUSTOMER (R1), STEP_UP_AUTH. If denied: BLOCK_CARD + CREATE_CASE (R2).
""",
    "out_of_region_use": """
Pattern: Out of Region Use
Card-present transaction in a billing region the customer has no history in.
Key signals: channel=in_person, addr1 not in usual regions, D1 > 30 days gap, dist1/dist2 large.
Policy: VERIFY_WITH_CUSTOMER (R1). If denied: BLOCK_CARD. Check region cluster (R6).
File SAR if exposure > $1000 (R2).
""",
    "account_takeover": """
Pattern: Account Takeover
Fraudster gains access and transacts across multiple cards with new device.
Key signals: Multiple cards active, new DeviceProfile, high C1, C6 elevated,
large amounts, both online and in_person in short window.
Policy: BLOCK_ALL_CARDS if 2+ confirmed (R10), FILE_REPORT always (R2),
MONITOR_CONNECTED_CARDS (R6).
""",
    "card_testing": """
Pattern: Card Testing
Fraudster tests stolen card with many small transactions before large purchase.
Key signals: C1 very high, amounts $1-$10, multiple declines (C6 elevated),
channel=online, short time window.
Policy: DECLINE + STEP_UP_AUTH + BLOCK_CARD immediately (R5).
File SAR if confirmed > $100.
""",
    "undocumented": """
Pattern: Undocumented / Novel
Suspicious signals but doesn't fit known patterns. Requires escalation.
Key signals: Unusual combination not matching known typologies.
Policy: CREATE_CASE + FILE_REPORT + ESCALATE_TO_ANALYST (R9).
""",
    "none": """
Pattern: No Fraud Pattern
Transaction appears legitimate. Risk score elevated but evidence insufficient.
Key signals: Amount consistent with baseline, usual channel/region, positive match flags.
Policy: CLOSE_NO_FRAUD if customer confirms (R3). MONITOR_CARD if uncertain.
"""
}

SAR_GUIDANCE = """
FinCEN SAR Narrative Guidelines:
- State who: subject name, account/card ID
- Describe what: type of suspicious activity, amounts, dates
- Explain why: signals that triggered review
- Include timeframe: first and last suspicious transaction dates
- Mention action taken: card blocked, customer contacted
- Professional tone, factual, 100-200 words
"""


# ── Vector store ──────────────────────────────────────────────────────────────

class LocalVectorStore:
    def __init__(self):
        self.embeddings = None   # np.ndarray shape (N, 384)
        self.metadata = []       # list of dicts with case info

    def add(self, texts: list, metadata: list):
        vecs = embed_batch(texts)
        if self.embeddings is None:
            self.embeddings = vecs
        else:
            self.embeddings = np.vstack([self.embeddings, vecs])
        self.metadata.extend(metadata)

    def search(self, query_text: str, top_k: int = 5) -> list:
        if self.embeddings is None or len(self.metadata) == 0:
            return []
        q = embed(query_text)
        # Cosine similarity (embeddings are normalized)
        scores = self.embeddings @ q
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_idx:
            results.append({
                "score": float(scores[i]),
                "metadata": self.metadata[i]
            })
        return results

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({"embeddings": self.embeddings, "metadata": self.metadata}, f)
        print(f"[GraphRAG] Vector store saved to {path}")

    def load(self, path: Path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        print(f"[GraphRAG] Loaded {len(self.metadata)} vectors from {path}")


_store = None

def get_store() -> LocalVectorStore:
    global _store
    if _store is None:
        _store = LocalVectorStore()
        if VECTOR_STORE_PATH.exists():
            _store.load(VECTOR_STORE_PATH)
    return _store


# ── Fetch closed cases from TigerGraph ───────────────────────────────────────

def fetch_closed_cases() -> list:
    """Fetch all closed cases via REST API."""
    url = f"{TG_HOST}:{TG_PORT}/query/{TG_GRAPH}/get_all_closed_cases"
    try:
        resp = requests.get(url, auth=(TG_USER, TG_PASS), timeout=60)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if results and "Result" in results[0]:
            return [r["attributes"] for r in results[0]["Result"]]
    except Exception as e:
        print(f"[GraphRAG] REST fetch failed: {e}")

    # Fallback — try pyTigerGraph
    try:
        import pyTigerGraph as tg
        conn = tg.TigerGraphConnection(
            host=TG_HOST,
            graphname=TG_GRAPH,
            username=TG_USER,
            password=TG_PASS,
            restppPort=TG_PORT
        )
        result = conn.runInstalledQuery("get_all_closed_cases")
        if result and "Result" in result[0]:
            return [r["attributes"] for r in result[0]["Result"]]
    except Exception as e:
        print(f"[GraphRAG] pyTigerGraph fetch failed: {e}")

    return []


# ── Build vector index ────────────────────────────────────────────────────────

def build_index():
    """
    Fetch all closed cases, embed analyst notes, save to local vector store.
    Run once — takes ~2 minutes for 5565 cases.
    """
    print("[GraphRAG] Fetching closed cases from TigerGraph...")
    cases = fetch_closed_cases()

    if not cases:
        print("[GraphRAG] No cases fetched. Make sure get_all_closed_cases query is installed.")
        return False

    print(f"[GraphRAG] Building index for {len(cases)} cases...")

    texts = []
    metadata = []
    for c in cases:
        text = (
            f"Pattern: {c.get('pattern', 'unknown')}. "
            f"Outcome: {c.get('outcome', 'unknown')}. "
            f"Exposure: ${c.get('exposure_usd', 0):.2f}. "
            f"Actions: {c.get('actions_taken', '')}. "
            f"Notes: {c.get('analyst_notes', '')}"
        )
        texts.append(text)
        metadata.append({
            "case_id": c.get("case_id", ""),
            "customer_id": c.get("customer_id", ""),
            "outcome": c.get("outcome", ""),
            "pattern": c.get("pattern", ""),
            "exposure_usd": c.get("exposure_usd", 0),
            "n_txns": c.get("n_txns", 0),
            "actions_taken": c.get("actions_taken", ""),
            "analyst_notes": c.get("analyst_notes", "")[:400]
        })

    store = get_store()
    store.add(texts, metadata)
    store.save(VECTOR_STORE_PATH)
    print(f"[GraphRAG] Index built. {len(metadata)} cases indexed.")
    return True


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_context(case: dict, txn: dict, baseline: dict, top_k: int = 5) -> str:
    """
    Main GraphRAG function.
    Returns enriched context string to inject into LLM prompt.
    """
    store = get_store()

    # Build search query from current case
    query_text = (
        f"channel={txn.get('channel', '')} "
        f"amount=${txn.get('amount', 0)} "
        f"region={txn.get('addr1', '')} "
        f"risk_score={txn.get('risk_score', 0)} "
        f"D1={txn.get('d1', 0)} days gap "
        f"M1={txn.get('m1', '')} M5={txn.get('m5', '')} "
        f"C1={txn.get('c1', 0)} C6={txn.get('c6', 0)} "
        f"usual_regions={baseline.get('usual_regions', [])} "
        f"avg_amount=${baseline.get('avg_amount', 0)}"
    )

    # Search vector store
    similar = store.search(query_text, top_k=top_k) if store.embeddings is not None else []

    # Guess likely pattern for policy retrieval
    pattern = guess_pattern(txn, baseline)
    policy = FRAUD_POLICY.get(pattern, FRAUD_POLICY["none"])

    # Build context string
    lines = []

    lines.append("=== FRAUD POLICY FOR LIKELY PATTERN ===")
    lines.append(policy.strip())

    lines.append("\n=== SAR NARRATIVE GUIDANCE ===")
    lines.append(SAR_GUIDANCE.strip())

    if similar:
        lines.append(f"\n=== TOP {len(similar)} SIMILAR HISTORICAL CASES ===")
        for i, r in enumerate(similar, 1):
            m = r["metadata"]
            lines.append(
                f"\n[Case {i} | {m.get('case_id')} | similarity={r['score']:.3f}]"
                f"\nOutcome: {m.get('outcome')} | Pattern: {m.get('pattern')} "
                f"| Exposure: ${m.get('exposure_usd', 0):.2f}"
                f"\nActions taken: {m.get('actions_taken', '')}"
                f"\nNotes: {m.get('analyst_notes', '')[:300]}"
            )
    else:
        lines.append("\n=== SIMILAR CASES ===")
        lines.append("Vector index not built yet — run: python graphrag.py build")
        lines.append("Using policy knowledge only.")

    return "\n".join(lines)


def guess_pattern(txn: dict, baseline: dict) -> str:
    channel = txn.get("channel", "")
    c1 = float(txn.get("c1", 0) or 0)
    c6 = float(txn.get("c6", 0) or 0)
    amount = float(txn.get("amount", 0) or 0)
    region = str(int(float(txn.get("addr1", 0)))) if txn.get("addr1") else ""
    usual = baseline.get("usual_regions", [])

    if c1 > 5 and amount < 20:
        return "card_testing"
    if c1 > 3 and c6 > 2 and channel == "online":
        return "account_takeover"
    if channel == "in_person" and region and region not in usual:
        return "out_of_region_use"
    if channel == "online":
        return "card_not_present_fraud"
    return "none"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        print("Building GraphRAG vector index...")
        success = build_index()
        if success:
            print("✅ Done. Vector store saved to vector_store.pkl")
        else:
            print("❌ Failed. Check TigerGraph connection and get_all_closed_cases query.")

    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Testing GraphRAG retrieval for HHG-001...\n")
        context = retrieve_context(
            case={"case_id": "HHG-001", "customer_id": "C12382"},
            txn={"txn_id": "3514030", "amount": 77.07, "channel": "in_person",
                 "risk_score": 0.61, "addr1": 444, "d1": 82,
                 "m1": "T", "m5": "F", "c1": 1, "c6": 0, "product_cd": "W"},
            baseline={"avg_amount": 115.0, "usual_regions": ["220", "221", "87"],
                      "usual_channels": ["in_person"]}
        )
        print(context)
        print("\n✅ GraphRAG retrieval working")

    else:
        print("Usage:")
        print("  python graphrag.py build  — build vector index from closed cases")
        print("  python graphrag.py test   — test retrieval on HHG-001")