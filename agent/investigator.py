"""
Main investigation agent — produces exact required submission format.
Aligned against Fraud Policy v1.0.
"""

import os
import json
import time
import copy
import re
from datetime import datetime
from dotenv import load_dotenv

import graph_client as gc
from llm_router import chat as routed_chat
from policy_engine import PolicyInput, apply_policy, R1_VERIFY_THRESHOLD
from graphrag import guess_pattern
from answer_schema import (
    CaseAnswer, Case, SAR, NextBestActions,
    EvidenceItem, EvidenceRequest, ActionItem,
    CaseStatus, Verdict, EvidenceSource, ActionRoute,
    QASummary,
    get_route, get_rule
)

load_dotenv()

# §6 Stopping rule: stop once probability clears these bounds with enough evidence
STOP_HIGH = 0.85
STOP_LOW = 0.15

_tokens = 0  # LLM tokens are still tracked here — gc.TOOL_CALLS handles tool calls


# ── LLM helpers ───────────────────────────────────────────────────────────────

def llm(prompt: str, system: str = "") -> str:
    """Call the configured provider chain and track the returned usage.

    Provider order is defined centrally in llm_router.py: Gemini, local Ollama,
    then Groq. This avoids pinning the investigation workflow to deprecated
    `groq/compound` and lets one unavailable provider fail over safely.
    """
    global _tokens
    text, used = routed_chat(prompt, system=system, max_tokens=2000)
    _tokens += used
    return text


def llm_json(prompt: str, system: str = "") -> dict:
    full_system = (system or "") + "\nRespond ONLY with valid JSON. No markdown, no preamble."
    # Ask the router to validate JSON before returning. The local parsing below
    # still produces the object used by the investigation code.
    global _tokens
    raw, used = routed_chat(prompt, system=full_system, json_mode=True, max_tokens=2000)
    _tokens += used
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── Graph evidence gathering ──────────────────────────────────────────────────

def _only_dicts(records: list, label: str) -> list:
    """Graph client functions are documented to return lists of attribute
    dicts (e.g. device_neighbors' docstring: "[t["attributes"] for t in
    results[0]["Txns"]]"), but an installed GSQL query can return a
    differently-shaped payload for some records (e.g. a bare vertex ID string
    instead of an attribute map) without graph_client.py itself raising an
    error. Every consumer here assumes dicts and calls .get()/[...] on each
    entry, so a single non-dict record crashes the whole investigation
    (AttributeError: 'str' object has no attribute 'get'). Filter those out
    here, once, with a warning, instead of crashing per case.
    """
    clean = [r for r in records if isinstance(r, dict)]
    if len(clean) != len(records):
        print(
            f"  [WARN] {label}: dropped {len(records) - len(clean)} non-dict "
            f"record(s) returned by the graph query (unexpected shape)."
        )
    return clean


def gather_evidence(case: dict, txn: dict) -> dict:
    customer_id = case["customer_id"]
    opened_at = case["opened_at"]

    print(f"  [1/6] Baseline...")
    baseline = gc.card_baseline(customer_id)

    print(f"  [2/6] 72h window...")
    window = _only_dicts(gc.card_window(customer_id, opened_at, 72), "card_window")

    print(f"  [3/6] Customer cards...")
    cards = _only_dicts(gc.customer_cards(customer_id), "customer_cards")

    print(f"  [4/6] Similar closed cases...")
    likely_pattern = guess_pattern(txn, baseline)
    similar = gc.similar_closed_cases(likely_pattern, customer_id)
    similar_dict = similar if isinstance(similar, dict) else {}
    similar_dict["by_pattern"] = _only_dicts(similar_dict.get("by_pattern", []), "similar_closed_cases.by_pattern")
    similar_dict["by_customer"] = _only_dicts(similar_dict.get("by_customer", []), "similar_closed_cases.by_customer")
    baseline["has_prior_fraud"] = any(
        cc.get("outcome") == "confirmed_fraud"
        for cc in similar_dict.get("by_customer", [])
    )

    print(f"  [5/6] Region neighbors...")
    region = []
    if txn.get("addr1"):
        try:
            region = _only_dicts(gc.region_neighbors(str(int(float(txn["addr1"]))), opened_at, 7), "region_neighbors")
        except Exception:
            pass

    print(f"  [6/6] Device neighbors...")
    device_key = gc.get_device_key(txn.get("txn_id", ""))
    device_txns = []
    if device_key:
        try:
            device_txns = _only_dicts(gc.device_neighbors(device_key, opened_at, 30), "device_neighbors")
        except Exception as e:
            print(f"  [WARN] device_neighbors failed: {e}")

    device_cards = sorted({
        str(t.get("card_id")) for t in device_txns
        if t.get("card_id") and str(t.get("card_id")) != str(case.get("card_id"))
    })
    # §R9: another customer entirely sharing this device is a coordinated-abuse signal
    other_customers = {
        str(t.get("customer_id")) for t in device_txns
        if t.get("customer_id") and str(t.get("customer_id")) != str(customer_id)
    }

    device_data = {
        "device_key": device_key,
        "transactions": device_txns,
        "cards": device_cards,
        "other_customers": sorted(other_customers),
    }

    return {
        "case": case,
        "flagged_txn": txn,
        "history": [],
        "window": window,
        "cards": cards,
        "similar_cases": similar_dict,
        "region_neighbors": region,
        "device_data": device_data,
        "baseline": baseline,
        "prior_case_ids": [],  # filled in by analyse_evidence
    }


# ── Enforce probability consistency ──────────────────────────────────────────

def enforce_consistency(analysis: dict, trigger_risk_score: float) -> dict:
    prob = analysis.get("fraud_probability", trigger_risk_score)
    min_prob = max(0.0, trigger_risk_score - 0.20)
    max_prob = min(1.0, trigger_risk_score + 0.20)
    prob = max(min_prob, min(max_prob, prob))
    analysis["fraud_probability"] = round(prob, 2)

    # §6 Stopping rule thresholds — 0.85/0.15, not an arbitrary 0.80/0.30
    p = analysis["fraud_probability"]
    if p >= STOP_HIGH:
        analysis["verdict"] = "fraud"
    elif p <= STOP_LOW:
        analysis["verdict"] = "legitimate"
    else:
        analysis["verdict"] = "uncertain"

    return analysis


# ── Ground analysis in real graph data (accuracy fixes) ──────────────────────

def ground_affected_transactions(analysis: dict, bundle: dict, txn: dict) -> dict:
    if analysis.get("verdict") == "legitimate":
        return analysis

    pattern = analysis.get("pattern", "none")
    flagged_id = str(txn.get("txn_id", ""))
    window = bundle.get("window", [])

    verified_ids = {flagged_id} if flagged_id else set()

    if pattern == "card_testing":
        for t in window:
            try:
                if float(t.get("amount", 0) or 0) < 20:
                    verified_ids.add(str(t.get("txn_id", "")))
            except (TypeError, ValueError):
                continue
    elif pattern == "account_takeover":
        for t in window:
            verified_ids.add(str(t.get("txn_id", "")))
    elif pattern in ("card_not_present_fraud", "card_not_present_new_device"):
        for t in window:
            if t.get("channel") == "online":
                verified_ids.add(str(t.get("txn_id", "")))
    elif pattern == "out_of_region_use":
        usual = set(bundle.get("baseline", {}).get("usual_regions", []))
        for t in window:
            region = t.get("addr1")
            if region:
                try:
                    region = str(int(float(region)))
                except (TypeError, ValueError):
                    region = str(region)
                if region not in usual:
                    verified_ids.add(str(t.get("txn_id", "")))

    verified_ids.discard("")
    if not verified_ids and flagged_id:
        verified_ids = {flagged_id}

    ids_sorted = sorted(
        verified_ids,
        key=lambda tid: next(
            (t.get("ts", "") for t in window if str(t.get("txn_id", "")) == tid), ""
        )
    )
    if not ids_sorted and flagged_id:
        ids_sorted = [flagged_id]

    analysis["affected_txn_ids"] = ids_sorted
    analysis["first_suspicious_txn_id"] = ids_sorted[0] if ids_sorted else ""

    total = 0.0
    for tid in ids_sorted:
        if tid == flagged_id:
            total += float(txn.get("amount", 0) or 0)
        else:
            match = next((t for t in window if str(t.get("txn_id", "")) == tid), None)
            if match:
                try:
                    total += float(match.get("amount", 0) or 0)
                except (TypeError, ValueError):
                    pass
    analysis["exposure_usd"] = round(total, 2) if total else float(txn.get("amount", 0) or 0)
    return analysis


def ground_shared_origin(analysis: dict, bundle: dict) -> dict:
    """
    Populate connected_card_ids / connected_device_profiles / shared_origin
    from the actual device_neighbors query. Real corroborating evidence like
    this is also allowed to push the probability past the trigger-score clamp
    in enforce_consistency() — a confirmed shared device is stronger signal
    than the original risk score alone.
    """
    device_data = bundle.get("device_data", {})
    device_cards = device_data.get("cards", [])
    device_key = device_data.get("device_key", "")

    # A shared device observation is lead evidence, not a confirmed shared
    # fraud origin.  The actual R6 predicate is derived later from explicit
    # confirmed-fraud records.  Do not let a neighbor count alter probability,
    # connected-card IDs, or policy actions here.
    return analysis


def ground_prior_case_bias(analysis: dict, bundle: dict) -> dict:
    """
    §5 case memory: nudge fraud_probability using the confirmed outcomes of
    similar prior cases, deterministically — instead of leaving this purely
    to whatever the LLM inferred from the prompt text.
    """
    similar_dict = bundle.get("similar_cases", {})
    if not isinstance(similar_dict, dict):
        return analysis
    pool = (similar_dict.get("by_pattern", []) + similar_dict.get("by_customer", []))[:8]
    if len(pool) < 2:
        return analysis

    fraud_count = sum(1 for cc in pool if cc.get("outcome") == "confirmed_fraud")
    fraud_ratio = fraud_count / len(pool)

    current_p = analysis.get("fraud_probability", 0.0)
    if fraud_ratio >= 0.7 and current_p < STOP_HIGH:
        analysis["fraud_probability"] = round(min(0.95, current_p + 0.05), 2)
    elif fraud_ratio <= 0.3 and current_p > STOP_LOW:
        analysis["fraud_probability"] = round(max(0.05, current_p - 0.05), 2)

    p = analysis["fraud_probability"]
    if p >= STOP_HIGH:
        analysis["verdict"] = "fraud"
    elif p <= STOP_LOW:
        analysis["verdict"] = "legitimate"
    return analysis


def augment_evidence_to_minimum(evidence_items: list, bundle: dict, txn: dict, analysis: dict) -> list:
    """
    §6 stopping rule requires >=2 independent evidence items before a
    decisive (>=0.85 or <=0.15) verdict is allowed to stand. If the LLM's own
    evidence list came up short, pad it with deterministic facts already
    sitting in the bundle — no extra LLM call, no extra tool call.
    """
    p = analysis.get("fraud_probability", 0.5)
    decisive = p >= STOP_HIGH or p <= STOP_LOW
    if not decisive or len(evidence_items) >= 2:
        return evidence_items

    baseline = bundle.get("baseline", {})
    window = bundle.get("window", [])
    customer_id = bundle["case"]["customer_id"]

    if baseline.get("total_txns"):
        evidence_items.append(EvidenceItem(
            claim=(
                f"Customer baseline: {baseline['total_txns']} prior transactions, "
                f"avg ${baseline.get('avg_amount', 0)}, usual channels "
                f"{baseline.get('usual_channels', [])}, usual regions {baseline.get('usual_regions', [])}."
            ),
            source=EvidenceSource.GRAPH,
            ref="query:card_baseline",
            entity_ids=[str(customer_id)]
        ))

    if len(evidence_items) < 2 and window:
        evidence_items.append(EvidenceItem(
            claim=f"{len(window)} transactions occurred on this account within the 72h window around the flagged transaction.",
            source=EvidenceSource.GRAPH,
            ref="query:card_window",
            entity_ids=[str(t.get("txn_id")) for t in window[:5]]
        ))

    if len(evidence_items) < 2:
        evidence_items.append(EvidenceItem(
            claim=f"Flagged transaction {txn.get('txn_id')} carried an upstream model risk_score of {txn.get('risk_score')}.",
            source=EvidenceSource.GRAPH,
            ref="query:get_transaction",
            entity_ids=[str(txn.get("txn_id"))]
        ))

    return evidence_items


def ensure_pattern_description(analysis: dict) -> dict:
    if analysis.get("pattern") != "undocumented":
        return analysis

    desc = (analysis.get("pattern_description") or "").strip()
    if len(desc.split()) >= 20:
        return analysis

    print("  [RETRY] pattern_description too short for undocumented pattern — expanding...")
    try:
        expanded = llm(
            f"""The following fraud case was classified as 'undocumented' — it doesn't
match any known typology. Write pattern_description as 2-4 sentences (at least
30 words) describing the unusual combination of signals observed.

Weak description given: "{desc}"
Case summary: {analysis.get('summary', '')}
Evidence: {json.dumps(analysis.get('evidence', []))}

Return ONLY the description text, no JSON, no preamble."""
        )
        if len(expanded.split()) >= 20:
            analysis["pattern_description"] = expanded.strip()
        else:
            analysis["pattern_description"] = (
                f"{desc} This case combines signals that do not cleanly match any "
                f"documented fraud typology (card testing, card-not-present, "
                f"out-of-region use, or account takeover) and was therefore "
                f"flagged as undocumented for analyst review rather than forced "
                f"into an existing pattern category."
            )
    except Exception as e:
        print(f"  [WARN] pattern_description expansion failed: {e}")
        analysis["pattern_description"] = (
            f"{desc} Signals observed do not match a documented fraud typology; "
            f"flagged undocumented pending analyst review of the full evidence set."
        )
    return analysis


def compute_matches_recurring_charge(txn: dict, baseline: dict) -> bool:
    """§R7 signal: does this charge look like the customer's own known recurring pattern?"""
    try:
        amount = float(txn.get("amount", 0) or 0)
        avg = float(baseline.get("avg_amount", 0) or 0)
    except (TypeError, ValueError):
        return False
    channel = txn.get("channel")
    usual_channels = baseline.get("usual_channels", [])
    return bool(avg) and abs(amount - avg) <= 5 and channel in usual_channels


def _confirmed_fraud_record(record: dict) -> bool:
    """Accept only an explicit graph-provided confirmation, never model prose."""
    return str(record.get("outcome", "")).lower() == "confirmed_fraud" or any(
        record.get(key) is True
        for key in ("confirmed_fraud", "fraud_confirmed", "is_confirmed_fraud")
    )


def _parse_timestamp(value: object) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")) if value else None
    except ValueError:
        return None


def find_card_testing_sequence(txn: dict, window: list[dict], card_id: str) -> list[dict]:
    """R5: return the actual small-authorization records (chronological) that
    make up a qualifying card-testing sequence — three or more sub-$5 online
    authorizations within one hour before a larger (>$5) purchase. Returns []
    if the pattern does not hold. Kept separate from the boolean check so
    evidence text can cite the real records instead of just a yes/no.
    """
    records = [row for row in window if str(row.get("card_id", card_id)) == str(card_id)] or list(window)
    flagged_at = _parse_timestamp(txn.get("ts"))
    try:
        flagged_amount = float(txn.get("amount", 0) or 0)
    except (TypeError, ValueError):
        return []
    if not flagged_at or flagged_amount <= 5:
        return []

    smalls = []
    for row in records:
        try:
            is_small = float(row.get("amount", 0) or 0) < 5
        except (TypeError, ValueError):
            is_small = False
        if is_small and row.get("channel") == "online":
            timestamp = _parse_timestamp(row.get("ts"))
            if timestamp and 0 <= (flagged_at - timestamp).total_seconds() <= 3600:
                smalls.append(row)

    if len(smalls) < 3:
        return []
    return sorted(smalls, key=lambda r: r.get("ts", ""))


def detect_card_testing(txn: dict, window: list[dict], card_id: str) -> bool:
    """R5: three small online authorizations within one hour before a larger purchase."""
    return bool(find_card_testing_sequence(txn, window, card_id))


def derive_policy_facts(bundle: dict, txn: dict, customer_response: dict | None,
                        validation_requested: bool) -> dict:
    """Derive the R1–R10 predicates from validated state for one case."""
    case = bundle["case"]
    baseline = bundle.get("baseline", {})
    device_records = bundle.get("device_data", {}).get("transactions", [])
    region_records = bundle.get("region_neighbors", [])
    customer_denied = bool(customer_response and customer_response.get("responded")
                           and customer_response.get("confirmed_fraud") is True)
    customer_confirmed = bool(customer_response and customer_response.get("responded")
                              and customer_response.get("confirmed_fraud") is False)
    no_response_24h = bool(validation_requested and customer_response
                           and customer_response.get("responded") is False)

    def confirmed_cards(records: list[dict]) -> set[str]:
        return {str(row["card_id"]) for row in records
                if row.get("card_id") and _confirmed_fraud_record(row)}

    confirmed_device_cards = confirmed_cards(device_records)
    confirmed_region_cards = confirmed_cards(region_records)
    if customer_denied:
        confirmed_device_cards.add(str(case["card_id"]))
        confirmed_region_cards.add(str(case["card_id"]))
    shared_origin_confirmed = len(confirmed_device_cards) >= 2 or len(confirmed_region_cards) >= 2

    confirmed_records = [row for row in device_records + region_records if _confirmed_fraud_record(row)]
    coordinated_abuse_confirmed = len({str(row["customer_id"]) for row in confirmed_records if row.get("customer_id")}) >= 2
    own_cards = {str(row["card_id"]) for row in bundle.get("cards", []) if row.get("card_id")}
    compromised_cards = (({str(case["card_id"])} if customer_denied else set()) |
                         confirmed_device_cards | confirmed_region_cards) & own_cards
    credentials_confirmed = any(
        row.get(key) is True
        for row in [txn] + device_records + region_records
        for key in ("credentials_compromised", "credential_compromise_confirmed")
    )
    card_testing = detect_card_testing(txn, bundle.get("window", []), case["card_id"])

    # §6 counts independent fraud/legitimacy indicators, never query rows.
    support = set()
    if float(txn.get("risk_score", 0) or 0) >= 0.70:
        support.add("upstream_risk_signal")
    try:
        amount = float(txn.get("amount", 0) or 0)
        maximum = float(baseline.get("max_amount", 0) or 0)
        average = float(baseline.get("avg_amount", 0) or 0)
        if (maximum and amount > maximum) or (average and amount >= average * 3):
            support.add("amount_anomaly")
    except (TypeError, ValueError):
        pass
    if txn.get("channel") and txn.get("channel") not in baseline.get("usual_channels", []):
        support.add("new_channel")
    if card_testing:
        support.add("card_testing_sequence")
    if shared_origin_confirmed:
        support.add("confirmed_shared_origin")
    if customer_denied:
        support.add("customer_denial")
    if customer_confirmed:
        support.add("customer_confirmation")

    return {
        "customer_denied": customer_denied,
        "customer_confirmed": customer_confirmed,
        "customer_no_response_24h": no_response_24h,
        "card_testing_detected": card_testing,
        "shared_origin_confirmed": shared_origin_confirmed,
        "coordinated_abuse_confirmed": coordinated_abuse_confirmed,
        "confirmed_compromised_cards": len(compromised_cards),
        "credentials_confirmed_compromised": credentials_confirmed,
        "independent_support_count": len(support),
        "weak_evidence": len(support) <= 1,
    }


# ── LLM analysis ─────────────────────────────────────────────────────────────

def analyse_evidence(bundle: dict, trigger_risk_score: float) -> dict:
    case = bundle["case"]
    txn = bundle["flagged_txn"]
    baseline = bundle["baseline"]
    similar = bundle["similar_cases"]
    similar_dict = similar if isinstance(similar, dict) else {}

    try:
        from graphrag import retrieve_context, get_store
        store = get_store()
        if store.embeddings is not None:
            rag_context = retrieve_context(case, txn, baseline)
        else:
            rag_context = "GraphRAG vector index not built yet — using policy knowledge only."
    except Exception as e:
        rag_context = f"GraphRAG unavailable: {e}"

    prior_case_ids = [
        cc.get("case_id", "") for cc in similar_dict.get("by_pattern", [])[:3]
        if cc.get("case_id")
    ]
    bundle["prior_case_ids"] = prior_case_ids

    prior_cases_summary = []
    for cc in similar_dict.get("by_pattern", [])[:1]:
        prior_cases_summary.append({
            "case_id": cc.get("case_id"),
            "outcome": cc.get("outcome"),
            "pattern": cc.get("pattern"),
            "exposure": cc.get("exposure_usd"),
            "actions": cc.get("actions_taken", ""),
            "notes": cc.get("analyst_notes", "")[:200]
        })

    window_summary = [
        {"ts": t.get("ts"), "amount": t.get("amount"), "channel": t.get("channel")}
        for t in bundle["window"][:3]
    ]

    min_prob = max(0.0, trigger_risk_score - 0.20)
    max_prob = min(1.0, trigger_risk_score + 0.20)

    prompt = f"""
You are a fraud analyst at a bank. Investigate this case and return a detailed JSON analysis.

CASE: {case['case_id']}
Customer: {case['customer_id']} | Card: {case['card_id']}
Trigger: {case.get('trigger_text', '')}
TRIGGER RISK SCORE: {trigger_risk_score}
fraud_probability MUST be between {min_prob:.2f} and {max_prob:.2f}

RETRIEVED KNOWLEDGE BASE CONTEXT:
{rag_context}

FLAGGED TRANSACTION:
- ID: {txn.get('txn_id')}
- Amount: ${txn.get('amount')}
- Channel: {txn.get('channel')}
- Risk Score: {txn.get('risk_score')}
- Billing Region addr1: {txn.get('addr1')}
- ProductCD: {txn.get('product_cd')}
- D1 (days since last txn): {txn.get('d1')}
- M1={txn.get('m1')} M2={txn.get('m2')} M3={txn.get('m3')} M5={txn.get('m5')}
- C1 (txn count): {txn.get('c1')} C6 (decline count): {txn.get('c6')}
- dist1={txn.get('dist1')} dist2={txn.get('dist2')}

CUSTOMER BASELINE ({baseline.get('total_txns', 0)} transactions):
- Avg amount: ${baseline.get('avg_amount', 0)}
- Max amount: ${baseline.get('max_amount', 0)}
- Usual channels: {baseline.get('usual_channels', [])}
- Usual regions: {baseline.get('usual_regions', [])}

72H WINDOW ({len(bundle['window'])} transactions):
{json.dumps(window_summary, indent=2)}

SIMILAR CLOSED CASES (from memory):
{json.dumps(prior_cases_summary, indent=2)}

REGION ACTIVITY last 7 days: {len(bundle['region_neighbors'])} transactions

CARDS ON THIS ACCOUNT: {[c.get('card_id') for c in bundle['cards']]}

Return this exact JSON:
{{
  "verdict": "fraud OR legitimate OR uncertain",
  "fraud_probability": number between {min_prob:.2f} and {max_prob:.2f},
  "pattern": "card_testing | card_not_present_fraud | card_not_present_new_device | out_of_region_use | account_takeover | undocumented | none",
  "pattern_description": "2-3 sentences ONLY if pattern=undocumented, else empty string",
  "status": "closed_fraud OR closed_legitimate OR open OR escalated",
  "affected_txn_ids": ["list of transaction IDs that are part of this fraud episode, including flagged one if fraud"],
  "first_suspicious_txn_id": "earliest fraud transaction ID or empty string if legitimate",
  "connected_card_ids": ["other card IDs compromised in the same episode"],
  "connected_device_profiles": ["device profile strings e.g. DeviceInfo|OS|browser|screen"],
  "exposure_usd": number,
  "shared_origin": true or false,
  "evidence": [
    {{
      "claim": "Full sentence describing specific finding with data points",
      "source": "graph OR document OR customer OR external",
      "ref": "query name or document section that produced this e.g. query:card_history(C12382)",
      "entity_ids": ["IDs this claim rests on"]
    }}
  ],
  "similar_prior_cases": {json.dumps(prior_case_ids)},
  "summary": "2-6 sentences an analyst could read",
  "analyst_notes": "internal notes on investigation reasoning",
  "needs_customer_contact": true or false,
  "customer_contact_question": "specific question with amount and date",
  "stop_reason": "why the investigation would end here without customer response"
}}

Evidence rules:
- At least 3 evidence items (policy §6 requires 2+ independent items to stop)
- Each claim must cite specific data (amounts, dates, IDs, region codes)
- source=graph for graph query results
- source=document for policy/typology knowledge. For these, do NOT paraphrase
  or restate the policy's wording — state only which rule applies (e.g. "R3
  applies: the customer confirmed the transaction") and reference it by ID
  (R1-R10) in the claim. Never invent or loosely reword what a rule says.
- source=customer is added separately by the system after contact — do not
  fabricate a customer-sourced evidence item yourself
- entity_ids must be real IDs from the dataset
- risk_score is an input signal, not a policy or pattern threshold. Do not
  claim that any score clears a fraud threshold unless the supplied policy
  explicitly states that threshold.
- C1-C14 and D1-D15 are opaque dataset features. State their field name and
  value, but never invent a specific definition such as "transaction count".
- When describing a time window, include the flagged transaction or explicitly
  say "other transactions" if you exclude it.
- For legitimate verdict: affected_txn_ids=[], exposure_usd=0
"""

    _prompt_len = len(prompt) + len(rag_context)
    if _prompt_len > 4000:
        window_summary = window_summary[:2]
        prior_cases_summary = []
        rag_context = (rag_context[:1200] + "...") if len(rag_context) > 1200 else rag_context

    analysis = llm_json(
        prompt,
        system="You are an expert fraud analyst. Be precise and data-driven. Return only valid JSON."
    )
    return enforce_consistency(analysis, trigger_risk_score)


# ── Customer response simulation ──────────────────────────────────────────────

def simulate_customer_response(question: str, prob: float) -> dict:
    """Simulate a customer reply for demo purposes.

    This is never a real customer statement. The returned dict is tagged
    ``"source": "simulated_customer_response"`` so nothing downstream can
    mistake it for an authoritative, real customer fact (e.g. a real R2/R3
    confirmation) without that provenance being visible in the record.
    """
    prompt = f"""
A bank customer was asked: "{question}"
Fraud probability: {prob}

Simulate realistic response:
- prob > 0.80: 70% confirms fraud
- prob < 0.40: 80% says legitimate
- 0.40-0.80: 40% no response, 30% fraud, 30% legitimate

Return JSON only:
{{
  "responded": true or false,
  "confirmed_fraud": true or false,
  "response_text": "exact customer statement or No response within 24 hours"
}}

IMPORTANT: "confirmed_fraud": true means the customer CONFIRMED THE TRANSACTION IS FRAUD
(i.e. they did NOT authorize it, they deny making it). "confirmed_fraud": false means
the customer confirmed the transaction was LEGITIMATE (they authorized it).
"""
    result = llm_json(prompt)
    result["source"] = "simulated_customer_response"
    return result


# ── Build action items ────────────────────────────────────────────────────────

def build_action_items(actions: list[str], exposure: float, action_reasons: dict[str, str],
                       initial_probability: float | None = None) -> list[ActionItem]:
    items = []
    for action in actions:
        # Reasons must come from the policy result for this investigation.
        # A generic action-to-rule lookup cannot know which predicate held.
        reason = action_reasons.get(action, get_rule(action, {"exposure": exposure}))
        # A pre-contact block cannot cite a customer denial that occurred only
        # after the initial recommendation was captured.
        if initial_probability is not None and action == "BLOCK_CARD":
            reason = (
                f"Pre-contact assessment: fraud probability {initial_probability:.2f}; "
                "recommendation recorded pending additional evidence"
            )
        items.append(ActionItem(
            action=action,
            route=ActionRoute(get_route(action, exposure)),
            reason=reason
        ))
    return items


def canonical_graph_ref(claim: str, proposed_ref: str, case: dict, txn: dict) -> str:
    """Use a traceable installed-query reference for graph evidence.

    LLMs frequently invent friendly query names such as ``customer_history``.
    The answer format requires the query that actually produced the fact.
    """
    text = f"{claim} {proposed_ref}".lower()
    customer_id = case["customer_id"]
    if any(word in text for word in ("baseline", "average", "usual channel", "usual region", "maximum")):
        return f"query:card_baseline(customer_id={customer_id})"
    if any(word in text for word in ("window", "burst", "within 72", "within 48")):
        return f"query:card_window(customer_id={customer_id}, hours=72)"
    if any(word in text for word in ("device", "profile", "shared origin")):
        return "query:device_neighbors"
    if any(word in text for word in ("region", "billing")):
        return "query:region_neighbors"
    if any(word in text for word in ("prior case", "closed case", "similar case")):
        return "query:similar_closed_cases"
    return f"query:get_transaction(txn_id={txn.get('txn_id', case['flagged_txn_id'])})"


def normalize_graph_claim(claim: str) -> str:
    """Remove unsupported glosses that commonly appear in LLM evidence prose."""
    # The dataset deliberately does not define individual C/D fields. Preserve
    # the observed value while removing a fabricated business definition.
    claim = re.sub(
        r"\btransaction count \(C1\)", "unnamed C1 count feature", claim,
        flags=re.IGNORECASE,
    )
    claim = re.sub(
        r"\bC1\) is only", "C1 is", claim, flags=re.IGNORECASE,
    )
    # The README says risk scores are signals, never a pattern threshold. Do
    # not allow a fabricated numerical threshold to look like policy evidence.
    claim = re.sub(
        r"\s*,?\s*which\s+(?:exceeds|is above|is higher than)\s+(?:the\s+)?[0-9.]+\s+threshold\s+for[^.]*\.?,?",
        ".", claim, flags=re.IGNORECASE,
    )
    return claim


def build_grounded_evidence(bundle: dict, txn: dict, analysis: dict,
                            customer_response: dict | None,
                            trigger_risk_score: float) -> list[EvidenceItem]:
    """Build submission evidence only from facts actually retrieved or recorded.

    LLM-written evidence may sound plausible while inventing thresholds, query
    names, customer actions, or feature meanings. The LLM still assesses the
    pattern and risk, but final evidence is assembled from the graph bundle and
    the explicit simulated response.
    """
    case = bundle["case"]
    baseline = bundle.get("baseline", {})
    window = bundle.get("window", [])
    txn_id = str(txn.get("txn_id", case["flagged_txn_id"]))
    amount = float(txn.get("amount", 0) or 0)
    evidence = [EvidenceItem(
        claim=(
            f"Flagged transaction {txn_id} was {txn.get('channel', 'unknown')} for "
            f"${amount:.2f}; its upstream model risk score was {trigger_risk_score:.2f}."
        ),
        source=EvidenceSource.GRAPH,
        ref=f"query:get_transaction(txn_id={txn_id})",
        entity_ids=[txn_id],
    )]

    if baseline.get("total_txns"):
        evidence.append(EvidenceItem(
            claim=(
                f"Customer {case['customer_id']} baseline contains {baseline['total_txns']} "
                f"transactions, average amount ${baseline.get('avg_amount', 0):.2f}, "
                f"maximum amount ${baseline.get('max_amount', 0):.2f}, usual channels "
                f"{baseline.get('usual_channels', [])}, and usual regions "
                f"{baseline.get('usual_regions', [])}."
            ),
            source=EvidenceSource.GRAPH,
            ref=f"query:card_baseline(customer_id={case['customer_id']})",
            entity_ids=[case["customer_id"]],
        ))

    if window:
        window_ids = [str(item.get("txn_id")) for item in window[:8] if item.get("txn_id")]
        channels = sorted({str(item.get("channel")) for item in window if item.get("channel")})
        evidence.append(EvidenceItem(
            claim=(
                f"The 72-hour graph window contains {len(window)} transaction(s) "
                f"with channel(s) {channels}."
            ),
            source=EvidenceSource.GRAPH,
            ref=f"query:card_window(customer_id={case['customer_id']}, hours=72)",
            entity_ids=window_ids,
        ))

        # If the flagged transaction is preceded by a qualifying card-testing
        # sequence, cite the real records (amounts, count, elapsed minutes)
        # rather than describing the pattern in the abstract.
        sequence = find_card_testing_sequence(txn, window, case["card_id"])
        if sequence:
            small_amounts = [f"${float(r.get('amount', 0) or 0):.2f}" for r in sequence]
            first_ts = _parse_timestamp(sequence[0].get("ts"))
            flagged_ts = _parse_timestamp(txn.get("ts"))
            elapsed_min = (
                round((flagged_ts - first_ts).total_seconds() / 60)
                if first_ts and flagged_ts else None
            )
            elapsed_clause = f" over {elapsed_min} minute(s)" if elapsed_min is not None else ""
            evidence.append(EvidenceItem(
                claim=(
                    f"{len(sequence)} online authorization(s) under $5 "
                    f"({', '.join(small_amounts)}){elapsed_clause}, followed by "
                    f"a ${amount:.2f} transaction — the sequence R5 requires."
                ),
                source=EvidenceSource.GRAPH,
                ref=f"query:card_window(customer_id={case['customer_id']}, hours=72)",
                entity_ids=[str(r.get("txn_id")) for r in sequence if r.get("txn_id")] + [txn_id],
            ))

    device_data = bundle.get("device_data", {})
    device_key = device_data.get("device_key", "")
    device_cards = [str(card) for card in device_data.get("cards", []) if card]
    if device_key or device_cards:
        evidence.append(EvidenceItem(
            claim=(
                f"Device lookup returned profile '{device_key}' and {len(device_cards)} "
                f"other card(s) in its configured graph-neighbor window."
            ),
            source=EvidenceSource.GRAPH,
            ref="query:device_neighbors",
            entity_ids=device_cards,
        ))

    prior_ids = [str(case_id) for case_id in bundle.get("prior_case_ids", []) if case_id]
    if prior_ids:
        evidence.append(EvidenceItem(
            claim=f"Retrieved {len(prior_ids)} similar closed case(s) for case-memory comparison.",
            source=EvidenceSource.GRAPH,
            ref="query:similar_closed_cases",
            entity_ids=prior_ids,
        ))

    if customer_response and customer_response.get("responded"):
        provenance = customer_response.get("source", "unknown")
        if provenance == "simulated_customer_response":
            claim = (
                f"Simulated customer response (not a real customer statement): "
                f"'{customer_response.get('response_text', '')}'"
            )
        elif provenance == "trigger_customer_report":
            claim = f"Customer-reported statement: '{customer_response.get('response_text', '')}'"
        else:
            claim = f"Customer response: '{customer_response.get('response_text', '')}'"
        evidence.append(EvidenceItem(
            claim=claim,
            source=EvidenceSource.CUSTOMER,
            ref=provenance,
            entity_ids=[txn_id],
        ))
    return evidence


def build_final_summary(case: dict, txn: dict, analysis: dict, actions: list[ActionItem],
                        customer_response: dict | None, trigger_risk_score: float) -> str:
    """Create an evidence-only final summary with no unsupported execution claim."""
    outcome = "No customer response was recorded." if not customer_response else (
        "Customer validation confirmed unauthorized activity." if customer_response.get("confirmed_fraud")
        else "Customer validation confirmed the transaction was legitimate."
    )
    action_names = ", ".join(item.action for item in actions)
    return (
        f"Transaction {txn.get('txn_id')} (${float(txn.get('amount', 0) or 0):.2f}, "
        f"{txn.get('channel', 'unknown')}) was investigated after an upstream risk score of "
        f"{trigger_risk_score:.2f}. {outcome} Final assessment: {analysis['verdict']} "
        f"(P={analysis['fraud_probability']:.2f}), pattern {analysis['pattern']}. "
        f"Recommended actions: {action_names}."
    )


def derive_sar_activity_dates(analysis: dict, txn: dict, bundle: dict, case: dict) -> list[str]:
    """Derive [first_date, last_date] from real transaction timestamps of the
    affected txns (Fix #4). Never substitute case.opened_at for both ends —
    that invents activity dates rather than reporting validated ones.

    Falls back to opened_at only if no timestamped affected transaction is
    available at all, and is a documented limitation, not a silent invention:
    both ends collapse to the same real record when only one exists.
    """
    window = bundle.get("window", [])
    flagged_id = str(txn.get("txn_id", ""))
    ts_by_id = {str(t.get("txn_id", "")): t.get("ts") for t in window if t.get("ts")}
    if flagged_id and txn.get("ts"):
        ts_by_id[flagged_id] = txn.get("ts")

    dates = sorted({
        str(ts_by_id[tid])[:10]
        for tid in analysis.get("affected_txn_ids", [])
        if tid in ts_by_id and ts_by_id[tid]
    })
    if dates:
        return [dates[0], dates[-1]]
    if flagged_id and txn.get("ts"):
        d = str(txn["ts"])[:10]
        return [d, d]
    # No validated timestamp anywhere for the affected transactions — fall
    # back to the case's opened_at date rather than leaving the required
    # field empty; this is a known limitation, not a fabricated activity date.
    d = case["opened_at"][:10]
    return [d, d]


def build_no_sar_reason(policy, verdict: str) -> str:
    """Derive the sar.reason text for file=False strictly from policy.rules_applied.

    A rule name must never appear here unless the policy engine actually
    applied it for this case (Fix #3) — no hardcoded fallback rule text.
    """
    rules = policy.rules_applied
    if "R3" in rules:
        return "R3: customer confirmation established a legitimate transaction; FILE_REPORT is not required."
    if rules:
        return (
            f"Rule(s) {', '.join(rules)} applied for this case, but none of them "
            f"establish a FILE_REPORT condition; FILE_REPORT is not required."
        )
    return "No deterministic policy rule was applied that establishes a FILE_REPORT condition."


def build_sar_narrative(case: dict, txn: dict, analysis: dict, exposure: float,
                        final_actions: list[ActionItem], customer_response: dict | None) -> str:
    """Produce a factual SAR narrative; never claim recommendations executed."""
    date = case["opened_at"][:10]
    action_names = ", ".join(item.action for item in final_actions)
    if customer_response and customer_response.get("confirmed_fraud"):
        if customer_response.get("source") == "simulated_customer_response":
            customer_fact = (
                "A simulated customer response (not a real customer statement) indicated denial of authorization."
            )
        else:
            customer_fact = "The customer denied authorizing the activity when contacted."
    else:
        customer_fact = "No customer denial was recorded before the investigation decision."

    connected = analysis.get("connected_card_ids", [])
    linkage_clause = (
        f" A shared device profile links this activity to card(s) {', '.join(connected)}."
        if connected else ""
    )
    prior = analysis.get("similar_prior_cases", [])
    prior_clause = (
        f" Similar closed case(s) {', '.join(prior)} were referenced as case-memory context."
        if prior else ""
    )

    return (
        f"This report concerns customer {case['customer_id']} and card {case['card_id']}. "
        f"On {date}, transaction {txn.get('txn_id')} was recorded through the "
        f"{txn.get('channel', 'unknown')} channel for ${float(txn.get('amount', 0) or 0):.2f}. "
        f"The investigation assessed the activity as {analysis['pattern']} with fraud probability "
        f"{analysis['fraud_probability']:.2f}. {customer_fact}{linkage_clause}{prior_clause} "
        f"The identified suspicious exposure is ${exposure:.2f}. "
        f"The institution recommends {action_names} under the applicable fraud policy approval routes."
    )


# ── Main investigation ────────────────────────────────────────────────────────

def review_case_with_llm(answer: CaseAnswer, policy, policy_facts: dict,
                         customer_response: dict | None) -> dict:
    """LLM #2 — reviews the candidate CaseAnswer for internal contradictions.

    The reviewer is diagnostic only (Fix #9). It may point at problems, but
    is never allowed to invent or assert: customer denial/confirmation,
    shared origin, card testing, coordinated abuse, credential compromise,
    transaction/card/device IDs, or any R1-R10 predicate. It cannot write to
    the case — only deterministic_final_validate() below can, and only from
    already-established facts.
    """
    default = {
        "issues": [],
        "severity": "none",
        "recommended_corrections": [],
        "requires_deterministic_revalidation": False,
    }
    try:
        facts = {
            "rules_applied": policy.rules_applied,
            "policy_facts": policy_facts,
            "customer_response_source": (customer_response or {}).get("source"),
            "customer_response_responded": (customer_response or {}).get("responded"),
        }
        prompt = f"""
You are a fraud-case QA reviewer, not the decision-maker. You may only point
at contradictions already visible in the JSON below — you must NEVER invent,
assert, or imply a new fact of your own, including: customer denial or
confirmation, shared origin, card testing, coordinated abuse, credential
compromise, or any transaction/card/device ID or R1-R10 predicate not already
present in DETERMINISTIC FACTS.

DETERMINISTIC FACTS (authoritative — from the policy engine and validated
graph/customer state; nothing here can be second-guessed, only compared
against the candidate answer):
{json.dumps(facts, indent=2)}

CANDIDATE FINAL ANSWER JSON:
{answer.to_json()}

Check specifically for:
- an R1-R10 rule named anywhere in the answer that is NOT in rules_applied
- a customer-response claim inconsistent with customer_response_responded/source
- unsupported verdict/pattern claims in the summary
- evidence-count vs independent-support-count confusion in the summary text
- initial vs final action inconsistencies given whether evidence_requests is empty
- SAR inconsistencies (file vs reason vs presence of FILE_REPORT action)
- any transaction/card/device ID in the answer not present in its own evidence list

Return ONLY this JSON, nothing else:
{{
  "issues": ["short description of each problem found; empty list if none"],
  "severity": "none | warning | critical",
  "recommended_corrections": ["short description of what looks wrong and where — never a new fact you are asserting"],
  "requires_deterministic_revalidation": true or false
}}
"""
        review = llm_json(
            prompt,
            system="You are a meticulous QA reviewer. You find problems; you never invent facts. Output only valid JSON."
        )
        review.setdefault("issues", [])
        review.setdefault("severity", "none")
        review.setdefault("recommended_corrections", [])
        review.setdefault("requires_deterministic_revalidation", bool(review["issues"]))
        return review
    except Exception as e:
        print(f"  [WARN] LLM reviewer failed, proceeding with deterministic validation only: {e}")
        return default


def deterministic_final_validate(answer: CaseAnswer, policy, evidence_requests: list,
                                 review: dict) -> tuple[CaseAnswer, list[str]]:
    """Authoritative final check (Fix #10) — runs after the LLM reviewer.

    This function is the only thing allowed to change the candidate answer
    at this stage, and it only ever derives a replacement value from facts
    already established elsewhere in the deterministic pipeline (policy,
    policy.rules_applied, evidence_requests). It never applies a reviewer
    "recommended_correction" directly, and it never invents a fact the
    reviewer merely suggested — a reviewer-flagged issue is corrected only
    if this function's own checks independently confirm it.

    Returns (answer, corrections) where corrections is a human-readable
    audit log of anything actually changed.
    """
    corrections: list[str] = []
    allowed_rules = set(policy.rules_applied)
    rule_pattern = re.compile(r"\bR(?:10|[1-9])\b")

    # 1. sar.reason must never name a rule that wasn't actually applied.
    mentioned = set(rule_pattern.findall(answer.sar.reason))
    if mentioned - allowed_rules:
        corrections.append(
            f"sar.reason cited rule(s) {sorted(mentioned - allowed_rules)} not in "
            f"rules_applied={sorted(allowed_rules)}; regenerated deterministically."
        )
        answer.sar.reason = (
            policy.sar_reason if policy.file_sar
            else build_no_sar_reason(policy, answer.case.verdict.value)
        )

    # 2. Same check for FINAL action reasons only. Initial (pre-contact)
    # actions come from a separate, earlier policy pass whose own
    # rules_applied isn't carried in the answer schema, so R1-citing initial
    # actions are not cross-checked here — only final actions, which must
    # match this case's actual final rules_applied.
    for item in answer.next_best_actions.final:
        item_mentioned = set(rule_pattern.findall(item.reason))
        unsupported = item_mentioned - allowed_rules
        if unsupported:
            corrections.append(
                f"final action {item.action} reason cited unsupported rule(s) "
                f"{sorted(unsupported)}; replaced with the default policy reason."
            )
            item.reason = get_rule(item.action, {})

    # 3. Every action needs a non-empty reason.
    for item in list(answer.next_best_actions.initial) + list(answer.next_best_actions.final):
        if not item.reason or not item.reason.strip():
            item.reason = get_rule(item.action, {})
            corrections.append(f"action {item.action} had an empty reason; filled from the default rule table.")

    # 4. sar.file must match presence of FILE_REPORT in final actions. The
    # schema enforces this at construction time; this is a second, explicit
    # check in case anything upstream mutated fields after construction.
    has_file_report = "FILE_REPORT" in [a.action for a in answer.next_best_actions.final]
    if has_file_report != answer.sar.file:
        corrections.append(
            "sar.file did not match presence of FILE_REPORT in final actions; "
            "corrected sar.file to match final actions."
        )
        answer.sar.file = has_file_report

    # 5. legitimate verdict must carry no affected fraud txns / zero exposure.
    if answer.case.verdict == Verdict.LEGITIMATE:
        if answer.case.affected_txn_ids or answer.case.exposure_usd:
            corrections.append(
                "legitimate verdict carried non-empty affected_txn_ids/exposure_usd; "
                "cleared to satisfy the legitimate-case invariant."
            )
            answer.case.affected_txn_ids = []
            answer.case.exposure_usd = 0.0
            answer.case.first_suspicious_txn_id = ""

    # 6. Evidence entity_ids: drop empty-string entries (a formatting
    # artifact, never authoritative content).
    for item in answer.case.evidence:
        cleaned = [e for e in item.entity_ids if e]
        if cleaned != item.entity_ids:
            item.entity_ids = cleaned

    # 7. Initial vs final actions must be identical when no evidence was
    # requested (the README's stated invariant).
    if not evidence_requests:
        initial_names = [a.action for a in answer.next_best_actions.initial]
        final_names = [a.action for a in answer.next_best_actions.final]
        if initial_names != final_names:
            corrections.append(
                "no evidence_requests were recorded but initial/final actions "
                "differed; final actions copied into initial to satisfy the "
                "no-new-evidence invariant."
            )
            answer.next_best_actions.initial = list(answer.next_best_actions.final)
            answer.next_best_actions.what_changed = "nothing"

    # 8. The reviewer is not authoritative. A "critical" flag this
    # function's own checks did NOT independently confirm is logged for the
    # analyst, but never turned into a field change on the reviewer's say-so.
    if review.get("severity") == "critical" and not corrections:
        corrections.append(
            "LLM reviewer flagged severity=critical with no issue independently "
            "confirmed by the deterministic validator; logged only — no field "
            "was changed without a validated basis."
        )

    return answer, corrections


def investigate_case(case: dict) -> CaseAnswer:
    global _tokens
    gc.TOOL_CALLS = 0
    _tokens = 0
    start_time = time.time()

    case_id = case["case_id"]
    trigger_type = case.get("trigger_type", "risk_score")
    print(f"\n{'='*60}")
    print(f"Investigating {case_id} | {case['customer_id']} | trigger={trigger_type}")
    print(f"{'='*60}")

    trigger_risk_score_raw = case.get("risk_score", "") or ""
    trigger_risk_score = float(trigger_risk_score_raw) if trigger_risk_score_raw else 0.5

    # Step 0 — Flagged transaction
    print(f"  [0/6] Fetching transaction {case['flagged_txn_id']}...")
    txn = gc.get_transaction(case["flagged_txn_id"])
    if not txn:
        txn = {
            "txn_id": case["flagged_txn_id"],
            "amount": 0,
            "channel": "unknown",
            "risk_score": trigger_risk_score
        }

    # Step 1 — Gather graph evidence
    bundle = gather_evidence(case, txn)

    # Step 2 — LLM analysis
    print(f"  [LLM] Analysing evidence...")
    analysis = analyse_evidence(bundle, trigger_risk_score)
    print(f"  [LLM] Verdict={analysis['verdict']} P={analysis['fraud_probability']} Pattern={analysis['pattern']}")

    # Step 2b — Ground the LLM's claims in real graph data
    analysis = ground_shared_origin(analysis, bundle)
    analysis = ground_prior_case_bias(analysis, bundle)
    analysis["fraud_probability"] = round(analysis["fraud_probability"], 2)
    analysis = ground_affected_transactions(analysis, bundle, txn)
    analysis = ensure_pattern_description(analysis)

    # §6 is a guard on an LLM assessment, not a request to pad the evidence
    # list.  A decisive score without two independent validated indicators
    # remains unresolved and therefore goes through the normal validation path.
    preliminary_facts = derive_policy_facts(bundle, txn, None, False)
    if (
        trigger_type != "customer_report"
        and analysis["fraud_probability"] >= STOP_HIGH
        and preliminary_facts["independent_support_count"] < 2
    ):
        analysis["fraud_probability"] = STOP_HIGH - 0.01
        analysis["verdict"] = "uncertain"

    # Step 2c — Progress the case: write an OPEN record before contact/decision
    # (case memory should be updated as the investigation progresses,
    # not only once at the very end)
    gc.write_investigation_case(
        case_id=case_id, customer_id=case["customer_id"], card_id=case["card_id"],
        opened_at=case["opened_at"], status="open", verdict=analysis["verdict"],
        fraud_probability=analysis["fraud_probability"], pattern=analysis["pattern"],
        exposure_usd=float(analysis.get("exposure_usd", 0) or 0),
        summary=analysis.get("summary", "")
    )

    # Step 3 — Additional evidence / contact
    # Preserve the pre-contact assessment. Initial actions must describe what
    # was known before a simulated customer response, never the later verdict.
    pre_contact_analysis = copy.deepcopy(analysis)
    customer_response = None
    evidence_requests = []
    step_counter = len(bundle["window"]) + 4

    if trigger_type == "customer_report":
        customer_response = {
            "responded": True,
            "confirmed_fraud": True,
            "response_text": (
                case.get("trigger_text", "")
                .replace("Customer " + case["customer_id"] + " message: ", "")
                .replace("Refers to " + case["flagged_txn_id"] + ".", "")
                .strip()
                .replace("'", "")
            ),
            "source": "trigger_customer_report",
        }
        _tokens += 50
        print(f"  [CONTACT] Using known customer denial from trigger")

    elif STOP_LOW < analysis["fraud_probability"] < STOP_HIGH:
        # An unresolved probability is not a defensible closed verdict.  Always
        # gather and record a permitted validation step in this band instead of
        # letting an LLM silently declare a case closed without new evidence.
        # Outside this band the verdict is already decisive per §6.
        print(f"  [CONTACT] Simulating customer contact...")
        question = analysis.get(
            "customer_contact_question",
            f"Did you make a ${txn.get('amount')} transaction on {case['opened_at'][:10]}?"
        )
        customer_response = simulate_customer_response(question, analysis["fraud_probability"])
        _tokens += 200

        evidence_requests.append(EvidenceRequest(
            type="customer_validation",
            asked_after_step=step_counter,
            assumed_response=customer_response.get("response_text", "No response")
        ))

        if customer_response.get("responded"):
            if customer_response.get("confirmed_fraud"):
                analysis["fraud_probability"] = min(0.95, analysis["fraud_probability"] + 0.20)
                analysis["verdict"] = "fraud"
                analysis["status"] = "closed_fraud"
                print(f"  [CONTACT] Confirmed fraud → P={analysis['fraud_probability']}")
            else:
                # A customer confirmation settles the question under R3.  Do
                # not leave a case as "uncertain" while marking it closed
                # legitimate, which violates the answer-file contract.
                analysis["fraud_probability"] = 0.05
                analysis["verdict"] = "legitimate"
                analysis["status"] = "closed_legitimate"
                print(f"  [CONTACT] Confirmed legitimate → P={analysis['fraud_probability']}")

    # Analyst-provided context is recorded as an explicit evidence request too
    if trigger_type == "analyst_request":
        evidence_requests.append(EvidenceRequest(
            type="analyst_info",
            asked_after_step=0,
            assumed_response=case.get("trigger_text", "Analyst-provided context used as-is")
        ))

    analysis["fraud_probability"] = round(analysis["fraud_probability"], 2)
    analysis = ground_affected_transactions(analysis, bundle, txn)

    # Derive every policy predicate from this case's graph/customer state.
    # In particular, no LLM pattern label, probability, or prose can create a
    # rule condition.
    policy_facts = derive_policy_facts(
        bundle, txn, customer_response,
        any(request.type == "customer_validation" for request in evidence_requests),
    )
    card_testing_cleared_over_100 = (
        policy_facts["card_testing_detected"] and float(txn.get("amount", 0) or 0) > 100
    )
    matches_recurring_charge = compute_matches_recurring_charge(txn, bundle["baseline"])

    # §R8 signal: does the deterministic graph-side pattern guess disagree
    # with what the LLM classified? A real disagreement between two
    # independent readings of the evidence is exactly what R8 means by
    # "the evidence conflicts".
    heuristic_pattern = guess_pattern(txn, bundle["baseline"])
    # Two distinct concepts (Fix #7): not having enough independent support is
    # NOT the same thing as two independent readings actually disagreeing.
    # Only the latter is a real "evidence conflict" for R8's purposes.
    insufficient_evidence = policy_facts["independent_support_count"] < 2
    actual_evidence_conflict = (
        heuristic_pattern not in ("none", analysis["pattern"])
        and analysis["pattern"] not in ("none",)
    )
    evidence_conflicts = actual_evidence_conflict
    print(
        f"  [FACTS] insufficient_evidence={insufficient_evidence} "
        f"actual_evidence_conflict={actual_evidence_conflict}"
    )

    # Step 4 — Policy engine
    print(f"  [POLICY] Running rules...")
    similar_dict = bundle["similar_cases"] if isinstance(bundle["similar_cases"], dict) else {}
    prior_cases = similar_dict.get("by_customer", [])
    has_prior = any(cc.get("outcome") == "confirmed_fraud" for cc in prior_cases)
    exposure = analysis.get("exposure_usd", txn.get("amount", 0))

    policy_input = PolicyInput(
        verdict=analysis["verdict"],
        fraud_probability=analysis["fraud_probability"],
        exposure_usd=exposure,
        pattern=analysis["pattern"],
        customer_responded=customer_response.get("responded") if customer_response else None,
        customer_confirmed_fraud=customer_response.get("confirmed_fraud") if customer_response else None,
        shared_origin=policy_facts["shared_origin_confirmed"],
        n_cards_confirmed=policy_facts["confirmed_compromised_cards"],
        has_prior_fraud=has_prior,
        channel=txn.get("channel", "unknown"),
        evidence_count=len(build_grounded_evidence(bundle, txn, analysis, customer_response, trigger_risk_score)),
        evidence_conflicts=evidence_conflicts,
        card_testing_cleared_over_100=card_testing_cleared_over_100,
        matches_recurring_charge=matches_recurring_charge,
        coordinated_across_customers=policy_facts["coordinated_abuse_confirmed"],
        **policy_facts,
    )

    policy = apply_policy(policy_input)

    # Build the first recommendation from the assessment before any requested
    # evidence returned. This makes the two action snapshots auditable.
    initial_policy_input = PolicyInput(
        verdict=pre_contact_analysis["verdict"],
        fraud_probability=pre_contact_analysis["fraud_probability"],
        exposure_usd=exposure,
        pattern=pre_contact_analysis["pattern"],
        customer_responded=None,
        customer_confirmed_fraud=None,
        shared_origin=preliminary_facts["shared_origin_confirmed"],
        n_cards_confirmed=preliminary_facts["confirmed_compromised_cards"],
        has_prior_fraud=has_prior,
        channel=txn.get("channel", "unknown"),
        evidence_count=len(build_grounded_evidence(bundle, txn, pre_contact_analysis, None, trigger_risk_score)),
        evidence_conflicts=evidence_conflicts,
        card_testing_cleared_over_100=card_testing_cleared_over_100,
        matches_recurring_charge=matches_recurring_charge,
        coordinated_across_customers=preliminary_facts["coordinated_abuse_confirmed"],
        **preliminary_facts,
    )
    initial_action_strings = apply_policy(initial_policy_input).initial_actions
    # When the agent itself has identified an unresolved evidence gap and sent
    # a validation request, the pre-response snapshot must show that request
    # rather than prematurely treating the later customer answer as known.
    if any(request.type == "customer_validation" for request in evidence_requests):
        initial_action_strings = ["CREATE_CASE", "VERIFY_WITH_CUSTOMER"]
        if pre_contact_analysis["pattern"] in (
            "card_testing", "account_takeover", "card_not_present_new_device"
        ):
            initial_action_strings.append("STEP_UP_AUTH")

    # The README requires identical snapshots when no additional evidence was
    # requested.  In that situation all available evidence is already known,
    # so record the complete policy recommendation in both snapshots.
    if not evidence_requests:
        initial_action_strings = policy.final_actions.copy()
        final_action_strings = policy.final_actions.copy()
    else:
        final_action_strings = policy.final_actions.copy()

    initial_actions = build_action_items(
        initial_action_strings, exposure,
        apply_policy(initial_policy_input).action_reasons,
        initial_probability=pre_contact_analysis["fraud_probability"] if evidence_requests else None,
    )
    final_actions = build_action_items(final_action_strings, exposure, policy.action_reasons)

    initial_names = set(initial_action_strings)
    final_names = set(final_action_strings)
    added = final_names - initial_names
    removed = initial_names - final_names

    if evidence_requests and (added or removed):
        if customer_response:
            what_changed = (
                f"Customer response: '{customer_response.get('response_text', 'no response')}'. "
                f"Added: {', '.join(added) if added else 'none'}. "
                f"Removed: {', '.join(removed) if removed else 'none'}."
            )
        else:
            what_changed = (
                f"Added: {', '.join(added) if added else 'none'}. "
                f"Removed: {', '.join(removed) if removed else 'none'}."
            )
    else:
        if customer_response and customer_response.get("responded"):
            disposition = "confirmed unauthorized activity" if customer_response.get("confirmed_fraud") else "confirmed the transaction was legitimate"
            what_changed = f"Customer validation {disposition}. Actions remained the same."
        elif customer_response and customer_response.get("responded") is False:
            what_changed = "No customer response was received within 24 hours; actions remained the same."
        else:
            what_changed = "nothing"

    # A recorded response is authoritative for the transition narrative even
    # when the action sets happen to be identical.
    if customer_response and customer_response.get("responded"):
        disposition = "confirmed unauthorized activity" if customer_response.get("confirmed_fraud") else "confirmed the transaction was legitimate"
        what_changed = f"Customer validation {disposition}; final actions reflect the settled case outcome."

    # STEP_UP_AUTH evidence request (was recommended but never actually
    # captured as an evidence-gathering step)
    all_actions = set(initial_action_strings) | set(policy.final_actions)
    if "STEP_UP_AUTH" in all_actions:
        if customer_response and customer_response.get("confirmed_fraud"):
            assumed = "Step-up authentication challenge failed / not completed"
        else:
            assumed = "Step-up authentication completed successfully"
        evidence_requests.append(EvidenceRequest(
            type="step_up_auth",
            asked_after_step=step_counter,
            assumed_response=assumed
        ))

    print(f"  [POLICY] Initial={initial_action_strings}")
    print(f"  [POLICY] Final={policy.final_actions}")

    # Step 5 — Assemble final evidence deterministically from retrieved data.
    # Do not export LLM-authored factual claims into a scored case file.
    evidence_items = build_grounded_evidence(
        bundle, txn, analysis, customer_response, trigger_risk_score
    )

    # Analyst-provided context becomes explicit EXTERNAL evidence
    if trigger_type == "analyst_request" and case.get("trigger_text"):
        evidence_items.append(EvidenceItem(
            claim=f"Analyst-provided context: {case['trigger_text']}",
            source=EvidenceSource.EXTERNAL,
            ref="analyst_request",
            entity_ids=[case_id]
        ))

    # Evidence records are retained for the analyst; §6 separately uses
    # policy_facts["independent_support_count"], not this list's length.

    # Keep only real, graph-returned cards and remove the investigated card
    # before any JSON or SAR field is assembled.
    valid_customer_cards = {
        str(card.get("card_id")) for card in bundle.get("cards", [])
        if card.get("card_id")
    }
    analysis["connected_card_ids"] = sorted({
        str(card_id) for card_id in analysis.get("connected_card_ids", [])
        if str(card_id) in valid_customer_cards and str(card_id) != str(case["card_id"])
    })

    # Fix #5: connected_device_profiles must come from validated graph device
    # data, not directly from the LLM's list. The only device profile string
    # this bundle can actually confirm is the one returned by device_neighbors
    # for the flagged transaction — the LLM may suggest others, but they are
    # dropped unless they match a validated value.
    validated_device_key = str(bundle.get("device_data", {}).get("device_key") or "")
    valid_device_profiles = {validated_device_key} if validated_device_key else set()
    analysis["connected_device_profiles"] = sorted({
        str(profile) for profile in analysis.get("connected_device_profiles", [])
        if str(profile) in valid_device_profiles
    })

    # Step 6 — SAR
    sar_narrative = ""
    sar_subjects = []
    sar_dates = []
    sar_amount = 0.0

    if policy.file_sar:
        print(f"  [SAR] Building factual narrative...")
        sar_narrative = build_sar_narrative(
            case, txn, analysis, exposure, final_actions, customer_response
        )
        sar_subjects = list(dict.fromkeys(
            [case["customer_id"], case["card_id"]] + analysis.get("connected_card_ids", [])
        ))
        sar_amount = float(exposure)
        sar_dates = derive_sar_activity_dates(analysis, txn, bundle, case)

    final_summary = build_final_summary(
        case, txn, analysis, final_actions, customer_response, trigger_risk_score
    )

    status_map = {
        "fraud": CaseStatus.CLOSED_FRAUD,
        "legitimate": CaseStatus.CLOSED_LEGITIMATE,
        "uncertain": CaseStatus.ESCALATED if "ESCALATE_TO_ANALYST" in policy.final_actions else CaseStatus.OPEN
    }
    # Status is a deterministic consequence of the final verdict and policy,
    # not an unvalidated LLM field.  This prevents combinations such as
    # closed_legitimate + uncertain.
    status = status_map[analysis["verdict"]]

    # Step 7 — Final graph write (progresses the case opened in Step 2c)
    written_to_graph = gc.write_investigation_case(
        case_id=case_id, customer_id=case["customer_id"], card_id=case["card_id"],
        opened_at=case["opened_at"], status=status.value, verdict=analysis["verdict"],
        fraud_probability=analysis["fraud_probability"], pattern=analysis["pattern"],
        exposure_usd=float(exposure), summary=final_summary
    )

    elapsed = round(time.time() - start_time, 1)

    independent_support_count = policy_facts["independent_support_count"]
    if customer_response and customer_response.get("responded"):
        if customer_response.get("confirmed_fraud"):
            stop_reason = "Customer validation confirmed unauthorized activity; the case is closed as fraud under R2."
        else:
            stop_reason = "Customer validation confirmed the transaction was legitimate; the case is closed under R3."
    elif customer_response and customer_response.get("responded") is False:
        stop_reason = "Customer validation received no response within 24 hours; the case remains open and actions follow R4."
    elif (analysis["fraud_probability"] >= STOP_HIGH or analysis["fraud_probability"] <= STOP_LOW) and independent_support_count >= 2:
        stop_reason = (
            f"§6 stopping threshold met: {independent_support_count} independent indicators support "
            f"a {analysis['verdict']} verdict (P={analysis['fraud_probability']})."
        )
    else:
        stop_reason = "Investigation remains open because additional evidence is needed before a defensible decision."

    answer = CaseAnswer(
        case_id=case_id,
        case=Case(
            status=status,
            verdict=Verdict(analysis["verdict"]),
            fraud_probability=analysis["fraud_probability"],
            pattern=analysis["pattern"],
            pattern_description=analysis.get("pattern_description", ""),
            affected_txn_ids=[str(x) for x in analysis.get("affected_txn_ids", [])]
                if analysis["verdict"] != "legitimate" else [],
            first_suspicious_txn_id=str(analysis.get("first_suspicious_txn_id", ""))
                if analysis["verdict"] != "legitimate" else "",
            # This field is for *other* cards linked to the compromise. The
            # investigated card itself is never a connected card.
            connected_card_ids=analysis["connected_card_ids"],
            connected_device_profiles=[str(x) for x in analysis.get("connected_device_profiles", [])],
            exposure_usd=float(exposure) if analysis["verdict"] != "legitimate" else 0.0,
            evidence=evidence_items,
            similar_prior_cases=[str(x) for x in bundle.get("prior_case_ids", [])],
            summary=final_summary,
            written_to_graph=written_to_graph,
            graph_case_id=case_id if written_to_graph else ""
        ),
        evidence_requests=evidence_requests,
        next_best_actions=NextBestActions(
            initial=initial_actions,
            final=final_actions,
            what_changed=what_changed
        ),
        sar=SAR(
            file=policy.file_sar,
            reason=(
                policy.sar_reason if policy.file_sar
                else build_no_sar_reason(policy, analysis["verdict"])
            ),
            narrative=sar_narrative,
            subjects=sar_subjects,
            total_amount_usd=sar_amount,
            activity_dates=sar_dates
        ),
        stop_reason=stop_reason,
        tool_calls=gc.TOOL_CALLS,
        tokens=_tokens,
        latency_s=elapsed
    )

    # Step 7b — LLM #2 reviewer (diagnostic only) then the deterministic
    # final validator (authoritative). Neither can invent a fact; the
    # validator only ever corrects from facts already established upstream.
    print(f"  [REVIEW] Running LLM reviewer + deterministic validator...")
    review = review_case_with_llm(answer, policy, policy_facts, customer_response)
    answer, validator_corrections = deterministic_final_validate(answer, policy, evidence_requests, review)
    if review["issues"]:
        print(f"  [REVIEW] Reviewer issues ({review['severity']}): {review['issues']}")
    if validator_corrections:
        print(f"  [REVIEW] Validator corrections: {validator_corrections}")
    answer.tokens = _tokens
    answer.qa = QASummary(
        reviewer_severity=review["severity"],
        reviewer_issue_count=len(review["issues"]),
        validator_correction_count=len(validator_corrections),
        clean=(not review["issues"] and not validator_corrections),
    )

    # Step 8 — Persist an auditable graph lifecycle, not only a JSON export.
    # The event payloads are intentionally compact JSON so they remain easy to
    # inspect in GraphStudio and can be replayed by the dashboard.
    event_time = case["opened_at"]
    lifecycle_events = [
        ("evidence", {
            "evidence": [item.model_dump() for item in answer.case.evidence],
            "evidence_requests": [item.model_dump() for item in answer.evidence_requests],
        }),
        ("recommendations", {
            "initial": [item.model_dump() for item in answer.next_best_actions.initial],
            "final": [item.model_dump() for item in answer.next_best_actions.final],
            "what_changed": answer.next_best_actions.what_changed,
        }),
        ("decision", {
            "status": status.value,
            "verdict": analysis["verdict"],
            "fraud_probability": analysis["fraud_probability"],
            "stop_reason": answer.stop_reason,
            "sar_file": answer.sar.file,
        }),
        ("review", {
            "reviewer_issues": review["issues"],
            "reviewer_severity": review["severity"],
            "reviewer_requires_revalidation": review["requires_deterministic_revalidation"],
            "validator_corrections": validator_corrections,
        }),
    ]
    for event_type, payload in lifecycle_events:
        event_id = f"{case_id}:{event_type}"
        if not gc.write_investigation_event(
            case_id, event_id, event_type, json.dumps(payload), event_time
        ):
            print(f"  [WARN] Graph lifecycle event not written: {event_id}")

    # Lifecycle writes are real MCP calls and belong in the reported count.
    answer.tool_calls = gc.TOOL_CALLS

    # Step 9 — Update semantic case memory so a later case can retrieve this
    # investigation, including reruns of the same case without duplicate rows.
    try:
        from graphrag import remember_investigation
        remember_investigation(answer)
    except Exception as e:
        # GraphRAG failure must not discard an otherwise valid investigation.
        print(f"  [WARN] Case-memory update failed: {e}")

    print(f"  ✅ Done in {elapsed}s | {gc.TOOL_CALLS} tool calls | {_tokens} tokens")
    return answer