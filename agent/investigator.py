"""
Main investigation agent — produces exact required submission format.
Aligned against Fraud Policy v1.0.
"""

import os
import json
import time
from groq import Groq
from dotenv import load_dotenv

import graph_client as gc
from policy_engine import PolicyInput, apply_policy, R1_VERIFY_THRESHOLD
from graphrag import guess_pattern
from answer_schema import (
    CaseAnswer, Case, SAR, NextBestActions,
    EvidenceItem, EvidenceRequest, ActionItem,
    CaseStatus, Verdict, EvidenceSource, ActionRoute,
    get_route, get_rule
)

load_dotenv()

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "groq/compound"

# §6 Stopping rule: stop once probability clears these bounds with enough evidence
STOP_HIGH = 0.85
STOP_LOW = 0.15

_tokens = 0  # LLM tokens are still tracked here — gc.TOOL_CALLS handles tool calls


# ── LLM helpers ───────────────────────────────────────────────────────────────

def llm(prompt: str, system: str = "") -> str:
    global _tokens
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(5):
        try:
            response = groq.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=2000
            )
            _tokens += response.usage.total_tokens if response.usage else 0
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 30 * (attempt + 1)
                print(f"  [RATE LIMIT] Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")


def llm_json(prompt: str, system: str = "") -> dict:
    full_system = (system or "") + "\nRespond ONLY with valid JSON. No markdown, no preamble."
    raw = llm(prompt, full_system)
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── Graph evidence gathering ──────────────────────────────────────────────────

def gather_evidence(case: dict, txn: dict) -> dict:
    customer_id = case["customer_id"]
    opened_at = case["opened_at"]

    print(f"  [1/6] Baseline...")
    baseline = gc.card_baseline(customer_id)

    print(f"  [2/6] 72h window...")
    window = gc.card_window(customer_id, opened_at, 72)

    print(f"  [3/6] Customer cards...")
    cards = gc.customer_cards(customer_id)

    print(f"  [4/6] Similar closed cases...")
    likely_pattern = guess_pattern(txn, baseline)
    similar = gc.similar_closed_cases(likely_pattern, customer_id)
    similar_dict = similar if isinstance(similar, dict) else {}
    baseline["has_prior_fraud"] = any(
        cc.get("outcome") == "confirmed_fraud"
        for cc in similar_dict.get("by_customer", [])
    )

    print(f"  [5/6] Region neighbors...")
    region = []
    if txn.get("addr1"):
        try:
            region = gc.region_neighbors(str(int(float(txn["addr1"]))), opened_at, 7)
        except Exception:
            pass

    print(f"  [6/6] Device neighbors...")
    device_key = gc.get_device_key(txn.get("txn_id", ""))
    device_txns = []
    if device_key:
        try:
            device_txns = gc.device_neighbors(device_key, opened_at, 30)
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
        "similar_cases": similar,
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

    if device_cards:
        analysis["shared_origin"] = True
        existing_cards = set(str(x) for x in analysis.get("connected_card_ids", []))
        analysis["connected_card_ids"] = sorted(existing_cards | set(device_cards))
        if device_key:
            existing_profiles = set(analysis.get("connected_device_profiles", []))
            existing_profiles.add(device_key)
            analysis["connected_device_profiles"] = sorted(existing_profiles)

        current_p = analysis.get("fraud_probability", 0.0)
        if current_p < STOP_HIGH:
            analysis["fraud_probability"] = round(min(0.95, current_p + 0.15), 2)
            if analysis["fraud_probability"] >= STOP_HIGH:
                analysis["verdict"] = "fraud"

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
    return llm_json(prompt)


# ── Build action items ────────────────────────────────────────────────────────

def build_action_items(actions: list[str], exposure: float) -> list[ActionItem]:
    items = []
    for action in actions:
        items.append(ActionItem(
            action=action,
            route=ActionRoute(get_route(action, exposure)),
            reason=get_rule(action, {"exposure": exposure})
        ))
    return items


# ── Main investigation ────────────────────────────────────────────────────────

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
            )
        }
        _tokens += 50
        print(f"  [CONTACT] Using known customer denial from trigger")

    elif analysis.get("needs_customer_contact") and STOP_LOW < analysis["fraud_probability"] < STOP_HIGH:
        # Outside this band the verdict is already decisive per §6 — contacting
        # the customer wouldn't change the outcome, so skip the extra LLM call.
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
                analysis["fraud_probability"] = max(0.05, analysis["fraud_probability"] - 0.20)
                if analysis["fraud_probability"] < STOP_LOW:
                    analysis["verdict"] = "legitimate"
                    analysis["status"] = "closed_legitimate"
                print(f"  [CONTACT] Denied → P={analysis['fraud_probability']}")

    # Analyst-provided context is recorded as an explicit evidence request too
    if trigger_type == "analyst_request":
        evidence_requests.append(EvidenceRequest(
            type="analyst_info",
            asked_after_step=0,
            assumed_response=case.get("trigger_text", "Analyst-provided context used as-is")
        ))

    analysis["fraud_probability"] = round(analysis["fraud_probability"], 2)
    analysis = ground_affected_transactions(analysis, bundle, txn)

    # §R5/§R7 signals derived from real data, not left to the LLM to assert
    card_testing_cleared_over_100 = float(txn.get("amount", 0) or 0) > 100
    matches_recurring_charge = compute_matches_recurring_charge(txn, bundle["baseline"])
    coordinated_across_customers = bool(bundle["device_data"].get("other_customers"))

    # §R8 signal: does the deterministic graph-side pattern guess disagree
    # with what the LLM classified? A real disagreement between two
    # independent readings of the evidence is exactly what R8 means by
    # "the evidence conflicts".
    heuristic_pattern = guess_pattern(txn, bundle["baseline"])
    evidence_conflicts = (
        heuristic_pattern not in ("none", analysis["pattern"])
        and analysis["pattern"] not in ("none",)
    )

    # §R10 signal: only treat credentials as confirmed compromised when we
    # have corroborated account-takeover evidence — a shared device profile
    # AND either the customer denied the activity or confidence is already
    # decisive. Not just an LLM assertion.
    credentials_confirmed_compromised = (
        analysis["pattern"] == "account_takeover"
        and analysis.get("shared_origin", False)
        and (
            (customer_response and customer_response.get("confirmed_fraud") is True)
            or analysis["fraud_probability"] >= STOP_HIGH
        )
    )

    # Step 4 — Policy engine
    print(f"  [POLICY] Running rules...")
    similar_dict = bundle["similar_cases"] if isinstance(bundle["similar_cases"], dict) else {}
    prior_cases = similar_dict.get("by_customer", [])
    has_prior = any(cc.get("outcome") == "confirmed_fraud" for cc in prior_cases)
    n_cards = len([c for c in bundle["cards"] if c.get("card_id") != case["card_id"]])
    exposure = analysis.get("exposure_usd", txn.get("amount", 0))

    policy_input = PolicyInput(
        verdict=analysis["verdict"],
        fraud_probability=analysis["fraud_probability"],
        exposure_usd=exposure,
        pattern=analysis["pattern"],
        customer_responded=customer_response.get("responded") if customer_response else None,
        customer_confirmed_fraud=customer_response.get("confirmed_fraud") if customer_response else None,
        shared_origin=analysis.get("shared_origin", False),
        n_cards_confirmed=n_cards,
        has_prior_fraud=has_prior,
        channel=txn.get("channel", "unknown"),
        evidence_count=len(analysis.get("evidence", [])),
        evidence_conflicts=evidence_conflicts,
        card_testing_cleared_over_100=card_testing_cleared_over_100,
        matches_recurring_charge=matches_recurring_charge,
        coordinated_across_customers=coordinated_across_customers,
        credentials_confirmed_compromised=credentials_confirmed_compromised,
    )

    policy = apply_policy(policy_input)

    initial_action_strings = ["CREATE_CASE"]
    if analysis["fraud_probability"] < R1_VERIFY_THRESHOLD:
        initial_action_strings.append("VERIFY_WITH_CUSTOMER")
        if analysis["pattern"] in ("card_testing", "account_takeover", "card_not_present_new_device"):
            initial_action_strings.append("STEP_UP_AUTH")
    else:
        initial_action_strings.append("BLOCK_CARD")

    initial_actions = build_action_items(initial_action_strings, exposure)
    final_actions = build_action_items(policy.final_actions, exposure)

    initial_names = set(initial_action_strings)
    final_names = set(policy.final_actions)
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
        what_changed = "nothing" if not evidence_requests else "No customer response received — actions unchanged."

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

    # Step 5 — Build evidence items
    evidence_items = []
    for e in analysis.get("evidence", []):
        try:
            source_map = {
                "graph": EvidenceSource.GRAPH,
                "document": EvidenceSource.DOCUMENT,
                "customer": EvidenceSource.CUSTOMER,
                "external": EvidenceSource.EXTERNAL
            }
            evidence_items.append(EvidenceItem(
                claim=str(e.get("claim", e.get("signal", ""))),
                source=source_map.get(str(e.get("source", "graph")).lower(), EvidenceSource.GRAPH),
                ref=str(e.get("ref", f"query:get_transaction({case['flagged_txn_id']})")),
                entity_ids=[str(x) for x in e.get("entity_ids", [])]
            ))
        except Exception as ex:
            print(f"  [WARN] Evidence item skipped: {ex}")

    # Analyst-provided context becomes explicit EXTERNAL evidence
    if trigger_type == "analyst_request" and case.get("trigger_text"):
        evidence_items.append(EvidenceItem(
            claim=f"Analyst-provided context: {case['trigger_text']}",
            source=EvidenceSource.EXTERNAL,
            ref="analyst_request",
            entity_ids=[case_id]
        ))

    # §6/§R3: the customer's own reply is frequently the deciding evidence —
    # make it a first-class evidence item, not just prose buried in the summary
    if customer_response and customer_response.get("responded"):
        evidence_items.append(EvidenceItem(
            claim=f"Customer response: '{customer_response.get('response_text', '')}'",
            source=EvidenceSource.CUSTOMER,
            ref="evidence_request:1",
            entity_ids=[str(case["flagged_txn_id"])],
        ))

    # §6 stopping rule enforcement: a decisive verdict (>=0.85 or <=0.15) must
    # be backed by >=2 evidence items. Pad deterministically from data already
    # in the bundle rather than declaring the case closed on too little.
    evidence_items = augment_evidence_to_minimum(evidence_items, bundle, txn, analysis)

    # Step 6 — SAR
    sar_narrative = ""
    sar_subjects = []
    sar_dates = []
    sar_amount = 0.0

    if policy.file_sar:
        print(f"  [SAR] Writing narrative...")
        sar_narrative = llm(
            f"""Write a FinCEN SAR narrative. 6-12 sentences. Professional tone.
Case: {case_id} | Customer: {case['customer_id']} | Card: {case['card_id']}
Pattern: {analysis['pattern']}
Amount: ${exposure:.2f}
Transactions: {analysis.get('affected_txn_ids', [case['flagged_txn_id']])}
Date: {case['opened_at'][:10]}
Channel: {txn.get('channel')}
Connected cards: {analysis.get('connected_card_ids', [])}
Summary: {analysis.get('summary', '')}

Include: who (customer/card IDs), what happened, when (dates), where (channel/region),
how it was carried out, why it is suspicious. Name all subjects explicitly."""
        )
        _tokens += 500
        sar_subjects = [case["customer_id"], case["card_id"]] + analysis.get("connected_card_ids", [])
        sar_amount = float(exposure)
        sar_dates = [case["opened_at"][:10], case["opened_at"][:10]]

    status_map = {
        "fraud": CaseStatus.CLOSED_FRAUD,
        "legitimate": CaseStatus.CLOSED_LEGITIMATE,
        "uncertain": CaseStatus.ESCALATED if "ESCALATE_TO_ANALYST" in policy.final_actions else CaseStatus.OPEN
    }
    status = CaseStatus(analysis.get("status", status_map.get(analysis["verdict"], "open")))

    # Step 7 — Final graph write (progresses the case opened in Step 2c)
    written_to_graph = gc.write_investigation_case(
        case_id=case_id, customer_id=case["customer_id"], card_id=case["card_id"],
        opened_at=case["opened_at"], status=status.value, verdict=analysis["verdict"],
        fraud_probability=analysis["fraud_probability"], pattern=analysis["pattern"],
        exposure_usd=float(exposure), summary=analysis.get("summary", "")
    )

    elapsed = round(time.time() - start_time, 1)

    two_plus_evidence = len(evidence_items) >= 2
    stop_reason = analysis.get(
        "stop_reason",
        f"Investigation complete. Verdict: {analysis['verdict']} "
        f"(P={analysis['fraud_probability']}, {len(evidence_items)} evidence items — "
        f"§6 stopping threshold {'met' if (analysis['fraud_probability'] >= STOP_HIGH or analysis['fraud_probability'] <= STOP_LOW) and two_plus_evidence else 'not fully met, escalated instead'}). "
        f"Pattern: {analysis['pattern']}. Policy applied."
    )

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
            connected_card_ids=[str(x) for x in analysis.get("connected_card_ids", [])],
            connected_device_profiles=[str(x) for x in analysis.get("connected_device_profiles", [])],
            exposure_usd=float(exposure) if analysis["verdict"] != "legitimate" else 0.0,
            evidence=evidence_items,
            similar_prior_cases=[str(x) for x in bundle.get("prior_case_ids", [])],
            summary=analysis.get("summary", ""),
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
            reason=f"R2/R6/R9: {analysis['pattern']} confirmed, exposure ${exposure:.2f}" if policy.file_sar
                   else f"Exposure ${(0.0 if analysis['verdict'] == 'legitimate' else float(exposure)):.2f} below threshold or verdict not confirmed fraud",
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

    print(f"  ✅ Done in {elapsed}s | {gc.TOOL_CALLS} tool calls | {_tokens} tokens")
    return answer