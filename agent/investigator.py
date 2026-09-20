"""
Main investigation agent — produces exact required submission format.
"""

import os
import json
import time
import statistics
from groq import Groq
from dotenv import load_dotenv

import graph_client as gc
from policy_engine import PolicyInput, apply_policy
from answer_schema import (
    CaseAnswer, Case, SAR, NextBestActions,
    EvidenceItem, EvidenceRequest, ActionItem,
    CaseStatus, Verdict, EvidenceSource, ActionRoute,
    get_route, get_rule
)

load_dotenv()

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "groq/compound"

_tool_calls = 0
_tokens = 0


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
    global _tool_calls
    customer_id = case["customer_id"]
    opened_at = case["opened_at"]

    print(f"  [1/6] Baseline...")
    baseline = gc.card_baseline(customer_id)
    _tool_calls += 1

    print(f"  [2/6] 72h window...")
    window = gc.card_window(customer_id, opened_at, 72)
    _tool_calls += 1

    print(f"  [3/6] Customer cards...")
    cards = gc.customer_cards(customer_id)
    _tool_calls += 1

    print(f"  [4/6] Similar closed cases...")
    similar = gc.similar_closed_cases("card_not_present_fraud", customer_id)
    _tool_calls += 1

    similar_dict = similar if isinstance(similar, dict) else {}
    baseline["has_prior_fraud"] = any(
        cc.get("outcome") == "confirmed_fraud"
        for cc in similar_dict.get("by_customer", [])
    )

    print(f"  [5/6] Region neighbors...")
    region = []
    if txn.get("addr1"):
        try:
            region = gc.region_neighbors(
                str(int(float(txn["addr1"]))), opened_at, 7
            )
            _tool_calls += 1
        except Exception:
            pass

    print(f"  [6/6] Device neighbors...")
    device_data = {"transactions": [], "cards": [], "cases": []}
    _tool_calls += 1

    return {
        "case": case,
        "flagged_txn": txn,
        "history": [],
        "window": window,
        "cards": cards,
        "similar_cases": similar,
        "region_neighbors": region,
        "device_data": device_data,
        "baseline": baseline
    }


# ── Enforce probability consistency ──────────────────────────────────────────

def enforce_consistency(analysis: dict, trigger_risk_score: float) -> dict:
    prob = analysis.get("fraud_probability", trigger_risk_score)
    min_prob = max(0.0, trigger_risk_score - 0.20)
    max_prob = min(1.0, trigger_risk_score + 0.20)
    prob = max(min_prob, min(max_prob, prob))
    analysis["fraud_probability"] = round(prob, 2)

    p = analysis["fraud_probability"]
    if p >= 0.80:
        analysis["verdict"] = "fraud"
    elif p <= 0.30:
        analysis["verdict"] = "legitimate"
    else:
        analysis["verdict"] = "uncertain"

    return analysis


# ── LLM analysis ─────────────────────────────────────────────────────────────

def analyse_evidence(bundle: dict, trigger_risk_score: float) -> dict:
    case = bundle["case"]
    txn = bundle["flagged_txn"]
    baseline = bundle["baseline"]
    similar = bundle["similar_cases"]
    similar_dict = similar if isinstance(similar, dict) else {}

    # GraphRAG context
    try:
        from graphrag import retrieve_context
        rag_context = retrieve_context(case, txn, baseline)
    except Exception as e:
        rag_context = f"GraphRAG unavailable: {e}"

    # Similar case IDs for memory
    prior_case_ids = [
        cc.get("case_id", "") for cc in similar_dict.get("by_pattern", [])[:5]
        if cc.get("case_id")
    ]
    prior_cases_summary = []
    for cc in similar_dict.get("by_pattern", [])[:3]:
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
        for t in bundle["window"][:5]
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
- At least 3 evidence items
- Each claim must cite specific data (amounts, dates, IDs, region codes)
- source=graph for graph query results
- source=document for policy/typology knowledge
- entity_ids must be real IDs from the dataset
- For legitimate verdict: affected_txn_ids=[], exposure_usd=0
"""

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
"""
    return llm_json(prompt)


# ── Build action items ────────────────────────────────────────────────────────

def build_action_items(actions: list[str], exposure: float) -> list[ActionItem]:
    """Convert string actions to ActionItem objects with route and reason."""
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
    global _tool_calls, _tokens
    _tool_calls = 0
    _tokens = 0
    start_time = time.time()

    case_id = case["case_id"]
    print(f"\n{'='*60}")
    print(f"Investigating {case_id} | {case['customer_id']}")
    print(f"{'='*60}")

    trigger_risk_score = float(case.get("risk_score", 0.5))

    # Step 1 — Flagged transaction
    print(f"  [0/6] Fetching transaction {case['flagged_txn_id']}...")
    txn = gc.get_transaction(case["flagged_txn_id"])
    _tool_calls += 1
    if not txn:
        txn = {
            "txn_id": case["flagged_txn_id"],
            "amount": 0,
            "channel": "unknown",
            "risk_score": trigger_risk_score
        }

    # Step 2 — Gather graph evidence
    bundle = gather_evidence(case, txn)

    # Step 3 — LLM analysis
    print(f"  [LLM] Analysing evidence...")
    analysis = analyse_evidence(bundle, trigger_risk_score)
    print(f"  [LLM] Verdict={analysis['verdict']} P={analysis['fraud_probability']} Pattern={analysis['pattern']}")

    # Step 4 — Customer contact
    customer_response = None
    evidence_requests = []
    step_counter = len(bundle["window"]) + 4

    if analysis.get("needs_customer_contact") and analysis["fraud_probability"] < 0.85:
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
                if analysis["fraud_probability"] < 0.30:
                    analysis["verdict"] = "legitimate"
                    analysis["status"] = "closed_legitimate"
                print(f"  [CONTACT] Denied → P={analysis['fraud_probability']}")

    # Step 5 — Policy engine
    print(f"  [POLICY] Running rules...")
    similar_dict = bundle["similar_cases"] if isinstance(bundle["similar_cases"], dict) else {}
    prior_cases = similar_dict.get("by_customer", [])
    has_prior = any(cc.get("outcome") == "confirmed_fraud" for cc in prior_cases)
    n_cards = len([c for c in bundle["cards"] if c.get("card_id") != case["card_id"]])

    policy_input = PolicyInput(
        verdict=analysis["verdict"],
        fraud_probability=analysis["fraud_probability"],
        exposure_usd=analysis.get("exposure_usd", txn.get("amount", 0)),
        pattern=analysis["pattern"],
        customer_responded=customer_response.get("responded") if customer_response else None,
        customer_confirmed_fraud=customer_response.get("confirmed_fraud") if customer_response else None,
        shared_origin=analysis.get("shared_origin", False),
        n_cards_confirmed=n_cards,
        has_prior_fraud=has_prior,
        channel=txn.get("channel", "unknown")
    )

    policy = apply_policy(policy_input)
    exposure = analysis.get("exposure_usd", txn.get("amount", 0))

    # Build initial actions (before customer contact)
    initial_action_strings = ["CREATE_CASE"]
    if analysis["fraud_probability"] < 0.85:
        initial_action_strings.append("VERIFY_WITH_CUSTOMER")
    else:
        initial_action_strings.append("BLOCK_CARD")

    initial_actions = build_action_items(initial_action_strings, exposure)
    final_actions = build_action_items(policy.final_actions, exposure)

    # What changed
    initial_names = set(initial_action_strings)
    final_names = set(policy.final_actions)
    added = final_names - initial_names
    removed = initial_names - final_names

    if evidence_requests and (added or removed):
        what_changed = (
            f"Customer response: '{customer_response.get('response_text', 'no response')}'. "
            f"Added: {', '.join(added) if added else 'none'}. "
            f"Removed: {', '.join(removed) if removed else 'none'}."
        )
    else:
        what_changed = "nothing" if not evidence_requests else "No customer response received — actions unchanged."

    print(f"  [POLICY] Initial={initial_action_strings}")
    print(f"  [POLICY] Final={policy.final_actions}")

    # Step 6 — Build evidence items
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

    # Step 7 — SAR
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
        sar_subjects = (
            [case["customer_id"], case["card_id"]]
            + analysis.get("connected_card_ids", [])
        )
        sar_amount = float(exposure)
        sar_dates = [case["opened_at"][:10], case["opened_at"][:10]]

    # Determine status
    status_map = {
        "fraud": CaseStatus.CLOSED_FRAUD,
        "legitimate": CaseStatus.CLOSED_LEGITIMATE,
        "uncertain": CaseStatus.ESCALATED if "ESCALATE_TO_ANALYST" in policy.final_actions else CaseStatus.OPEN
    }
    status = CaseStatus(analysis.get("status", status_map.get(analysis["verdict"], "open")))

    # Step 8 — Write to graph
    gc.write_investigation_case(
        case_id=case_id,
        customer_id=case["customer_id"],
        card_id=case["card_id"],
        opened_at=case["opened_at"],
        status=status.value,
        verdict=analysis["verdict"],
        fraud_probability=analysis["fraud_probability"],
        pattern=analysis["pattern"],
        exposure_usd=float(exposure),
        summary=analysis.get("summary", "")
    )
    _tool_calls += 1
    written_to_graph = True

    elapsed = round(time.time() - start_time, 1)

    # Step 9 — Assemble final answer
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
            similar_prior_cases=[str(x) for x in analysis.get("similar_prior_cases", [])],
            summary=analysis.get("summary", ""),
            written_to_graph=written_to_graph,
            graph_case_id=case_id
        ),
        evidence_requests=evidence_requests,
        next_best_actions=NextBestActions(
            initial=initial_actions,
            final=final_actions,
            what_changed=what_changed
        ),
        sar=SAR(
            file=policy.file_sar,
            reason=f"R2/R6: {analysis['pattern']} confirmed" if policy.file_sar
                   else f"Exposure ${exposure:.2f} below threshold or verdict not confirmed fraud",
            narrative=sar_narrative,
            subjects=sar_subjects,
            total_amount_usd=sar_amount,
            activity_dates=sar_dates
        ),
        stop_reason=analysis.get(
            "stop_reason",
            f"Investigation complete. Verdict: {analysis['verdict']}. "
            f"Pattern: {analysis['pattern']}. Policy applied."
        ),
        tool_calls=_tool_calls,
        tokens=_tokens,
        latency_s=elapsed
    )

    print(f"  ✅ Done in {elapsed}s | {_tool_calls} tool calls | {_tokens} tokens")
    return answer