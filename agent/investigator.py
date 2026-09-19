"""
Main investigation agent.
Orchestrates: graph_client → LLM reasoning → policy_engine → answer_schema
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
    CaseAnswer, CaseRecord, SAR, NextBestActions,
    EvidenceItem, EvidenceRequest, Verdict, Pattern
)

load_dotenv()

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "groq/compound"


# ── LLM helpers ──────────────────────────────────────────────────────────────

import time

def llm(prompt: str, system: str = "") -> str:
    """Single LLM call with retry on rate limit."""
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
                max_tokens=1500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 30 * (attempt + 1)
                print(f"  [RATE LIMIT] Waiting {wait}s before retry {attempt+1}/5...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded on rate limit")


def llm_json(prompt: str, system: str = "") -> dict:
    """LLM call expecting JSON response."""
    full_system = (system or "") + "\nRespond ONLY with valid JSON. No markdown, no explanation."
    raw = llm(prompt, full_system)
    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── Evidence gathering ────────────────────────────────────────────────────────

def gather_evidence(case: dict, txn: dict) -> dict:
    """
    Pull all evidence from the graph for a given case.
    Returns a structured evidence bundle.
    """
    customer_id = case["customer_id"]
    card_id = case["card_id"]
    txn_id = case["flagged_txn_id"]
    opened_at = case["opened_at"]

    print(f"  [1/5] Fetching card history for {customer_id}...")
    history = gc.card_history(customer_id)

    print(f"  [2/5] Fetching 72h window around flagged transaction...")
    window = gc.card_window(customer_id, opened_at, 72)

    print(f"  [3/5] Fetching customer cards...")
    cards = gc.customer_cards(customer_id)

    print(f"  [4/5] Searching similar closed cases...")
    similar = gc.similar_closed_cases("card_not_present_fraud", customer_id)

    print(f"  [5/5] Checking region neighbors...")
    region = []
    if txn.get("addr1"):
        region = gc.region_neighbors(
            str(int(txn["addr1"])), opened_at, 7
        )

    # Compute baseline stats from history
    baseline = {}
    if history:
        amounts = [t["amount"] for t in history]
        channels = [t.get("channel", "") for t in history]
        regions = [str(int(t.get("addr1", 0))) for t in history if t.get("addr1")]
        baseline = {
            "total_txns": len(history),
            "avg_amount": round(statistics.mean(amounts), 2),
            "max_amount": round(max(amounts), 2),
            "usual_channels": list(set(channels)),
            "usual_regions": list(set(regions))[:10],
            "has_prior_fraud": any(
                t.get("risk_score", 0) > 0.8 for t in history
            )
        }

    return {
        "case": case,
        "flagged_txn": txn,
        "history": history,
        "window": window,
        "cards": cards,
        "similar_cases": similar,
        "region_neighbors": region,
        "baseline": baseline
    }


# ── LLM analysis ─────────────────────────────────────────────────────────────

def analyse_evidence(bundle: dict) -> dict:
    """
    Ask the LLM to reason over the evidence bundle.
    Returns structured analysis JSON.
    """
    case = bundle["case"]
    txn = bundle["flagged_txn"]
    baseline = bundle["baseline"]
    similar = bundle["similar_cases"]

    # Summarise similar cases for the prompt
    similar_summary = []
    for cc in similar["by_pattern"][:3]:
        similar_summary.append({
            "case_id": cc.get("case_id"),
            "outcome": cc.get("outcome"),
            "pattern": cc.get("pattern"),
            "exposure": cc.get("exposure_usd"),
            "notes": cc.get("analyst_notes", "")[:200]
        })

    prompt = f"""
You are a fraud analyst at a bank. Investigate this flagged transaction.

CASE: {case['case_id']}
Customer: {case['customer_id']} | Card: {case['card_id']}
Opened: {case['opened_at']}
Trigger: {case['trigger_text']}

FLAGGED TRANSACTION:
- ID: {txn.get('txn_id')}
- Amount: ${txn.get('amount')}
- Channel: {txn.get('channel')}
- Risk Score: {txn.get('risk_score')}
- Billing Region (addr1): {txn.get('addr1')}
- ProductCD: {txn.get('product_cd')}
- Days since last txn (D1): {txn.get('d1')}
- Match flags: M1={txn.get('m1')} M2={txn.get('m2')} M3={txn.get('m3')} M5={txn.get('m5')}
- Email match: p_email={txn.get('p_email') or 'none'}

CUSTOMER BASELINE ({baseline.get('total_txns', 0)} transactions):
- Avg amount: ${baseline.get('avg_amount', 0)}
- Max amount: ${baseline.get('max_amount', 0)}
- Usual channels: {baseline.get('usual_channels', [])}
- Usual regions: {baseline.get('usual_regions', [])}

TRANSACTIONS IN 72H WINDOW: {len(bundle['window'])}
{json.dumps([{"ts": t["ts"], "amount": t["amount"], "channel": t.get("channel")} 
             for t in bundle["window"][:5]], indent=2)}

SIMILAR CLOSED CASES:
{json.dumps(similar_summary, indent=2)}

REGION ACTIVITY (same region, last 7 days): {len(bundle['region_neighbors'])} transactions

Analyse this case and respond with JSON:
{{
  "verdict": "fraud" | "legitimate" | "uncertain",
  "fraud_probability": 0.0-1.0,
  "pattern": "card_not_present_fraud" | "card_not_present_new_device" | "out_of_region_use" | "account_takeover" | "card_testing" | "undocumented" | "none",
  "exposure_usd": float,
  "shared_origin": true | false,
  "evidence": [
    {{"signal": "...", "value": "...", "weight": "low|medium|high", "supports": "fraud|legitimate|neutral"}}
  ],
  "analyst_notes": "2-3 sentence summary of the investigation",
  "needs_customer_contact": true | false,
  "customer_contact_question": "What to ask the customer"
}}
"""

    return llm_json(prompt, system="You are an expert fraud analyst. Be precise and data-driven.")


# ── Simulate customer response ────────────────────────────────────────────────

def simulate_customer_response(case_id: str, question: str, analysis: dict) -> dict:
    """
    Simulate customer response for uncertain cases.
    Returns dict with responded: bool, confirmed_fraud: bool
    """
    prompt = f"""
Case {case_id}: A customer was contacted about a suspicious transaction.

Question asked: {question}
Fraud probability: {analysis['fraud_probability']}
Pattern: {analysis['pattern']}
Analyst notes: {analysis['analyst_notes']}

Simulate a realistic customer response. Respond with JSON:
{{
  "responded": true | false,
  "confirmed_fraud": true | false,
  "response_text": "what the customer said or 'No response within 24 hours'"
}}

Guidelines:
- If fraud_probability > 0.8: customer likely confirms fraud (70% chance)
- If fraud_probability < 0.4: customer likely says it was legitimate (80% chance)  
- If 0.4-0.8: mixed — 40% no response, 30% confirms fraud, 30% says legitimate
"""
    return llm_json(prompt)


# ── Main investigation ────────────────────────────────────────────────────────

def investigate_case(case: dict) -> CaseAnswer:
    """
    Full investigation pipeline for one case.
    Returns a validated CaseAnswer ready to save as JSON.
    """
    case_id = case["case_id"]
    print(f"\n{'='*60}")
    print(f"Investigating {case_id} | Customer: {case['customer_id']}")
    print(f"{'='*60}")

    # Step 1 — Get flagged transaction
    print(f"  [0/5] Fetching flagged transaction {case['flagged_txn_id']}...")
    txn = gc.get_transaction(case["flagged_txn_id"])
    if not txn:
        txn = {"txn_id": case["flagged_txn_id"], "amount": 0, "channel": "unknown"}

    # Step 2 — Gather all evidence from graph
    bundle = gather_evidence(case, txn)

    # Step 3 — LLM analysis
    print(f"  [LLM] Analysing evidence...")
    analysis = analyse_evidence(bundle)
    print(f"  [LLM] Verdict: {analysis['verdict']} | P(fraud)={analysis['fraud_probability']}")

    # Step 4 — Customer contact if needed
    customer_response = None
    evidence_requests = []

    if analysis.get("needs_customer_contact") and analysis["fraud_probability"] < 0.85:
        print(f"  [CONTACT] Simulating customer contact...")
        question = analysis.get("customer_contact_question", "Did you make this transaction?")
        customer_response = simulate_customer_response(case_id, question, analysis)
        evidence_requests.append(EvidenceRequest(
            request_type="CUSTOMER_CONTACT",
            question=question,
            simulated_response=customer_response.get("response_text")
        ))
        print(f"  [CONTACT] Responded: {customer_response.get('responded')} | Fraud: {customer_response.get('confirmed_fraud')}")

    # Step 5 — Policy engine (deterministic)
    print(f"  [POLICY] Running policy engine...")
    prior_cases = bundle["similar_cases"]["by_customer"]
    has_prior = any(cc.get("outcome") == "confirmed_fraud" for cc in prior_cases)

    policy_input = PolicyInput(
        verdict=analysis["verdict"],
        fraud_probability=analysis["fraud_probability"],
        exposure_usd=analysis.get("exposure_usd", txn.get("amount", 0)),
        pattern=analysis["pattern"],
        customer_responded=customer_response.get("responded") if customer_response else None,
        customer_confirmed_fraud=customer_response.get("confirmed_fraud") if customer_response else None,
        shared_origin=analysis.get("shared_origin", False),
        n_cards_confirmed=len([c for c in bundle["cards"] if c.get("card_id") != case["card_id"]]),
        has_prior_fraud=has_prior,
        channel=txn.get("channel", "unknown")
    )

    policy = apply_policy(policy_input)
    print(f"  [POLICY] Initial: {policy.initial_actions}")
    print(f"  [POLICY] Final: {policy.final_actions}")

    # Step 6 — Build answer
    evidence_items = [
        EvidenceItem(
            signal=e["signal"],
            value=str(e["value"]),
            weight=e["weight"],
            supports=e["supports"]
        )
        for e in analysis.get("evidence", [])
    ]

    sar_narrative = None
    if policy.file_sar:
        sar_narrative = llm(
            f"""Write a FinCEN-style SAR narrative for this fraud case.
Case: {case_id}
Customer: {case['customer_id']}
Pattern: {analysis['pattern']}
Amount: ${analysis.get('exposure_usd', txn.get('amount', 0))}
Notes: {analysis['analyst_notes']}
Keep it under 150 words. Professional tone."""
        )

    answer = CaseAnswer(
        case=CaseRecord(
            case_id=case_id,
            customer_id=case["customer_id"],
            card_id=case["card_id"],
            opened_at=case["opened_at"],
            flagged_txn_id=case["flagged_txn_id"],
            trigger_type=case.get("trigger_type", "risk_score"),
            trigger_text=case.get("trigger_text", ""),
            verdict=Verdict(analysis["verdict"]),
            fraud_probability=analysis["fraud_probability"],
            pattern=analysis["pattern"],
            exposure_usd=analysis.get("exposure_usd", txn.get("amount", 0)),
            evidence=evidence_items,
            analyst_notes=analysis["analyst_notes"]
        ),
        sar=SAR(
            file=policy.file_sar,
            subject_name=case["customer_id"] if policy.file_sar else None,
            subject_id=case["customer_id"] if policy.file_sar else None,
            amount=analysis.get("exposure_usd") if policy.file_sar else None,
            activity_type=analysis["pattern"] if policy.file_sar else None,
            narrative=sar_narrative
        ),
        next_best_actions=NextBestActions(
            initial=policy.initial_actions,
            final=policy.final_actions,
            rules_applied=policy.rules_applied
        ),
        evidence_requests=evidence_requests
    )

    # Step 7 — Write back to graph
    gc.write_investigation_case(
        case_id=case_id,
        customer_id=case["customer_id"],
        card_id=case["card_id"],
        opened_at=case["opened_at"],
        status="closed",
        verdict=analysis["verdict"],
        fraud_probability=analysis["fraud_probability"],
        pattern=analysis["pattern"],
        exposure_usd=analysis.get("exposure_usd", 0),
        summary=analysis["analyst_notes"]
    )

    return answer