"""
Deterministic policy engine — R1 to R10.
The LLM decides WHAT happened. This decides WHAT TO DO.
No LLM calls here — pure Python logic, fully unit-testable.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PolicyInput:
    verdict: str                        # fraud | legitimate | uncertain
    fraud_probability: float            # 0.0 - 1.0
    exposure_usd: float                 # total amount at risk
    pattern: str                        # detected fraud pattern
    customer_responded: Optional[bool]  # True=confirmed legit, False=denied, None=no reply
    customer_confirmed_fraud: Optional[bool]  # True if customer said it was fraud
    shared_origin: bool                 # device/region/email shared with other fraud
    n_cards_confirmed: int              # number of confirmed fraud cards for this customer
    has_prior_fraud: bool               # customer has prior confirmed fraud cases
    channel: str                        # online | in_person


@dataclass
class PolicyOutput:
    initial_actions: list[str]
    final_actions: list[str]
    rules_applied: list[str]
    file_sar: bool
    escalate: bool


def run_policy(p: PolicyInput) -> PolicyOutput:
    initial = []
    final = []
    rules = []
    file_sar = False
    escalate = False

    # ── Initial actions (before investigation) ─────────────────────────────

    # Always create a case when flagged
    initial.append("CREATE_CASE")

    # R1: Verify before blocking if probability < 0.85
    if p.fraud_probability < 0.85:
        initial.append("VERIFY_WITH_CUSTOMER")
        rules.append("R1")
    else:
        # High confidence fraud — block immediately
        initial.append("BLOCK_CARD")
        rules.append("R1-high-confidence")

    # ── Final actions (after evidence gathering) ───────────────────────────

    if p.verdict == "legitimate" or p.customer_confirmed_fraud is False:
        # R3: Customer confirmed legitimate — close
        final.append("CLOSE_NO_FRAUD")
        rules.append("R3")
        return PolicyOutput(initial, final, rules, False, False)

    if p.verdict == "fraud" or (
        p.fraud_probability >= 0.85 and p.customer_responded is not False
    ):
        # Confirmed fraud path
        final.append("CREATE_CASE")
        final.append("BLOCK_CARD")
        rules.append("R2")

        # R2: Customer denied or no contact — file report if exposure > $1000
        if p.exposure_usd >= 1000:
            final.append("FILE_REPORT")
            file_sar = True
            rules.append("R2-sar")

        # Reimburse if customer reported it
        if p.customer_confirmed_fraud:
            final.append("REIMBURSE_CUSTOMER")

        # R6: Shared origin — monitor connected cards, always file report
        if p.shared_origin:
            final.append("MONITOR_CONNECTED_CARDS")
            if "FILE_REPORT" not in final:
                final.append("FILE_REPORT")
                file_sar = True
            rules.append("R6")

        # R10: Block ALL cards only if 2+ confirmed fraud cards
        if p.n_cards_confirmed >= 2:
            if "BLOCK_CARD" in final:
                final.remove("BLOCK_CARD")
            final.append("BLOCK_ALL_CARDS")
            rules.append("R10")

        # R5: Card testing pattern
        if p.pattern == "card_testing":
            if "BLOCK_CARD" not in final and "BLOCK_ALL_CARDS" not in final:
                final.append("BLOCK_CARD")
            final.append("STEP_UP_AUTH")
            final.append("DECLINE")
            rules.append("R5")

        # Always file SAR for account takeover
        if p.pattern == "account_takeover":
            if "FILE_REPORT" not in final:
                final.append("FILE_REPORT")
                file_sar = True

        return PolicyOutput(initial, final, rules, file_sar, False)

    # ── Uncertain path ─────────────────────────────────────────────────────

    if p.verdict == "uncertain":

        # R4: No customer reply
        if p.customer_responded is None:
            final.append("MONITOR_CARD")
            if p.exposure_usd >= 500:
                final.append("DECLINE")
            rules.append("R4")

        # R7: Disputed but matches known recurring pattern
        if p.has_prior_fraud and p.customer_confirmed_fraud is False:
            final.append("VERIFY_WITH_CUSTOMER")
            final.append("WARN_CUSTOMER")
            rules.append("R7")

        # R8: Uncertain + high exposure → escalate
        if p.exposure_usd >= 500 and not final:
            final.append("ESCALATE_TO_ANALYST")
            escalate = True
            rules.append("R8")
        elif p.exposure_usd >= 500:
            final.append("ESCALATE_TO_ANALYST")
            escalate = True
            rules.append("R8")

        # R9: Undocumented pattern — always file and escalate
        if p.pattern == "undocumented":
            final.append("CREATE_CASE")
            final.append("FILE_REPORT")
            final.append("ESCALATE_TO_ANALYST")
            file_sar = True
            escalate = True
            rules.append("R9")

        # R6: Shared origin even in uncertain cases
        if p.shared_origin:
            final.append("MONITOR_CONNECTED_CARDS")
            rules.append("R6")

        if not final:
            final.append("MONITOR_CARD")

        return PolicyOutput(initial, final, rules, file_sar, escalate)

    # Fallback
    final.append("MONITOR_CARD")
    return PolicyOutput(initial, final, rules, file_sar, escalate)


def deduplicate(actions: list[str]) -> list[str]:
    """Remove duplicates while preserving order."""
    seen = set()
    out = []
    for a in actions:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def apply_policy(p: PolicyInput) -> PolicyOutput:
    """Run policy and deduplicate actions."""
    result = run_policy(p)
    result.initial_actions = deduplicate(result.initial_actions)
    result.final_actions = deduplicate(result.final_actions)
    result.rules_applied = deduplicate(result.rules_applied)
    return result


# ── Unit tests ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running policy engine tests...\n")

    # Test 1: High confidence fraud, high exposure
    p1 = PolicyInput(
        verdict="fraud", fraud_probability=0.92,
        exposure_usd=1500, pattern="card_not_present_fraud",
        customer_responded=None, customer_confirmed_fraud=None,
        shared_origin=False, n_cards_confirmed=1,
        has_prior_fraud=False, channel="online"
    )
    r1 = apply_policy(p1)
    print("Test 1 — High confidence fraud, high exposure:")
    print(f"  Initial: {r1.initial_actions}")
    print(f"  Final:   {r1.final_actions}")
    print(f"  SAR:     {r1.file_sar}")
    print(f"  Rules:   {r1.rules_applied}\n")
    assert "FILE_REPORT" in r1.final_actions
    assert "BLOCK_CARD" in r1.final_actions

    # Test 2: Legitimate — customer confirmed
    p2 = PolicyInput(
        verdict="legitimate", fraud_probability=0.2,
        exposure_usd=50, pattern="none",
        customer_responded=True, customer_confirmed_fraud=False,
        shared_origin=False, n_cards_confirmed=0,
        has_prior_fraud=False, channel="in_person"
    )
    r2 = apply_policy(p2)
    print("Test 2 — Legitimate:")
    print(f"  Final: {r2.final_actions}\n")
    assert "CLOSE_NO_FRAUD" in r2.final_actions
    assert r2.file_sar is False

    # Test 3: Uncertain, no customer reply, high exposure
    p3 = PolicyInput(
        verdict="uncertain", fraud_probability=0.55,
        exposure_usd=800, pattern="out_of_region_use",
        customer_responded=None, customer_confirmed_fraud=None,
        shared_origin=False, n_cards_confirmed=0,
        has_prior_fraud=False, channel="in_person"
    )
    r3 = apply_policy(p3)
    print("Test 3 — Uncertain, no reply, high exposure:")
    print(f"  Final: {r3.final_actions}\n")
    assert "ESCALATE_TO_ANALYST" in r3.final_actions

    # Test 4: Shared origin — R6
    p4 = PolicyInput(
        verdict="fraud", fraud_probability=0.88,
        exposure_usd=300, pattern="account_takeover",
        customer_responded=None, customer_confirmed_fraud=None,
        shared_origin=True, n_cards_confirmed=1,
        has_prior_fraud=True, channel="online"
    )
    r4 = apply_policy(p4)
    print("Test 4 — Fraud + shared origin (R6):")
    print(f"  Final: {r4.final_actions}")
    print(f"  Rules: {r4.rules_applied}\n")
    assert "MONITOR_CONNECTED_CARDS" in r4.final_actions
    assert "R6" in r4.rules_applied

    # Test 5: Two confirmed cards — R10 BLOCK_ALL_CARDS
    p5 = PolicyInput(
        verdict="fraud", fraud_probability=0.95,
        exposure_usd=2000, pattern="account_takeover",
        customer_responded=None, customer_confirmed_fraud=None,
        shared_origin=True, n_cards_confirmed=2,
        has_prior_fraud=True, channel="online"
    )
    r5 = apply_policy(p5)
    print("Test 5 — Two confirmed cards (R10):")
    print(f"  Final: {r5.final_actions}\n")
    assert "BLOCK_ALL_CARDS" in r5.final_actions
    assert "BLOCK_CARD" not in r5.final_actions

    print("✅ All policy engine tests passed")