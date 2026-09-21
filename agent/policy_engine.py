"""
Deterministic policy engine — matches Fraud Policy v1.0 exactly.
Action names and approval routes here must match the policy document verbatim
(see get_route/get_rule in answer_schema.py for the routing table).
The LLM decides WHAT happened. This decides WHAT TO DO.
No LLM calls here — pure Python logic, fully unit-testable.
"""

from dataclasses import dataclass
from typing import Optional

# §R1: verify before blocking on a weak/single signal below this probability
R1_VERIFY_THRESHOLD = 0.70


@dataclass
class PolicyInput:
    verdict: str                                    # fraud | legitimate | uncertain
    fraud_probability: float                        # 0.0 - 1.0
    exposure_usd: float                              # total amount at risk
    pattern: str                                     # detected fraud pattern
    customer_responded: Optional[bool]               # True=replied, None=no reply
    customer_confirmed_fraud: Optional[bool]         # True=denied txn (fraud), False=confirmed legit, None=no reply
    shared_origin: bool                              # device/region/email shared with other fraud
    n_cards_confirmed: int                           # confirmed-fraud cards for this customer
    has_prior_fraud: bool                            # customer has prior confirmed fraud cases
    channel: str                                     # online | in_person
    evidence_count: int = 0                          # §6 stopping rule: independent evidence items gathered
    evidence_conflicts: bool = False                 # §R8: graph-derived signals disagree with the LLM's read
    card_testing_cleared_over_100: bool = False      # §R5: a >$100 purchase already cleared
    matches_recurring_charge: bool = False           # §R7: disputed charge matches known recurring pattern
    coordinated_across_customers: bool = False       # §R9: undocumented + coordinated/repeated abuse signal
    credentials_confirmed_compromised: bool = False  # §R10 OR-condition


@dataclass
class PolicyOutput:
    initial_actions: list[str]
    final_actions: list[str]
    rules_applied: list[str]
    file_sar: bool
    escalate: bool


def deduplicate(actions: list[str]) -> list[str]:
    """Remove duplicates while preserving order."""
    seen = set()
    out = []
    for a in actions:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def run_policy(p: PolicyInput) -> PolicyOutput:
    initial: list[str] = []
    final: list[str] = []
    rules: list[str] = []
    file_sar = False
    escalate = False

    # ── Initial actions (before customer contact / final evidence) ─────────
    # §3a: a case is opened whenever an investigation is warranted.
    initial.append("CREATE_CASE")

    # §R1: verify (or step up) before blocking below the 0.70 confidence bar
    if p.fraud_probability < R1_VERIFY_THRESHOLD:
        initial.append("VERIFY_WITH_CUSTOMER")
        if p.pattern in ("card_testing", "account_takeover", "card_not_present_new_device"):
            initial.append("STEP_UP_AUTH")
        rules.append("R1")
    else:
        initial.append("BLOCK_CARD")
        rules.append("R1")

    # ── Final actions (after evidence gathering / customer contact) ────────

    # §R3: customer confirms the transaction themselves — close, no fraud
    if p.customer_confirmed_fraud is False or p.verdict == "legitimate":
        final.append("CREATE_CASE")
        final.append("CLOSE_NO_FRAUD")
        rules.append("R3")
        return PolicyOutput(deduplicate(initial), deduplicate(final), deduplicate(rules), False, False)

    # §R9: undocumented pattern — its own path, doesn't get forced into R2/R5/etc.
    if p.pattern == "undocumented":
        final.append("CREATE_CASE")
        final.append("ESCALATE_TO_ANALYST")
        escalate = True
        rules.append("R9")
        # §3a: report when coordinated/repeated abuse or exposure exceeds $1,000
        if p.coordinated_across_customers or p.exposure_usd > 1000:
            final.append("FILE_REPORT")
            file_sar = True
        return PolicyOutput(deduplicate(initial), deduplicate(final), deduplicate(rules), file_sar, escalate)

    # §R7: disputed but matches the customer's own recurring pattern — never block
    if p.matches_recurring_charge and p.customer_confirmed_fraud is not False:
        final.append("CREATE_CASE")
        final.append("VERIFY_WITH_CUSTOMER")
        final.append("WARN_CUSTOMER")
        rules.append("R7")
        return PolicyOutput(deduplicate(initial), deduplicate(final), deduplicate(rules), False, False)

    # ── Confirmed / high-confidence fraud path ──────────────────────────────
    if p.verdict == "fraud" or p.customer_confirmed_fraud is True or p.fraud_probability >= 0.85:
        final.append("CREATE_CASE")
        final.append("BLOCK_CARD")
        rules.append("R2")

        # §R2: exposure > $1,000 → file report
        if p.exposure_usd > 1000:
            final.append("FILE_REPORT")
            file_sar = True
            rules.append("R2-sar")

        # §R5: card testing — decline + step-up; only BLOCK_CARD if a >$100
        # purchase already cleared (otherwise the decline/step-up is enough)
        if p.pattern == "card_testing":
            final.append("DECLINE_TRANSACTION")
            final.append("STEP_UP_AUTH")
            if not p.card_testing_cleared_over_100 and "BLOCK_CARD" in final:
                final.remove("BLOCK_CARD")
            rules.append("R5")

        # §R6: shared origin — monitor connected cards, always ends up with a report
        if p.shared_origin:
            final.append("MONITOR_CONNECTED_CARDS")
            if "FILE_REPORT" not in final:
                final.append("FILE_REPORT")
                file_sar = True
            rules.append("R6")

        # Account takeover always warrants a report regardless of exposure
        if p.pattern == "account_takeover" and "FILE_REPORT" not in final:
            final.append("FILE_REPORT")
            file_sar = True

        # §R10: BLOCK_ALL_CARDS only with 2+ confirmed cards OR compromised creds
        if p.n_cards_confirmed >= 2 or p.credentials_confirmed_compromised:
            if "BLOCK_CARD" in final:
                final.remove("BLOCK_CARD")
            final.append("BLOCK_ALL_CARDS")
            rules.append("R10")

        return PolicyOutput(deduplicate(initial), deduplicate(final), deduplicate(rules), file_sar, escalate)

    # ── Uncertain path ───────────────────────────────────────────────────────
    if p.verdict == "uncertain":
        final.append("CREATE_CASE")

        # §R4: no reply within 24h — decline pending auth, monitor
        if not p.customer_responded:  # asked and no reply (False), or never answered (None)
            final.append("MONITOR_CARD")
            final.append("DECLINE_TRANSACTION")
            rules.append("R4")

        # §R8: uncertain and exposure > $500, or evidence conflicts — escalate
        if p.exposure_usd > 500 or p.evidence_conflicts:
            final.append("ESCALATE_TO_ANALYST")
            escalate = True
            rules.append("R8")

        # §R6: shared origin even while uncertain — name it, monitor connected
        # cards, and file a report (policy doesn't make this conditional on
        # a confirmed-fraud verdict)
        if p.shared_origin:
            final.append("MONITOR_CONNECTED_CARDS")
            if "FILE_REPORT" not in final:
                final.append("FILE_REPORT")
                file_sar = True
            rules.append("R6")

        if len(final) == 1:  # nothing but CREATE_CASE applied — default to monitoring
            final.append("MONITOR_CARD")

        return PolicyOutput(deduplicate(initial), deduplicate(final), deduplicate(rules), file_sar, escalate)

    # ── Fallback (shouldn't normally be reached) ────────────────────────────
    final.append("MONITOR_CARD")
    return PolicyOutput(deduplicate(initial), deduplicate(final), deduplicate(rules), file_sar, escalate)


def apply_policy(p: PolicyInput) -> PolicyOutput:
    """Run policy — actions are already deduplicated inside run_policy."""
    return run_policy(p)


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
    print(f"  Final: {r1.final_actions}  Rules: {r1.rules_applied}\n")
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

    # Test 3: Uncertain, no reply, high exposure → escalate
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
    assert "DECLINE_TRANSACTION" in r3.final_actions  # bug fix check: not "DECLINE"

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
    print(f"  Final: {r4.final_actions}  Rules: {r4.rules_applied}\n")
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

    # Test 6: Card testing, no purchase cleared over $100 — decline/step-up only
    p6 = PolicyInput(
        verdict="fraud", fraud_probability=0.9,
        exposure_usd=45, pattern="card_testing",
        customer_responded=None, customer_confirmed_fraud=None,
        shared_origin=False, n_cards_confirmed=0,
        has_prior_fraud=False, channel="online",
        card_testing_cleared_over_100=False
    )
    r6 = apply_policy(p6)
    print("Test 6 — Card testing, nothing cleared yet:")
    print(f"  Final: {r6.final_actions}\n")
    assert "BLOCK_CARD" not in r6.final_actions
    assert "DECLINE_TRANSACTION" in r6.final_actions
    assert "STEP_UP_AUTH" in r6.final_actions

    # Test 7: Card testing, a >$100 purchase already cleared — block too
    p7 = PolicyInput(
        verdict="fraud", fraud_probability=0.9,
        exposure_usd=150, pattern="card_testing",
        customer_responded=None, customer_confirmed_fraud=None,
        shared_origin=False, n_cards_confirmed=0,
        has_prior_fraud=False, channel="online",
        card_testing_cleared_over_100=True
    )
    r7 = apply_policy(p7)
    print("Test 7 — Card testing, $100+ purchase cleared:")
    print(f"  Final: {r7.final_actions}\n")
    assert "BLOCK_CARD" in r7.final_actions

    # Test 8: Recurring charge dispute — R7, never block
    p8 = PolicyInput(
        verdict="uncertain", fraud_probability=0.5,
        exposure_usd=60, pattern="none",
        customer_responded=True, customer_confirmed_fraud=True,
        shared_origin=False, n_cards_confirmed=0,
        has_prior_fraud=False, channel="online",
        matches_recurring_charge=True
    )
    r8 = apply_policy(p8)
    print("Test 8 — Recurring charge dispute (R7):")
    print(f"  Final: {r8.final_actions}  Rules: {r8.rules_applied}\n")
    assert "BLOCK_CARD" not in r8.final_actions
    assert "WARN_CUSTOMER" in r8.final_actions
    assert "R7" in r8.rules_applied

    # Test 9: Undocumented, coordinated — always files + escalates
    p9 = PolicyInput(
        verdict="uncertain", fraud_probability=0.6,
        exposure_usd=200, pattern="undocumented",
        customer_responded=None, customer_confirmed_fraud=None,
        shared_origin=False, n_cards_confirmed=0,
        has_prior_fraud=False, channel="online",
        coordinated_across_customers=True
    )
    r9 = apply_policy(p9)
    print("Test 9 — Undocumented, coordinated (R9):")
    print(f"  Final: {r9.final_actions}  Rules: {r9.rules_applied}\n")
    assert "FILE_REPORT" in r9.final_actions
    assert "ESCALATE_TO_ANALYST" in r9.final_actions
    assert "R9" in r9.rules_applied

    print("✅ All policy engine tests passed")

    # Test 10: Uncertain + shared origin — must now file a report too
    p10 = PolicyInput(
        verdict="uncertain", fraud_probability=0.5,
        exposure_usd=200, pattern="out_of_region_use",
        customer_responded=None, customer_confirmed_fraud=None,
        shared_origin=True, n_cards_confirmed=0,
        has_prior_fraud=False, channel="in_person"
    )
    r10 = apply_policy(p10)
    print("Test 10 — Uncertain + shared origin (R6 now files too):")
    print(f"  Final: {r10.final_actions}\n")
    assert "MONITOR_CONNECTED_CARDS" in r10.final_actions
    assert "FILE_REPORT" in r10.final_actions
    assert r10.file_sar is True

    # Test 11: Uncertain, low exposure, but evidence conflicts — still escalate
    p11 = PolicyInput(
        verdict="uncertain", fraud_probability=0.5,
        exposure_usd=100, pattern="card_not_present_fraud",
        customer_responded=None, customer_confirmed_fraud=None,
        shared_origin=False, n_cards_confirmed=0,
        has_prior_fraud=False, channel="online",
        evidence_conflicts=True
    )
    r11 = apply_policy(p11)
    print("Test 11 — Uncertain, low exposure, conflicting evidence (R8):")
    print(f"  Final: {r11.final_actions}\n")
    assert "ESCALATE_TO_ANALYST" in r11.final_actions

    print("✅ All extended policy engine tests passed")