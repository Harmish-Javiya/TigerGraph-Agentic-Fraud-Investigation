"""
Deterministic policy engine — matches Fraud Policy v1.0 exactly.
Action names and approval routes here must match the policy document verbatim
(see get_route/get_rule in answer_schema.py for the routing table).
The LLM decides WHAT happened. This decides WHAT TO DO.
No LLM calls here — pure Python logic, fully unit-testable.
"""

from dataclasses import dataclass, field
from typing import Optional

# §R1: verify before blocking on a weak/single signal below this probability
R1_VERIFY_THRESHOLD = 0.70


@dataclass
class PolicyInput:
    verdict: str                                    # fraud | legitimate | uncertain
    fraud_probability: float                        # 0.0 - 1.0
    exposure_usd: float                              # total amount at risk
    pattern: str                                     # detected fraud pattern
    customer_responded: Optional[bool]               # True=replied, None=no validation request
    customer_confirmed_fraud: Optional[bool]         # retained for compatibility; never inferred from the LLM
    shared_origin: bool                              # retained for compatibility; use shared_origin_confirmed below
    n_cards_confirmed: int                           # retained for compatibility; use confirmed_compromised_cards below
    has_prior_fraud: bool                            # historical context, not a policy fact by itself
    channel: str                                     # online | in_person
    evidence_count: int = 0                          # retrieved evidence records (not §6 support count)
    evidence_conflicts: bool = False                 # §R8: graph-derived signals disagree with the LLM's read
    card_testing_cleared_over_100: bool = False      # §R5: a >$100 purchase already cleared
    matches_recurring_charge: bool = False           # §R7: disputed charge matches known recurring pattern
    coordinated_across_customers: bool = False       # §R9: undocumented + coordinated/repeated abuse signal
    credentials_confirmed_compromised: bool = False  # §R10 OR-condition
    # These predicates are populated from validated graph/customer state in
    # investigator.py.  Model probability, pattern labels, and prose must not
    # set them.
    customer_denied: bool = False
    customer_confirmed: bool = False
    customer_no_response_24h: bool = False
    card_testing_detected: bool = False
    shared_origin_confirmed: bool = False
    coordinated_abuse_confirmed: bool = False
    confirmed_compromised_cards: int = 0
    independent_support_count: int = 0
    independent_support_signals: list[str] = field(default_factory=list)
    weak_evidence: bool = False


@dataclass
class PolicyOutput:
    initial_actions: list[str]
    final_actions: list[str]
    rules_applied: list[str]
    file_sar: bool
    escalate: bool
    action_reasons: dict[str, str]
    sar_reason: str


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
    reasons: dict[str, str] = {"CREATE_CASE": "§3a: investigation warranted, case opened"}
    sar_reasons: list[str] = []

    def add(action: str, reason: str, *, target: list[str] = final) -> None:
        target.append(action)
        # If this action was already justified by an earlier rule in this same
        # run, combine the citations (e.g. "R2: ...; R5: ...") instead of
        # silently dropping the second rule's reason. Still fully deterministic
        # — nothing here is authored by an LLM, only concatenated from the
        # rule text this function itself is passing in.
        if action in reasons and reason not in reasons[action]:
            reasons[action] = f"{reasons[action]}; {reason}"
        else:
            reasons.setdefault(action, reason)

    def output() -> PolicyOutput:
        sar_reason = (
            "; ".join(sar_reasons) if sar_reasons
            else "No deterministic policy condition requires FILE_REPORT."
        )
        return PolicyOutput(
            deduplicate(initial), deduplicate(final), deduplicate(rules),
            file_sar, escalate, reasons, sar_reason,
        )

    # ── Initial actions (before customer contact / final evidence) ─────────
    # §3a: a case is opened whenever an investigation is warranted.
    initial.append("CREATE_CASE")

    # §R1: verify (or step up) before blocking below the 0.70 confidence bar
    if p.fraud_probability < R1_VERIFY_THRESHOLD and p.weak_evidence:
        add("VERIFY_WITH_CUSTOMER", "R1: weak evidence below 0.70 requires verification before blocking", target=initial)
        if p.pattern in ("card_testing", "account_takeover", "card_not_present_new_device"):
            add("STEP_UP_AUTH", "R1: weak evidence requires step-up authentication before blocking", target=initial)
        rules.append("R1")

    # ── Final actions (after evidence gathering / customer contact) ────────

    # §R3: customer confirms the transaction themselves — close, no fraud
    if p.customer_confirmed:
        add("CREATE_CASE", "§3a: investigation warranted, case opened")
        add("CLOSE_NO_FRAUD", "R3: customer confirmed the transaction was legitimate")
        rules.append("R3")
        return output()

    # §R9: undocumented pattern — its own path, doesn't get forced into R2/R5/etc.
    if p.pattern == "undocumented" and p.coordinated_abuse_confirmed:
        add("CREATE_CASE", "§3a: investigation warranted, case opened")
        add("ESCALATE_TO_ANALYST", "R9: validated coordinated abuse does not fit a documented pattern")
        escalate = True
        rules.append("R9")
        add("FILE_REPORT", "R9: validated coordinated undocumented abuse requires a report")
        file_sar = True
        sar_reasons.append("R9: validated coordinated undocumented abuse requires FILE_REPORT.")
        return output()

    # §R7: disputed but matches the customer's own recurring pattern — never block
    if p.matches_recurring_charge:
        add("CREATE_CASE", "§3a: customer dispute warrants a case")
        add("VERIFY_WITH_CUSTOMER", "R7: charge matches a validated recurring pattern; verify rather than block")
        add("WARN_CUSTOMER", "R7: validated recurring legitimate-dispute pattern")
        rules.append("R7")
        return output()

    # ── Confirmed / high-confidence fraud path ──────────────────────────────
    if p.customer_denied:
        add("CREATE_CASE", "R2: customer denied authorizing the transaction")
        add("BLOCK_CARD", "R2: customer denial established unauthorized use")
        rules.append("R2")

        # §3a/R2: a report needs denial plus high exposure or a validated
        # shared-origin fact.  Probability alone is never enough.
        if p.exposure_usd > 1000 or p.shared_origin_confirmed:
            add("FILE_REPORT", "R2/§3a: customer denial plus high exposure or validated shared origin requires a report")
            file_sar = True
            rules.append("R2-sar")
            sar_reasons.append("R2/§3a: customer denial plus high exposure or validated shared origin requires FILE_REPORT.")

    # R5 is established only by the required transaction sequence, never by
    # an LLM pattern label.
    if p.card_testing_detected:
        add("CREATE_CASE", "§3a: card-testing evidence warrants a case")
        add("DECLINE_TRANSACTION", "R5: validated card-testing sequence")
        add("STEP_UP_AUTH", "R5: validated card-testing sequence")
        if p.card_testing_cleared_over_100:
            add("BLOCK_CARD", "R5: validated card testing included a cleared purchase over $100")
        rules.append("R5")

    # R6 requires a validated common element and confirmed fraud on several
    # cards, not merely a device-neighbor record.
    if p.shared_origin_confirmed:
        add("CREATE_CASE", "R6: validated shared origin across confirmed fraud")
        add("MONITOR_CONNECTED_CARDS", "R6: validated shared origin across confirmed fraud")
        add("FILE_REPORT", "R6: validated shared origin across confirmed fraud requires a report")
        file_sar = True
        sar_reasons.append("R6: validated shared origin across confirmed fraud requires FILE_REPORT.")
        rules.append("R6")

    # R10 is a stricter replacement for an individual-card block.
    if p.confirmed_compromised_cards >= 2 or p.credentials_confirmed_compromised:
        if "BLOCK_CARD" in final:
            final.remove("BLOCK_CARD")
        add("BLOCK_ALL_CARDS", "R10: two confirmed compromised cards or confirmed credential compromise")
        rules.append("R10")

    if final:
        return output()

    # ── Uncertain path ───────────────────────────────────────────────────────
    if p.verdict == "uncertain":
        add("CREATE_CASE", "§3a: investigation remains unresolved")

        # §R4: no reply within 24h — decline pending auth, monitor
        # R4 applies only after a customer-validation request was actually made
        # and the customer did not reply.  `None` means no request was made,
        # not "no response"; treating the two as equivalent created fictional
        # R4 outcomes in otherwise evidence-only cases.
        if p.customer_no_response_24h:
            add("MONITOR_CARD", "R4: validation request received no response within 24 hours")
            add("DECLINE_TRANSACTION", "R4: no response within 24 hours for pending authorization")
            rules.append("R4")

        # §R8: uncertain and exposure > $500, or evidence conflicts — escalate
        if p.exposure_usd > 500 or p.evidence_conflicts:
            add("ESCALATE_TO_ANALYST", "R8: unresolved high exposure or conflicting validated evidence")
            escalate = True
            rules.append("R8")

        # §R6: shared origin even while uncertain — name it, monitor connected
        # cards, and file a report (policy doesn't make this conditional on
        # a confirmed-fraud verdict)
        if len(final) == 1:  # nothing but CREATE_CASE applied — default to monitoring
            add("MONITOR_CARD", "Investigation remains open pending additional evidence")

        return output()

    # ── Fallback (shouldn't normally be reached) ────────────────────────────
    add("CREATE_CASE", "§3a: investigation warranted, case opened")
    add("MONITOR_CARD", "Evidence does not establish a deterministic policy predicate")
    return output()


def apply_policy(p: PolicyInput) -> PolicyOutput:
    """Run policy — actions are already deduplicated inside run_policy."""
    return run_policy(p)


# ── Unit tests ────────────────────────────────────────────────────────────────
def run_regression_tests() -> None:
    """Policy-predicate regression coverage; no LLM or graph dependency."""
    base = dict(
        verdict="fraud", fraud_probability=0.90, exposure_usd=1500,
        pattern="card_not_present_fraud", customer_responded=None,
        customer_confirmed_fraud=None, shared_origin=False, n_cards_confirmed=0,
        has_prior_fraud=False, channel="online", weak_evidence=True,
    )

    high_only = apply_policy(PolicyInput(**base))
    assert not {"R2", "R5", "R6", "R9", "R10"} & set(high_only.rules_applied)
    assert "BLOCK_CARD" not in high_only.final_actions
    assert "FILE_REPORT" not in high_only.final_actions

    denied = apply_policy(PolicyInput(**base, customer_denied=True))
    assert "R2" in denied.rules_applied and "BLOCK_CARD" in denied.final_actions
    confirmed = apply_policy(PolicyInput(**{**base, "verdict": "legitimate", "fraud_probability": 0.05},
                                         customer_confirmed=True))
    assert "R3" in confirmed.rules_applied and "CLOSE_NO_FRAUD" in confirmed.final_actions
    early_no_reply = apply_policy(PolicyInput(**{**base, "verdict": "uncertain", "fraud_probability": 0.5,
                                                 "customer_responded": False}))
    assert "R4" not in early_no_reply.rules_applied
    no_reply_24h = apply_policy(PolicyInput(**{**base, "verdict": "uncertain", "fraud_probability": 0.5},
                                             customer_no_response_24h=True))
    assert "R4" in no_reply_24h.rules_applied
    testing = apply_policy(PolicyInput(**base, card_testing_detected=True,
                                       card_testing_cleared_over_100=True))
    assert "R5" in testing.rules_applied and "BLOCK_CARD" in testing.final_actions
    shared = apply_policy(PolicyInput(**base, shared_origin_confirmed=True))
    assert "R6" in shared.rules_applied and "MONITOR_CONNECTED_CARDS" in shared.final_actions
    recurring = apply_policy(PolicyInput(**base, matches_recurring_charge=True))
    assert "R7" in recurring.rules_applied and "BLOCK_CARD" not in recurring.final_actions
    coordinated = apply_policy(PolicyInput(**{**base, "verdict": "uncertain", "fraud_probability": 0.5,
                                              "pattern": "undocumented"},
                                            coordinated_abuse_confirmed=True))
    assert "R9" in coordinated.rules_applied and "FILE_REPORT" in coordinated.final_actions
    two_cards = apply_policy(PolicyInput(**base, customer_denied=True, confirmed_compromised_cards=2))
    creds = apply_policy(PolicyInput(**base, customer_denied=True, credentials_confirmed_compromised=True))
    assert "R10" in two_cards.rules_applied and "BLOCK_ALL_CARDS" in two_cards.final_actions
    assert "R10" in creds.rules_applied and "BLOCK_ALL_CARDS" in creds.final_actions
    assert high_only.action_reasons.get("BLOCK_CARD") is None
    print("✅ 12 deterministic policy-predicate regression tests passed")


if __name__ == "__main__":
    run_regression_tests()


# Retained as historical examples; the assertions pre-date fact predicates and
# are intentionally not executed.  The suite above is the executable contract.
if __name__ == "__main__" and False:
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
        customer_responded=False, customer_confirmed_fraud=None,
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
