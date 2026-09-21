"""
Answer schema matching the exact required submission format.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Optional
from enum import Enum


# ── Enums ─────────────────────────────────────────────────────────────────────

class CaseStatus(str, Enum):
    OPEN = "open"
    CLOSED_FRAUD = "closed_fraud"
    CLOSED_LEGITIMATE = "closed_legitimate"
    ESCALATED = "escalated"

class Verdict(str, Enum):
    FRAUD = "fraud"
    LEGITIMATE = "legitimate"
    UNCERTAIN = "uncertain"

class Pattern(str, Enum):
    CARD_TESTING = "card_testing"
    CARD_NOT_PRESENT = "card_not_present_fraud"
    CARD_NOT_PRESENT_NEW_DEVICE = "card_not_present_new_device"
    OUT_OF_REGION = "out_of_region_use"
    ACCOUNT_TAKEOVER = "account_takeover"
    UNDOCUMENTED = "undocumented"
    NONE = "none"

class EvidenceSource(str, Enum):
    GRAPH = "graph"
    DOCUMENT = "document"
    CUSTOMER = "customer"
    EXTERNAL = "external"

class ActionRoute(str, Enum):
    AUTO = "auto"
    L1 = "L1"
    L2 = "L2"


# ── Evidence ──────────────────────────────────────────────────────────────────

class EvidenceItem(BaseModel):
    claim: str = Field(description="Full sentence describing what was found")
    source: EvidenceSource = Field(description="graph | document | customer | external")
    ref: str = Field(description="Query name, document section, or request ID")
    entity_ids: list[str] = Field(
        default_factory=list,
        description="Transaction IDs, case IDs, card IDs this claim rests on"
    )


class EvidenceRequest(BaseModel):
    type: str = Field(description="customer_validation | step_up_auth | analyst_info")
    asked_after_step: int = Field(description="Which step triggered this request")
    assumed_response: str = Field(description="What we assumed the customer/analyst said")


# ── Next Best Actions ─────────────────────────────────────────────────────────

class ActionItem(BaseModel):
    action: str = Field(description="Policy action e.g. BLOCK_CARD, FILE_REPORT")
    route: ActionRoute = Field(description="auto | L1 | L2")
    reason: str = Field(description="Policy rule citation e.g. R2, R5")


class NextBestActions(BaseModel):
    initial: list[ActionItem] = Field(
        description="Actions before evidence requests came back"
    )
    final: list[ActionItem] = Field(
        description="Actions after assumed evidence responses"
    )
    what_changed: str = Field(
        description="Why final differs from initial, or 'nothing'"
    )


# ── SAR ───────────────────────────────────────────────────────────────────────

class SAR(BaseModel):
    file: bool
    reason: str = Field(description="Why to file or why not. Cite policy rule.")
    narrative: str = Field(
        default="",
        description="Required when file=true. 6-12 sentences. Who/what/when/where/how/why."
    )
    subjects: list[str] = Field(
        default_factory=list,
        description="Customer, card, merchant, device IDs in the narrative"
    )
    total_amount_usd: float = Field(default=0.0)
    activity_dates: list[str] = Field(
        default_factory=list,
        description="[first_date, last_date] in YYYY-MM-DD format"
    )

    @model_validator(mode="after")
    def validate_sar(self):
        if self.file:
            if not self.narrative:
                raise ValueError("SAR narrative required when file=true")
            if not self.subjects:
                raise ValueError("SAR subjects required when file=true")
            if len(self.activity_dates) != 2:
                raise ValueError("SAR activity_dates must have exactly 2 dates")
        else:
            self.narrative = ""
            self.subjects = []
            self.total_amount_usd = 0.0
            self.activity_dates = []
        return self


# ── Case ──────────────────────────────────────────────────────────────────────

class Case(BaseModel):
    status: CaseStatus
    verdict: Verdict
    fraud_probability: float = Field(ge=0.0, le=1.0)
    pattern: str
    pattern_description: str = Field(
        default="",
        description="Required when pattern=undocumented. Otherwise empty string."
    )
    affected_txn_ids: list[str] = Field(
        default_factory=list,
        description="All transaction IDs in the fraud episode. Empty if legitimate."
    )
    first_suspicious_txn_id: str = Field(
        default="",
        description="Where the fraud started. Empty if legitimate."
    )
    connected_card_ids: list[str] = Field(
        default_factory=list,
        description="Other cards in the same compromise or ring."
    )
    connected_device_profiles: list[str] = Field(
        default_factory=list,
        description="Device profile strings linking this case to other cards."
    )
    exposure_usd: float = Field(
        default=0.0,
        description="Sum of affected_txn_ids amounts. 0 if legitimate."
    )
    evidence: list[EvidenceItem] = Field(default_factory=list)
    similar_prior_cases: list[str] = Field(
        default_factory=list,
        description="CC-XXXX IDs from closed_cases_history used as memory."
    )
    summary: str = Field(description="2-6 sentences an analyst could read.")
    written_to_graph: bool = Field(default=False)
    graph_case_id: str = Field(default="")

    @model_validator(mode="after")
    def validate_case(self):
        if self.pattern == "undocumented":
            word_count = len(self.pattern_description.split())
            if word_count < 20:
                raise ValueError(
                    f"pattern_description required when pattern=undocumented "
                    f"and must be at least 20 words (got {word_count})"
                )
        if self.verdict == "legitimate":
            self.affected_txn_ids = []
            self.exposure_usd = 0.0
            self.first_suspicious_txn_id = ""
        return self


# ── Full Answer ───────────────────────────────────────────────────────────────

class CaseAnswer(BaseModel):
    case_id: str
    case: Case
    evidence_requests: list[EvidenceRequest] = Field(default_factory=list)
    next_best_actions: NextBestActions
    sar: SAR
    stop_reason: str = Field(description="Why the investigation ended here.")
    tool_calls: int = Field(default=0)
    tokens: int = Field(default=0)
    latency_s: float = Field(default=0.0)

    @model_validator(mode="after")
    def sar_consistent_with_actions(self):
        """FILE_REPORT in final actions must match sar.file."""
        final_action_names = [a.action for a in self.next_best_actions.final]
        has_file_report = "FILE_REPORT" in final_action_names
        if has_file_report != self.sar.file:
            raise ValueError(
                f"FILE_REPORT in actions={has_file_report} "
                f"but sar.file={self.sar.file} — must match"
            )
        return self

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)


# ── Route helper ──────────────────────────────────────────────────────────────

def get_route(action: str, exposure: float) -> str:
    """
    Determine approval route based on action and exposure, per Fraud Policy
    v1.0 §2. Only 'auto' actions may be executed by the agent; L1/L2 actions
    are recommended and wait for a human.
    """
    auto_actions = {
        "ALLOW_TRANSACTION", "MONITOR_CARD", "MONITOR_CONNECTED_CARDS",
        "WARN_CUSTOMER", "VERIFY_WITH_CUSTOMER", "STEP_UP_AUTH",
        "GENERATE_REPORT", "CREATE_CASE", "ESCALATE_TO_ANALYST", "CLOSE_NO_FRAUD",
    }
    if action in auto_actions:
        return "auto"
    if action == "DECLINE_TRANSACTION":
        return "L1"
    if action == "BLOCK_CARD":
        return "L1" if exposure <= 2500 else "L2"
    if action == "BLOCK_ALL_CARDS":
        return "L2"
    if action == "FILE_REPORT":
        return "L2"
    return "auto"


def get_rule(action: str, context: dict) -> str:
    """Map action to the policy rule that triggered it, per Fraud Policy v1.0."""
    rules = {
        "ALLOW_TRANSACTION": "Default — no rule triggered, transaction allowed",
        "VERIFY_WITH_CUSTOMER": "R1: probability < 0.70, verify before blocking",
        "BLOCK_CARD": "R2/R5: confirmed unauthorized use",
        "CLOSE_NO_FRAUD": "R3: customer confirmed legitimate",
        "MONITOR_CARD": "R4: no customer reply within 24h",
        "DECLINE_TRANSACTION": "R4/R5: high risk transaction pending or card testing",
        "STEP_UP_AUTH": "R1/R5: verification signal or card testing pattern detected",
        "MONITOR_CONNECTED_CARDS": "R6: shared device/region origin",
        "FILE_REPORT": "R2/R6/R9: confirmed fraud with shared origin, high exposure, or undocumented coordinated abuse",
        "GENERATE_REPORT": "Internal record only — no case opened",
        "WARN_CUSTOMER": "R7: disputed but matches recurring pattern",
        "ESCALATE_TO_ANALYST": "R8/R9: uncertain with high exposure/conflicting evidence, or undocumented pattern",
        "CREATE_CASE": "§3a: investigation warranted, case opened",
        "BLOCK_ALL_CARDS": "R10: 2+ cards confirmed compromised or credentials confirmed compromised",
    }
    return rules.get(action, "policy rule")


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    example = CaseAnswer(
        case_id="HHG-001",
        case=Case(
            status=CaseStatus.CLOSED_LEGITIMATE,
            verdict=Verdict.UNCERTAIN,
            fraud_probability=0.61,
            pattern="out_of_region_use",
            pattern_description="",
            affected_txn_ids=["3514030"],
            first_suspicious_txn_id="3514030",
            connected_card_ids=[],
            connected_device_profiles=[],
            exposure_usd=77.07,
            evidence=[
                EvidenceItem(
                    claim="Transaction 3514030 occurred in billing region 444, which does not appear in C12382's transaction history of regions [220, 221, 87]",
                    source=EvidenceSource.GRAPH,
                    ref="query:card_history(customer_id=C12382)",
                    entity_ids=["3514030", "C12382"]
                ),
                EvidenceItem(
                    claim="D1=82 days since last transaction — an unusually long gap suggesting possible account dormancy or takeover",
                    source=EvidenceSource.GRAPH,
                    ref="query:get_transaction(txn_id=3514030)",
                    entity_ids=["3514030"]
                ),
                EvidenceItem(
                    claim="Customer did not respond within 24 hours when asked about the transaction",
                    source=EvidenceSource.CUSTOMER,
                    ref="evidence_request:1",
                    entity_ids=[]
                )
            ],
            similar_prior_cases=["CC-0002", "CC-0141"],
            summary="Transaction 3514030 flagged at risk score 0.61 for out-of-region in-person use. Billing region 444 not in customer history. 82-day gap since last activity. No device record (in-person). Customer did not respond within 24h. Monitoring recommended pending response.",
            written_to_graph=True,
            graph_case_id="HHG-001"
        ),
        evidence_requests=[
            EvidenceRequest(
                type="customer_validation",
                asked_after_step=3,
                assumed_response="Customer did not respond within 24 hours"
            )
        ],
        next_best_actions=NextBestActions(
            initial=[
                ActionItem(action="CREATE_CASE", route=ActionRoute.AUTO, reason="R2: flagged transaction requires case creation"),
                ActionItem(action="VERIFY_WITH_CUSTOMER", route=ActionRoute.AUTO, reason="R1: probability 0.61 < 0.85, verify before blocking")
            ],
            final=[
                ActionItem(action="MONITOR_CARD", route=ActionRoute.AUTO, reason="R4: no customer reply within 24h"),
                ActionItem(action="DECLINE", route=ActionRoute.L1, reason="R4: exposure $77.07 under $500 threshold but pattern warrants caution")
            ],
            what_changed="No customer response received within 24h — escalated from verify to monitor per R4. Exposure below $500 so no SAR required."
        ),
        sar=SAR(
            file=False,
            reason="R2 not triggered: customer has not denied the transaction and exposure $77.07 is below $1000 threshold",
        ),
        stop_reason="Customer contact attempted but no response within 24h. Exposure too low for immediate block. Card placed under monitoring per R4.",
        tool_calls=7,
        tokens=4200,
        latency_s=22.5
    )
    print(example.to_json())
    print("\n✅ answer_schema.py working correctly")