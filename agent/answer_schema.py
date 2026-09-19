from pydantic import BaseModel, Field, model_validator
from typing import Optional
from enum import Enum


# ── Enums ────────────────────────────────────────────────────────────────────

class Verdict(str, Enum):
    FRAUD = "fraud"
    LEGITIMATE = "legitimate"
    UNCERTAIN = "uncertain"

class Pattern(str, Enum):
    CARD_NOT_PRESENT = "card_not_present_fraud"
    CARD_NOT_PRESENT_NEW_DEVICE = "card_not_present_new_device"
    OUT_OF_REGION = "out_of_region_use"
    ACCOUNT_TAKEOVER = "account_takeover"
    CARD_TESTING = "card_testing"
    UNDOCUMENTED = "undocumented"
    NONE = "none"

class Action(str, Enum):
    # Case actions
    CREATE_CASE = "CREATE_CASE"
    CLOSE_NO_FRAUD = "CLOSE_NO_FRAUD"
    ESCALATE_TO_ANALYST = "ESCALATE_TO_ANALYST"
    # Card actions
    BLOCK_CARD = "BLOCK_CARD"
    BLOCK_ALL_CARDS = "BLOCK_ALL_CARDS"
    MONITOR_CARD = "MONITOR_CARD"
    MONITOR_CONNECTED_CARDS = "MONITOR_CONNECTED_CARDS"
    # Transaction actions
    DECLINE = "DECLINE"
    STEP_UP_AUTH = "STEP_UP_AUTH"
    # Customer actions
    VERIFY_WITH_CUSTOMER = "VERIFY_WITH_CUSTOMER"
    WARN_CUSTOMER = "WARN_CUSTOMER"
    REIMBURSE_CUSTOMER = "REIMBURSE_CUSTOMER"
    # Reporting
    FILE_REPORT = "FILE_REPORT"


# ── Evidence pieces ───────────────────────────────────────────────────────────

class EvidenceItem(BaseModel):
    signal: str = Field(description="Short name of the signal")
    value: str = Field(description="What was observed")
    weight: str = Field(description="low | medium | high")
    supports: str = Field(description="fraud | legitimate | neutral")


class EvidenceRequest(BaseModel):
    request_type: str = Field(description="e.g. CUSTOMER_CONTACT, ADDITIONAL_TXN_DATA")
    question: str = Field(description="What we are asking or checking")
    simulated_response: Optional[str] = Field(
        default=None,
        description="Simulated answer used by agent to reach final decision"
    )


# ── SAR (Suspicious Activity Report) ─────────────────────────────────────────

class SAR(BaseModel):
    file: bool = Field(description="Whether to file a SAR")
    subject_name: Optional[str] = Field(default=None)
    subject_id: Optional[str] = Field(default=None)
    amount: Optional[float] = Field(default=None)
    activity_type: Optional[str] = Field(default=None)
    narrative: Optional[str] = Field(
        default=None,
        description="FinCEN-style narrative. Required when file=True"
    )

    @model_validator(mode="after")
    def narrative_required_when_filing(self):
        if self.file and not self.narrative:
            raise ValueError("SAR narrative is required when file=True")
        return self


# ── Next Best Actions ─────────────────────────────────────────────────────────

class NextBestActions(BaseModel):
    initial: list[str] = Field(
        description="Actions taken immediately when case opened, before investigation"
    )
    final: list[str] = Field(
        description="Actions after full investigation and evidence gathering"
    )
    rules_applied: list[str] = Field(
        default_factory=list,
        description="Policy rules that drove the final actions e.g. R2, R6"
    )


# ── Case Record ───────────────────────────────────────────────────────────────

class CaseRecord(BaseModel):
    case_id: str
    customer_id: str
    card_id: str
    opened_at: str
    flagged_txn_id: str
    trigger_type: str
    trigger_text: str

    verdict: Verdict
    fraud_probability: float = Field(ge=0.0, le=1.0)
    pattern: str
    exposure_usd: float = Field(ge=0.0)

    evidence: list[EvidenceItem] = Field(default_factory=list)
    analyst_notes: str = Field(description="Human-readable summary of the investigation")


# ── Full Answer File ──────────────────────────────────────────────────────────

class CaseAnswer(BaseModel):
    case: CaseRecord
    sar: SAR
    next_best_actions: NextBestActions
    evidence_requests: list[EvidenceRequest] = Field(default_factory=list)

    @model_validator(mode="after")
    def sar_consistent_with_actions(self):
        """FILE_REPORT in actions must match sar.file."""
        has_file_report = "FILE_REPORT" in self.next_best_actions.final
        if has_file_report != self.sar.file:
            raise ValueError(
                f"Inconsistency: FILE_REPORT in actions={has_file_report} "
                f"but sar.file={self.sar.file}"
            )
        return self

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    example = CaseAnswer(
        case=CaseRecord(
            case_id="HHG-001",
            customer_id="C12382",
            card_id="C12382-K1",
            opened_at="2016-12-05 01:55:28",
            flagged_txn_id="3514030",
            trigger_type="risk_score",
            trigger_text="Real-time model scored transaction 3514030 ($77.07) at 0.61",
            verdict=Verdict.UNCERTAIN,
            fraud_probability=0.61,
            pattern=Pattern.OUT_OF_REGION,
            exposure_usd=77.07,
            evidence=[
                EvidenceItem(
                    signal="in_person_no_device",
                    value="channel=in_person, no device record",
                    weight="medium",
                    supports="neutral"
                ),
                EvidenceItem(
                    signal="long_gap",
                    value="D1=82 days since last transaction",
                    weight="medium",
                    supports="fraud"
                )
            ],
            analyst_notes="Transaction is in-person at region 444. Long gap since last activity. Risk score 0.61 — uncertain. Customer contact required."
        ),
        sar=SAR(file=False),
        next_best_actions=NextBestActions(
            initial=["CREATE_CASE", "VERIFY_WITH_CUSTOMER"],
            final=["VERIFY_WITH_CUSTOMER", "MONITOR_CARD"],
            rules_applied=["R1", "R4"]
        ),
        evidence_requests=[
            EvidenceRequest(
                request_type="CUSTOMER_CONTACT",
                question="Did you make a $77.07 in-person purchase on 2016-12-05?",
                simulated_response="Customer did not respond within 24h"
            )
        ]
    )
    print(example.to_json())
    print("\n✅ answer_schema.py working correctly")