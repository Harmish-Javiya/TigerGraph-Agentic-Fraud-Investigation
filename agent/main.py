"""
Main entry point.
Loads the 20 case pack, runs the investigator on each case,
and saves one JSON answer file per case to the cases/ folder.
"""

import os
import json
import csv
import time
from pathlib import Path
from investigator import investigate_case

AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_DIR.parent

CASES_DIR = Path(os.getenv("CASES_DIR", str(PROJECT_ROOT / "cases")))
CASES_DIR.mkdir(exist_ok=True, parents=True)

CASE_PACK_PATH = os.getenv(
    "CASE_PACK_PATH",
    str(PROJECT_ROOT / "HHGOA_IEEE_DATASETS" / "case_pack.csv")
)

def load_case_pack(path: str) -> list[dict]:
    """Load all 20 cases from case_pack.csv."""
    cases = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(row)
    print(f"Loaded {len(cases)} cases from {path}")
    return cases


def save_answer(answer, case_id: str):
    """Save one answer JSON file."""
    path = CASES_DIR / f"{case_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        f.write(answer.to_json())
    print(f"  ✅ Saved {path}")


def run_single(case_id: str):
    """Run investigation on a single case by ID. Useful for testing."""
    cases = load_case_pack(CASE_PACK_PATH)
    case = next((c for c in cases if c["case_id"] == case_id), None)
    if not case:
        print(f"Case {case_id} not found in case pack")
        return
    answer = investigate_case(case)
    save_answer(answer, case_id)
    print(f"\n=== {case_id} Result ===")
    print(answer.to_json())


def run_all():
    """Run investigation on all 20 cases."""
    cases = load_case_pack(CASE_PACK_PATH)
    results = {"success": [], "failed": []}

    for i, case in enumerate(cases, 1):
        case_id = case["case_id"]
        if (CASES_DIR / f"{case_id}.json").exists():
            print(f"  ⏭ {case_id} already done, skipping")
            results["success"].append(case_id)
            continue
        
        print(f"\n[{i}/20] Starting {case_id}...")
        try:
            answer = investigate_case(case)
            save_answer(answer, case_id)
            results["success"].append(case_id)
        except Exception as e:
            print(f"  ❌ FAILED {case_id}: {e}")
            results["failed"].append({"case_id": case_id, "error": str(e)})
        # Small delay to avoid rate limiting on Groq free tier
        time.sleep(2)

    # Summary
    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(results['success'])}/20 cases investigated")
    if results["failed"]:
        print(f"FAILED: {[f['case_id'] for f in results['failed']]}")
    print(f"Answer files saved to: {CASES_DIR.resolve()}")

    # Save run summary
    summary_path = CASES_DIR / "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Run single case: python main.py HHG-001
        run_single(sys.argv[1])
    else:
        # Run all 20
        run_all()