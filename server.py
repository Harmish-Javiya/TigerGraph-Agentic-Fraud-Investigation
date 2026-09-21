"""
Flask backend for the fraud-investigation dashboard.

Expected project layout (adjust PROJECT_ROOT-relative paths below if yours differs):

    project_root/
        server.py                  <- this file
        static/index.html          <- dashboard UI
        agent/
            main.py
            investigator.py
            graph_client.py
            policy_engine.py
            answer_schema.py
            graphrag.py
        HHGOA_IEEE_DATASETS/
            case_pack.csv
        cases/                      <- output JSON files land here
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timezone

from flask import Flask, jsonify, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent
AGENT_DIR = PROJECT_ROOT / "agent"
CASE_PACK = PROJECT_ROOT / "HHGOA_IEEE_DATASETS" / "case_pack.csv"
CASES_DIR = PROJECT_ROOT / "cases"
STATIC_DIR = PROJECT_ROOT / "static"

# Bug fix (#4): make the case-pack / cases-dir paths absolute and pass them in
# via env vars BEFORE importing main, instead of relying on main.py's
# cwd-relative default ("../HHGOA_IEEE_DATASETS/case_pack.csv"), which only
# worked because we happened to chdir into agent/ first.
os.environ["CASE_PACK_PATH"] = str(CASE_PACK)
os.environ["CASES_DIR"] = str(CASES_DIR)

os.chdir(AGENT_DIR)
sys.path.insert(0, str(AGENT_DIR))

import main as investigator_main  # noqa: E402  (import after path/env setup)

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")

_lock = threading.Lock()
_jobs: dict = {}
_job_counter = 0


def _new_job(total: int) -> str:
    global _job_counter
    with _lock:
        _job_counter += 1
        job_id = f"job-{_job_counter}"
        _jobs[job_id] = {
            "status": "running",
            "total": total,
            "done": 0,
            "current_case": None,
            "results": {},          # case_id -> "success" | "failed:<error>"
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
        }
    return job_id


def _update_job(job_id: str, **kwargs):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _mark_case(job_id: str, case_id: str, outcome: str):
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job["results"][case_id] = outcome
        job["done"] += 1
        job["current_case"] = case_id


def _run_one(job_id: str, case: dict):
    case_id = case["case_id"]
    try:
        answer = investigator_main.investigate_case(case)
        investigator_main.save_answer(answer, case_id)
        # Bug fix (#2): compute sar_filed once, correctly — no dead
        # overwritten assignment before it.
        sar_filed = any(a.action == "FILE_REPORT" for a in answer.next_best_actions.final)
        with _lock:
            _jobs[job_id].setdefault("sar_filed", {})[case_id] = sar_filed
        _mark_case(job_id, case_id, "success")
    except Exception as e:
        _mark_case(job_id, case_id, f"failed:{e}")


def _run_case_job(job_id: str, case_id: str):
    cases = investigator_main.load_case_pack(investigator_main.CASE_PACK_PATH)
    case = next((c for c in cases if c["case_id"] == case_id), None)
    if not case:
        _mark_case(job_id, case_id, "failed:not_found")
    else:
        _run_one(job_id, case)
    _update_job(job_id, status="done", current_case=None,
                finished_at=datetime.now(timezone.utc).isoformat())


def _run_all_job(job_id: str):
    cases = investigator_main.load_case_pack(investigator_main.CASE_PACK_PATH)
    for case in cases:
        _update_job(job_id, current_case=case["case_id"])
        _run_one(job_id, case)
        time.sleep(15)  # Groq free-tier rate limiting, matches main.py's run_all
    _update_job(job_id, status="done", current_case=None,
                finished_at=datetime.now(timezone.utc).isoformat())


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/api/cases")
def api_cases():
    cases = []
    if CASES_DIR.exists():
        for f in sorted(CASES_DIR.glob("*.json")):
            if f.name == "_summary.json":
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            # Skip stale files left over from an earlier schema version —
            # a valid CaseAnswer always has these top-level keys.
            if not (
                isinstance(data, dict)
                and data.get("case_id")
                and isinstance(data.get("case"), dict)
                and "status" in data["case"]
                and "tool_calls" in data
            ):
                print(f"  [WARN] Skipping stale/malformed case file: {f.name}")
                continue
            cases.append(data)
    return jsonify(cases)


@app.route("/api/run_case", methods=["POST"])
def api_run_case():
    body = request.get_json(force=True, silent=True) or {}
    case_id = body.get("case_id")
    if not case_id:
        return jsonify({"error": "case_id required"}), 400
    job_id = _new_job(total=1)
    threading.Thread(target=_run_case_job, args=(job_id, case_id), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/run_all", methods=["POST"])
def api_run_all():
    cases = investigator_main.load_case_pack(investigator_main.CASE_PACK_PATH)
    job_id = _new_job(total=len(cases))
    threading.Thread(target=_run_all_job, args=(job_id,), daemon=True).start()
    return jsonify({"job_id": job_id, "total": len(cases)})


@app.route("/api/job_status")
def api_job_status():
    # Bug fix (#3): dedicated status endpoint backed by a shared dict, so the
    # dashboard can poll real progress instead of diffing /api/cases.
    job_id = request.args.get("job_id")
    if not job_id or job_id not in _jobs:
        return jsonify({"error": "unknown job_id"}), 404
    return jsonify(_jobs[job_id])


if __name__ == "__main__":
    CASES_DIR.mkdir(exist_ok=True, parents=True)
    app.run(host="0.0.0.0", port=5050, debug=True, use_reloader=False)