PROJECT: TigerGraph Agentic Fraud Investigation

GOAL
Build a fraud-investigation agent for the TigerGraph Hacker House Goa challenge. For each benchmark case, it must gather graph evidence, classify fraud pattern/risk, request additional evidence when uncertain, apply Fraud Policy R1-R10, write the case to TigerGraph, maintain case memory, and produce a strict JSON answer file.

DATA
- Dataset: HHGOA_IEEE_DATASETS/
- 590,742 transactions, 13,564 customers, 9,706 device profiles
- 5,565 historical closed cases
- 20 benchmark cases in case_pack.csv
- No true fraud label is available for benchmark cases.
- Never use public IEEE-CIS/Kaggle data to recover benchmark outcomes.

ARCHITECTURE
Trigger
-> TigerGraph MCP graph queries
-> GraphRAG policy + historical-case context
-> LLM fraud assessment
-> deterministic policy engine
-> simulated customer/analyst/step-up evidence when needed
-> JSON case output + TigerGraph writes + local vector-memory update
-> Flask analyst dashboard

KEY FILES
- agent/investigator.py
  Main workflow. Gets graph evidence, invokes LLM, grounds results, requests evidence for unresolved cases, applies policy, creates CaseAnswer, writes graph lifecycle events, updates GraphRAG memory.
- agent/policy_engine.py
  Deterministic R1-R10 policy rules.
- agent/answer_schema.py
  Pydantic JSON contract.
- agent/graph_client.py
  TigerGraph MCP query calls.
- agent/graphrag.py
  Local MiniLM vector retrieval over historical and newly completed cases.
- server.py
  Flask backend; dashboard API and background case jobs.
- static/index.html
  Dashboard: case table, progress, investigation timeline, evidence, requests, initial/final actions, decision rationale.
- cases/
  Generated answer JSON files.
- HHGOA_IEEE_DATASETS/README.md
  Authoritative challenge rules, policy, answer format, and dataset definitions.
- README.md
  Root setup and run guide.
- gsql/investigation_lifecycle.gsql
  Adds graph audit events for evidence, recommendations, and final decision.

TIGERGRAPH REQUIREMENTS
Graph name: fraud_investigation

Existing expected installed queries:
- get_transaction
- card_baseline
- card_window
- device_neighbors
- region_neighbors
- similar_closed_cases
- customer_cards
- write_investigation_case

New query to install:
- write_investigation_event

Run this in GSQL after the base graph exists:
@gsql/investigation_lifecycle.gsql

It creates:
- InvestigationEvent vertex
- HAS_EVENT edge from InvestigationCase to InvestigationEvent
- write_investigation_event query

MCP runs at:
http://localhost:8000/mcp/

ENVIRONMENT
Required environment variables:
- GROQ_API_KEY
- TG_HOST=http://localhost
- TG_RESTPP_PORT=9000
- TG_GRAPHNAME=fraud_investigation
- TG_USERNAME
- TG_PASSWORD

Optional model settings:
- GEMINI_API_KEY / GEMINI_MODEL
- LOCAL_BASE_URL / LOCAL_MODEL
- GROQ_MODEL

IMPORTANT BEHAVIOR / FIXES ALREADY MADE
1. Uncertain cases (0.15 < probability < 0.85) always record a simulated customer-validation request.
2. Customer confirmation of legitimacy produces:
   - verdict=legitimate
   - status=closed_legitimate
   - fraud_probability=0.05
   - no affected fraud transaction IDs
   - exposure=0
3. `None` does NOT mean “customer did not respond.”
   R4 is applied only when a validation request exists and explicit response is false.
4. A closed-legitimate status cannot coexist with an uncertain verdict.
5. If no additional evidence was requested, initial and final actions are identical as required by the dataset README.
6. Stop reasons are generated from actual final state, not untrusted LLM prose.
7. LLM-created fake graph-query names are normalized to actual query references.
8. Policy paraphrases from the LLM are excluded as factual evidence; policy reasoning belongs in action reasons.
9. Each run writes compact graph audit events:
   - evidence
   - recommendations
   - decision
10. Each completed result upserts into agent/vector_store.pkl for later semantic retrieval.

POLICY CONSTRAINTS
- R1: verify before blocking when probability < 0.70 on weak evidence.
- R2: customer denial -> block card + create case; report when policy conditions apply.
- R3: customer confirmation -> close no fraud.
- R4: actual no response after 24h -> monitor + decline pending authorization.
- R5: card testing -> decline + step-up; block if >$100 cleared.
- R6: shared origin -> monitor linked cards + report.
- R7: recurring legitimate dispute -> verify + warn, do not block.
- R8: uncertain/high exposure or conflicting evidence -> escalate.
- R9: undocumented coordinated abuse -> report + escalate.
- R10: block all cards only with at least two confirmed compromised cards or confirmed credential compromise.
- L1/L2 actions are recommendations, not autonomously executed.

RUN COMMANDS
Start MCP:
venv314\Scripts\python.exe -m tigergraph_mcp run --host 0.0.0.0 --port 8000

Test MCP:
venv314\Scripts\python.exe agent\test_mcp.py

Run one case:
cd agent
..\venv314\Scripts\python.exe main.py HHG-012

Run all cases:
cd agent
..\venv314\Scripts\python.exe main.py

Run dashboard:
venv314\Scripts\python.exe server.py

Dashboard:
http://localhost:5050

VALIDATION
venv314\Scripts\python.exe -X utf8 agent\policy_engine.py
venv314\Scripts\python.exe -m py_compile agent\investigator.py agent\policy_engine.py agent\graph_client.py agent\graphrag.py

CURRENT CAVEATS
- Base TigerGraph schema/loading/query GSQL source is not fully stored in this repository; PROGRESS/progress.md documents the deployed graph. The new lifecycle GSQL is included.
- The dashboard is improved but does not yet render an interactive TigerGraph relationship visualization.
- External enrichment APIs are optional and not yet connected.
- Do not expose .env values or commit API keys.
- Existing/generated files under cases/ may contain user-created changes; preserve them unless explicitly asked to regenerate or replace them.