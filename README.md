# TigerGraph Agentic Fraud Investigation

An investigation agent for the TigerGraph Hacker House Goa fraud challenge. It
investigates a transaction alert, retrieves graph evidence and similar cases,
assesses fraud risk, records any simulated evidence request, applies a
deterministic policy, and produces one schema-validated JSON case file.

The system is designed to make the investigation traceable:

```text
Trigger -> TigerGraph / MCP evidence -> GraphRAG + LLM assessment
        -> customer or analyst evidence request when uncertain
        -> deterministic R1-R10 policy -> case JSON + graph audit events
        -> local semantic case memory -> analyst dashboard
```

## What is included

- `agent/investigator.py` - case workflow, evidence grounding, policy handoff,
  graph lifecycle events, and GraphRAG memory updates.
- `agent/graph_client.py` - TigerGraph MCP query client.
- `agent/graphrag.py` - local MiniLM vector index over closed and newly run
  investigation cases.
- `agent/policy_engine.py` - deterministic Fraud Policy v1.0 rules R1-R10.
- `agent/answer_schema.py` - Pydantic contract for every case JSON file.
- `server.py` and `static/index.html` - Flask analyst dashboard with case
  timeline, evidence, requests, actions, and raw JSON inspection.
- `gsql/investigation_lifecycle.gsql` - reproducible graph audit-event schema
  and query used by the agent.
- `HHGOA_IEEE_DATASETS/` - provided benchmark data and policy README.

## Prerequisites

1. **Python 3.11+**. The repository currently includes a Windows virtual
   environment at `venv314`; recreating one with Python 3.11 or 3.12 is also
   supported.
2. **TigerGraph Community Edition or Savanna** with a graph named
   `fraud_investigation`.
3. **TigerGraph MCP** running on `http://localhost:8000/mcp/`.
4. At least one LLM provider. The investigator uses the provider chain in
   `agent/llm_router.py`: Gemini first, then local Ollama/Qwen, then Groq.
5. Internet access on the first GraphRAG run so SentenceTransformers can
   download `all-MiniLM-L6-v2` if it is not already cached.

## Python setup

From PowerShell at the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install flask python-dotenv groq mcp httpx openai pydantic numpy requests sentence-transformers
```

If using the checked-in environment, substitute `venv314\Scripts\python.exe`
for `.venv\Scripts\python.exe` in the commands below. Do not commit API keys
or replace the supplied dataset files.

## Configuration

Create `.env` in the project root and `agent/.env` as needed. Keep both files
out of version control.

```dotenv
GROQ_API_KEY=your_key
TG_HOST=http://localhost
TG_RESTPP_PORT=9000
TG_GRAPHNAME=fraud_investigation
TG_USERNAME=tigergraph
TG_PASSWORD=your_password
```

Optional model settings:

```dotenv
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.5-flash-lite
LOCAL_BASE_URL=http://localhost:11434/v1
LOCAL_MODEL=qwen3:4b-instruct
GROQ_MODEL=openai/gpt-oss-120b
```

Set the credentials for at least one provider. When `GEMINI_API_KEY` is set,
Gemini is selected first; if it fails, the router falls through to local Ollama
and then Groq. The active investigation path no longer uses `groq/compound`.

## TigerGraph setup

The base graph must contain the investigation entities described in
`PROGRESS/progress.md` and these installed queries:

```text
get_transaction              card_baseline
card_window                  device_neighbors
region_neighbors             similar_closed_cases
customer_cards               write_investigation_case
```

Install the repository's audit extension after the base graph is ready:

```gsql
@gsql/investigation_lifecycle.gsql
```

It adds `InvestigationEvent` vertices plus `HAS_EVENT` edges and installs
`write_investigation_event`. Every run then writes three compact, append-only
events: `evidence`, `recommendations`, and `decision`.

Start the MCP service (adjust the Python executable if using `.venv`):

```powershell
venv314\Scripts\python.exe -m tigergraph_mcp run --host 0.0.0.0 --port 8000
```

Confirm that all expected queries are available:

```powershell
venv314\Scripts\python.exe agent\test_mcp.py
```

The output must include `write_investigation_event` before relying on graph
audit persistence.

## Run an investigation

Run one case:

```powershell
Set-Location agent
..\venv314\Scripts\python.exe main.py HHG-012
```

Run the full case pack:

```powershell
Set-Location agent
..\venv314\Scripts\python.exe main.py
```

Outputs are written to `cases/<case_id>.json`. Each result includes:

- assessment, evidence, linked transaction/card/device IDs, and retrieved
  historical cases;
- evidence requests plus the recorded simulated response, when needed;
- initial and final policy actions with approval route;
- SAR decision and narrative when filing is required;
- a stop reason consistent with the final state.

After a run, the final result is upserted into `agent/vector_store.pkl`. A
rerun replaces that case's previous semantic-memory record instead of adding a
duplicate.

### Output integrity guardrails

The LLM assesses pattern and risk, but it is not trusted to author
submission-critical facts. Final evidence is rebuilt from retrieved graph
results, customer-response records, and case-memory IDs. Summaries and SAR
narratives are generated from those same validated facts; policy actions,
approval routes, exposure, and connected-card IDs are deterministic. Regenerate
existing JSON files after upgrading this code, because earlier files do not gain
these safeguards automatically.

## Dashboard

In a second PowerShell terminal at the repository root:

```powershell
venv314\Scripts\python.exe server.py
```

Open [http://localhost:5050](http://localhost:5050). The dashboard lets you:

- list saved cases and start one or all investigations;
- follow background-job progress;
- inspect an investigation timeline, evidence, requests, decision, and
  before/after action recommendations;
- open raw JSON only when detailed diagnostics are needed.

## Validation

Run policy tests after policy changes:

```powershell
venv314\Scripts\python.exe -X utf8 agent\policy_engine.py
```

Compile the agent code:

```powershell
venv314\Scripts\python.exe -m py_compile agent\investigator.py agent\policy_engine.py agent\graph_client.py agent\graphrag.py
```

## Operational notes

- The agent recommends L1/L2 actions; it does not execute card blocks or SAR
  filings.
- `None` is not treated as a customer non-response. R4 is only applied after a
  recorded validation request receives an explicit no-response outcome.
- A customer confirmation deterministically resolves the case as fraud or
  legitimate; an uncertain verdict cannot be emitted with a closed status.
- Do not use public IEEE-CIS labels to recover outcomes for the supplied
  transformed benchmark. See `HHGOA_IEEE_DATASETS/README.md` for the challenge
  rules and answer contract.
