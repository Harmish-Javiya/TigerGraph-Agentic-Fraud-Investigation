# TigerGraph Agentic Fraud Investigation — Progress Log

## Project Overview
**Challenge:** Hacker House Goa 2026 — TigerGraph Agentic Fraud Investigation  
**Dataset:** IEEE-CIS Fraud Detection (HHGOA_IEEE)  
**Goal:** Build an AI agent that investigates 20 fraud cases and produces structured JSON answer files  
**Stack:** TigerGraph CE 4.3.0 (Docker) + GSQL + Python Agent + TigerGraph MCP  

---

## ✅ Phase 1 — Infrastructure Setup

### TigerGraph Community Edition on Docker
- Downloaded TigerGraph CE 4.3.0-rc1 Docker image from `dl.tigergraph.com`
- Ran container with correct port mappings:
  ```
  docker run -d \
    --name tigergraph \
    -p 14240:14240 \
    -p 9000:9000 \
    -p 9002:9002 \
    --ulimit nofile=1000000:1000000 \
    tigergraph/community:4.3.0-rc1
  ```
- GraphStudio accessible at `http://localhost:14240`
- Credentials: `tigergraph / tigergraph`

---

## ✅ Phase 2 — Graph Schema

### Graph Created
- Name: `fraud_investigation`

### Vertices (8)

| Vertex | Primary Key | Key Attributes |
|---|---|---|
| `Customer` | `customer_id` | e.g. `C12382` |
| `Card` | `card_id` | e.g. `C12382-K1`, card1-6 |
| `Transaction` | `txn_id` | ts, amount, product_cd, channel, risk_score, addr1-2, dist1-2, p_email, r_email, C1-14, D1-15, M1-9 |
| `DeviceProfile` | `device_key` | Concatenation of `DeviceInfo|id_30|id_31|id_33` |
| `BillingRegion` | `region_code` | From `addr1` column |
| `EmailDomain` | `domain` | From `P_emaildomain` / `R_emaildomain` |
| `ClosedCase` | `case_id` | All closed investigation fields including analyst_notes |
| `InvestigationCase` | `case_id` | Agent writes findings here during investigation |

### Edges (11)

| Edge | From | To | Notes |
|---|---|---|---|
| `OWNS` | Customer | Card | |
| `MADE` | Card | Transaction | Via customer_id (transactions.csv has no card_id) |
| `FROM_DEVICE` | Transaction | DeviceProfile | Online transactions only |
| `BILLED_IN` | Transaction | BillingRegion | From addr1 |
| `PURCHASER_EMAIL` | Transaction | EmailDomain | |
| `RECIPIENT_EMAIL` | Transaction | EmailDomain | |
| `INVOLVES_TXN` | ClosedCase | Transaction | |
| `ON_CARD` | ClosedCase | Card | |
| `CONNECTED_TO` | ClosedCase | Card | Connected compromised cards |
| `CASE_ON_CARD` | InvestigationCase | Card | |
| `CASE_TXN` | InvestigationCase | Transaction | |

### Schema Creation Method
Used `SCHEMA_CHANGE JOB` — the correct atomic approach for TigerGraph:
```sql
CREATE SCHEMA_CHANGE JOB init_schema FOR GRAPH fraud_investigation { ... }
RUN SCHEMA_CHANGE JOB init_schema
```

---

## ✅ Phase 3 — Data Loading

### Key Discovery: card_id vs customer_id
- `transactions.csv` has `customer_id` (e.g. `C12382`) but **no `card_id`**
- `closed_cases_history.csv` and `case_pack.csv` have `card_id` (e.g. `C12382-K1`)
- `-K1`, `-K2` suffix = one customer can have multiple cards
- Card vertices built from closed_cases + case_pack, not from transactions
- `MADE` edge uses `customer_id` as the card key from transactions

### DeviceProfile Key Fix
- `DeviceInfo` alone is not unique (many devices share same model)
- Preprocessed `identity.csv` with Python to create composite key:
  ```python
  device_key = DeviceInfo + '|' + id_30 + '|' + id_31 + '|' + id_33
  ```
- Generated two files: `device_profiles.csv` and `txn_device.csv`

### Final Graph Counts

| Vertex | Count |
|---|---|
| Transaction | 590,742 |
| Customer | 13,564 |
| Card | 15,481 |
| ClosedCase | 5,565 |
| InvestigationCase | 20 |
| BillingRegion | 332 |
| EmailDomain | 60 |
| DeviceProfile | 9,706 |

### Loading Jobs Created
1. `load_closed_cases` — from `closed_cases_history.csv`
2. `load_case_pack` — from `case_pack.csv`
3. `load_transactions` — from `transactions.csv` (590k rows, 15 seconds)
4. `load_cards_from_cases` — Card vertices + OWNS edges from closed cases
5. `load_cards_from_casepack` — Card vertices + OWNS edges from case pack
6. `load_devices` — DeviceProfile vertices from preprocessed file
7. `load_txn_device` — FROM_DEVICE edges

---

## ✅ Phase 4 — GSQL Investigation Queries

All 8 queries installed and tested:

| Query | Parameters | Purpose |
|---|---|---|
| `get_transaction` | `txn_id` | Fetch single transaction + all attributes |
| `card_history` | `customer_id` | All transactions on a customer's cards (baseline) |
| `card_window` | `customer_id, center_ts, hours` | Transaction burst around a time window |
| `device_neighbors` | `device_key` | Other cards/cases sharing the same device |
| `region_neighbors` | `region_code, from_ts, days` | Cards active in same billing region |
| `similar_closed_cases` | `pattern_name, customer_id` | Past cases matching pattern or customer |
| `customer_cards` | `customer_id` | All cards a customer owns |
| `write_investigation_case` | all case fields | Agent writes findings back to graph |

### Key GSQL Lessons Learned
- Multi-hop traversal must be split into separate SELECT steps
- Reverse edge syntax: `(<EDGE_NAME)` not `(EDGE_NAME<-)`
- `PRIMARY_ID_AS_ATTRIBUTE="true"` — don't repeat the attribute name
- Use `SCHEMA_CHANGE JOB` not `USE GLOBAL` for clean schema creation
- One `CREATE` per block to isolate errors

### Manual Investigation of HHG-001 (Test Run)
- `flagged_txn_id`: 3514030
- `amount`: $77.07
- `channel`: **in_person** (ProductCD = W, no device record)
- `risk_score`: 0.61
- `addr1`: 444 (billing region)
- `addr2`: 87 (home country)
- Match flags: M1=T, M2=T, M3=T, M5=F
- D1=82, D2=82 (82 days since last transaction — long gap)
- card_history query working — returns full transaction history for C12382

---

## 🔲 Phase 5 — Python Agent (Next)

### What to build
```
trigger → baseline → burst → shared-origin
        → case-memory retrieval → probability estimate
        → [evidence request if 0.15 < p < 0.85]
        → policy engine → JSON emit → write_case to graph
```

### Files to create
```
fraud-investigation-agent/
├── schema/
│   └── schema.gsql
├── queries/
│   ├── get_transaction.gsql
│   ├── card_history.gsql
│   ├── card_window.gsql
│   ├── device_neighbors.gsql
│   ├── region_neighbors.gsql
│   ├── similar_closed_cases.gsql
│   ├── customer_cards.gsql
│   └── write_investigation_case.gsql
├── data_prep/
│   └── preprocess_devices.py
├── agent/
│   ├── graph_client.py      ← TigerGraph REST API calls
│   ├── policy_engine.py     ← Deterministic R1-R10 rule logic
│   ├── investigator.py      ← Main agent loop
│   └── answer_schema.py     ← Pydantic models for JSON output
├── cases/
│   └── (20 answer JSON files go here)
└── README.md
```

### TigerGraph REST API endpoint
```
http://localhost:9000/restpp/query/fraud_investigation/<query_name>
```

### Policy rules to implement (R1-R10)
- R1: Verify before block if probability < 0.70
- R2: Customer denies → BLOCK_CARD + CREATE_CASE + FILE_REPORT if exposure > $1000
- R3: Customer confirms → CLOSE_NO_FRAUD
- R4: No reply 24h → MONITOR_CARD + DECLINE if exposure > $500
- R5: Card testing → DECLINE + STEP_UP_AUTH, BLOCK if > $100 cleared
- R6: Shared origin → CREATE_CASE + FILE_REPORT + MONITOR_CONNECTED_CARDS
- R7: Disputed but matches recurring pattern → VERIFY + WARN, no block
- R8: Uncertain + exposure > $500 → ESCALATE_TO_ANALYST
- R9: Undocumented pattern → CREATE_CASE + FILE_REPORT + ESCALATE
- R10: Never BLOCK_ALL_CARDS unless 2+ cards confirmed fraud

---

## 🔲 Phase 6 — Answer Files (Final)

- 20 JSON files in `cases/` folder
- One per case: HHG-001.json to HHG-020.json
- Each contains: `case`, `sar`, `next_best_actions`, `evidence_requests`
- Validate every ID exists in dataset before submitting
- Assert `sar.file == ("FILE_REPORT" in final actions)`

---

## Important Notes

### Dataset quirks
- `risk_score` is an input, never a verdict
- Half the 20 cases are legitimate — don't over-block
- `uncertain` verdict earns full credit on ambiguous cases
- Finding an undocumented pattern (outside 5 known) is scored
- Made-up IDs score zero — always use IDs from dataset

### Docker commands
```powershell
docker stop tigergraph    # Safe stop, data preserved
docker start tigergraph   # Restart
docker rm tigergraph      # DANGER — deletes everything
```

### Stopping criteria for agent
- Fraud probability ≥ 0.85 or ≤ 0.15 with 2+ independent evidence pieces
- Customer verification response received
- Further steps won't change the decision
