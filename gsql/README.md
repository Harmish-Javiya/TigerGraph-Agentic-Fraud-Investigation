# TigerGraph GSQL assets

The application expects the existing `fraud_investigation` graph and its eight
read/write investigation queries to be installed. `investigation_lifecycle.gsql`
is the reproducible extension added by this repository: it creates append-only
graph events for each case's evidence, recommendation transition, and decision.

Run it in GraphStudio's GSQL editor or with the `gsql` command-line client:

```gsql
@investigation_lifecycle.gsql
```

Then confirm the agent can discover the query:

```powershell
venv314\Scripts\python.exe agent\test_mcp.py
```

The output must contain `write_investigation_event`. Do not run this schema
change more than once against the same graph; TigerGraph will report that the
vertex and edge already exist.
