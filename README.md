# BioAgent

An autonomous bioinformatics AI analyst built with LangGraph and Claude. Given a sample ID, BioAgent fetches live concordance and reproducibility data from the Biomarker Concordance Pipeline API, reasons about the findings, conditionally searches PubMed for relevant literature, and produces a structured clinical-grade quality report -- streamed in real time through a Streamlit chat interface.

[![CI](https://github.com/gbadedata/bioagent/actions/workflows/ci.yml/badge.svg)](https://github.com/gbadedata/bioagent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-purple)
![Claude](https://img.shields.io/badge/Claude-Sonnet%204-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Table of contents

- [What this project does](#what-this-project-does)
- [Why LangGraph and not a simple agent](#why-langgraph-and-not-a-simple-agent)
- [Architecture](#architecture)
- [The graph in detail](#the-graph-in-detail)
- [Tools](#tools)
- [PubMed keyword strategy](#pubmed-keyword-strategy)
- [Graceful degradation](#graceful-degradation)
- [Streaming and the async/Streamlit challenge](#streaming-and-the-asyncstreamlit-challenge)
- [Quick start](#quick-start)
- [API reference](#api-reference)
- [Dashboard](#dashboard)
- [CI pipeline](#ci-pipeline)
- [Development](#development)
- [Design decisions](#design-decisions)
- [Companion project](#companion-project)

---

## What this project does

BioAgent does three things autonomously -- without human intervention at each step:

**Quality report generation.** Fetches concordance and reproducibility data from the Biomarker Concordance Pipeline API, interprets metrics against GIAB HG001 v4.2.1 benchmarks, and writes a structured Markdown QC report.

**Anomaly detection and explanation.** Identifies runs below threshold, reasons about likely causes using Claude, and recommends specific remediation actions.

**Literature-contextualised interpretation.** Constructs targeted PubMed search queries derived from the actual metric values, retrieves relevant papers, and cites them in the report.

---

## Why LangGraph and not a simple agent

A simple LangChain agent with a tools list processes one question and stops. BioAgent needs to do multi-step conditional reasoning:

- Fetch API data, then decide based on what it finds whether to search literature
- If PubMed returns irrelevant results, reformulate the query and retry
- If the pipeline API is unreachable, degrade gracefully rather than hallucinating

This requires a stateful graph with cycles and conditional routing -- exactly what LangGraph provides. LangGraph models the agent as a directed graph where each node is a reasoning or action step, and conditional edges determine which node runs next based on the current state.

The key distinction: LangGraph agents are bounded. Every cycle has a maximum retry count. The graph cannot loop indefinitely. This is a critical property for a system that consumes paid API credit.

---

## Architecture

```
User input (sample_id)
        |
        v
  [ fetch_data ]  -- calls 5 pipeline API tools in sequence
        |
        +-- critical tools failed AND retries exhausted
        |         |
        |         v
        |   [ graceful_degradation ] --> END
        |
        +-- critical tools failed AND retries remain
        |         |
        |         +---> [ fetch_data ]  (cycle, max 1 retry)
        |
        +-- data collected
              |
              v
         [ analyse ]  -- LLM constructs PubMed search query from metric values
              |
              v
   [ search_literature ]  -- calls PubMed, retries with broader query if empty
              |             (max 2 retries)
              v
   [ synthesise_report ]  -- LLM writes structured QC report from all data
              |
              v
             END
```

The report synthesis step is pure LLM reasoning -- not a tool call. This distinction matters: tools do deterministic things (call APIs, query databases). The LLM interprets, explains, and writes.

---

## The graph in detail

### State

The graph maintains a `TypedDict` state object across all nodes:

```python
class AgentState(TypedDict):
    messages:             list           # Conversation history
    sample_id:            str
    task:                 str
    runs_data:            dict | None    # From get_pipeline_runs
    concordance_summary:  dict | None    # From get_concordance_summary
    concordance_details:  list | None    # From get_concordance_results
    reproducibility_data: dict | None    # From get_reproducibility
    alerts_data:          list | None    # From get_active_alerts
    pubmed_citations:     list | None    # From search_pubmed
    fetch_retries:        int            # Bounded at MAX_FETCH_RETRIES (1)
    pubmed_retries:       int            # Bounded at MAX_PUBMED_RETRIES (2)
    failed_tools:         list[str]      # Tools that returned errors
    tools_called:         list[str]      # Audit trail
    report:               str | None     # Final output
    status:               str            # running | complete | degraded
```

### Conditional routing

After `fetch_data`, the router inspects state to decide the next node:

```python
def route_after_fetch(state: AgentState) -> str:
    critical = {"get_concordance_summary", "get_pipeline_runs"}
    critical_failed = critical.intersection(set(state["failed_tools"]))

    if critical_failed and state["fetch_retries"] >= MAX_FETCH_RETRIES:
        return "graceful_degradation"
    if critical_failed and state["fetch_retries"] < MAX_FETCH_RETRIES:
        return "fetch_data"          # cycle back
    return "analyse"
```

---

## Tools

Six deterministic functions. Each returns a structured dict with a `success` flag. The agent uses these flags for routing decisions. No tool does any reasoning -- that is the LLM's job.

| Tool | Endpoint | On failure |
|---|---|---|
| `get_pipeline_runs` | `GET /api/v1/runs?sample_id={id}` | Returns `{"success": False, "error": "..."}` |
| `get_concordance_summary` | `GET /api/v1/concordance/summary/{id}` | Returns structured error |
| `get_concordance_results` | `GET /api/v1/concordance?limit=50` | Returns structured error |
| `get_reproducibility` | `GET /api/v1/reproducibility/{id}/latest` | Returns structured error |
| `get_active_alerts` | `GET /api/v1/alerts?unresolved_only=true` | Returns structured error |
| `search_pubmed` | NCBI Entrez esearch + efetch | Returns `{"success": True, "data": []}` with explanation |

All tools use a 10-second timeout. Connection errors, timeouts, and HTTP errors are all caught and returned as structured errors rather than exceptions.

---

## PubMed keyword strategy

The `analyse` node constructs PubMed search queries from actual metric values, not generic terms. This ensures citations are relevant to the specific findings.

| Finding | Query constructed |
|---|---|
| SNV F1 below threshold | `germline variant calling sensitivity specificity GIAB` |
| Indel F1 below threshold | `indel calling accuracy short read sequencing` |
| ICC below 0.90 | `intraclass correlation coefficient sequencing reproducibility` |
| VAF CV above 15% | `variant allele frequency technical variation replicate` |
| All metrics passing | `germline variant calling quality validation clinical` |

**Fallback strategy:** If the constructed query returns zero results, the agent retries with `germline variant calling sequencing quality` (broader). If that also returns nothing, it proceeds without citations rather than citing irrelevant papers. Maximum 2 retries total.

---

## Graceful degradation

If the Biomarker Concordance Pipeline API is unreachable, BioAgent does not hallucinate. It enters the `graceful_degradation` node which:

1. Reports exactly which tools failed and why
2. Lists what data could not be retrieved
3. Tells the user to start the API with the exact command
4. Exits cleanly

The report never contains invented metric values. A system that generates clinical-looking reports based on no data is dangerous.

```
I was unable to complete the analysis for sample HG001.

The following tools failed to return data:
- get_pipeline_runs
- get_concordance_summary

To start the API, run this in your terminal:

    cd ~/biomarker-concordance-pipeline
    source .venv/bin/activate
    export DATABASE_URL='postgresql+asyncpg://biomarker:biomarker@localhost:5432/biomarker'
    uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## Streaming and the async/Streamlit challenge

Streamlit's execution model reruns the entire script on each user interaction. Running an async LangGraph stream inside Streamlit hits `RuntimeError: This event loop is already running` immediately.

The solution is `nest_asyncio` combined with running the agent in a separate thread:

```python
import nest_asyncio
nest_asyncio.apply()

thread = threading.Thread(target=_run_agent, daemon=True)
thread.start()

# Animate progress while agent runs
while thread.is_alive():
    status_placeholder.info(f"Running: {steps[step_idx % len(steps)]}...")
    time.sleep(1.5)
    step_idx += 1
```

The progress animation runs in the main Streamlit thread while the agent runs in a daemon thread. When the agent completes, the result is written to a shared dict and the main thread renders it.

---

## Quick start

### Prerequisites

- Python 3.12
- Biomarker Concordance Pipeline API running on port 8000 (see companion project)
- Anthropic API key from `https://console.anthropic.com`

### Installation

```bash
git clone https://github.com/gbadedata/bioagent.git
cd bioagent

python3 -m venv .venv
source .venv/bin/activate

pip install \
  langchain langchain-anthropic langchain-core langgraph anthropic \
  biopython httpx fastapi uvicorn \
  "pydantic>=2.7" "pydantic-settings>=2.3" \
  structlog tenacity python-dotenv nest-asyncio \
  streamlit
```

### Configuration

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY
```

### Start the companion pipeline API first

```bash
cd ~/biomarker-concordance-pipeline
source .venv/bin/activate
export DATABASE_URL='postgresql+asyncpg://biomarker:biomarker@localhost:5432/biomarker'
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Run the Streamlit dashboard

```bash
cd ~/bioagent
source .venv/bin/activate
streamlit run dashboard/app.py --server.port 8501
```

Open `http://localhost:8501` and type `analyse HG001`.

### Run the BioAgent API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001
```

---

## API reference

### `POST /api/v1/agent/analyse`

Trigger an autonomous analysis run. Returns a job ID immediately (HTTP 202).

```bash
curl -X POST http://localhost:8001/api/v1/agent/analyse \
  -H "Content-Type: application/json" \
  -d '{"sample_id": "HG001", "task": "full_analysis"}'
```

```json
{
  "job_id": "a3f8c2d1-...",
  "status": "queued",
  "message": "Analysis queued for sample 'HG001'. Poll GET /api/v1/agent/results/a3f8c2d1-..."
}
```

### `GET /api/v1/agent/results/{job_id}`

Poll for the completed report.

```json
{
  "job_id": "a3f8c2d1-...",
  "sample_id": "HG001",
  "status": "complete",
  "report": "## Quality Report: HG001\n\n### Executive Summary\n...",
  "pubmed_citations": [
    {"pmid": "31039644", "url": "https://pubmed.ncbi.nlm.nih.gov/31039644/"}
  ],
  "tools_called": ["get_pipeline_runs", "get_concordance_summary", "..."],
  "alerts_found": 0,
  "duration_seconds": 14.7
}
```

### `GET /api/v1/agent/jobs`

List all submitted jobs and their statuses.

### `GET /health`

```json
{"status": "ok", "service": "bioagent", "version": "1.0.0"}
```

**Authentication note:** No authentication is implemented. This is intentional for local portfolio use. Production deployment requires an `X-API-Key` header validated against a secrets store.

---

## Dashboard

The Streamlit dashboard supports three conversation modes:

**Mode 1 -- Analyse a sample:**
```
User:  analyse HG001
Agent: [progress animation while agent runs]
       [full quality report with PubMed citations]
       [expandable tool call trace]
```

**Mode 2 -- Ask a follow-up:**
```
User:  why is the indel F1 lower than the SNV F1?
Agent: [retrieves concordance data, reasons about indel calling difficulty, cites literature]
```

**Mode 3 -- Threshold query:**
```
User:  what would happen if I set the minimum indel F1 to 0.97?
Agent: [reasons about current indel F1 of 0.9656, explains it would fail, recommends action]
```

**Cost guardrail:** The dashboard tracks runs per session and warns after 20 analyses (approximately $1.00 in API credit). This prevents accidental credit drain during a demo.

**Tool call trace:** Every response includes an expandable panel showing each tool called and a human-readable summary of what it returned. Raw API responses including internal UUIDs and timestamps are never shown to the user.

---

## CI pipeline

```
Lint and test (runs on every push)
    Set up Python 3.12
    Install all dependencies
    Lint with ruff
    Run tests (tools and API, LLM mocked)

Docker build (runs after lint-and-test)
    Build multi-stage Dockerfile
    Smoke test -- import succeeds
```

The graph tests are excluded from CI because they require a full LangGraph state machine with mocked LLM, which adds complexity without value in a headless environment. Run them locally with `pytest tests/test_graph.py`.

---

## Development

```bash
# Run all tests
pytest tests/ -v --tb=short

# Run only tool tests (no mocked LLM needed)
pytest tests/test_tools.py -v

# Run graph tests (mocked LLM)
pytest tests/test_graph.py -v

# Lint
ruff check agent/ api/ tests/

# Run dashboard in development mode
streamlit run dashboard/app.py --server.port 8501
```

---

## Design decisions

**Why LangGraph over a plain LangChain agent?** Plain agents process one turn and stop. LangGraph supports cycles, conditional routing, and bounded retries. For an agent that needs to retry failed tool calls and reformulate search queries based on results, a graph is the correct abstraction.

**Why separate the API from the dashboard?** The FastAPI endpoint is what a production pipeline scheduler would call -- a programmatic interface with job IDs and polling. The Streamlit dashboard is what a scientist would use interactively. Keeping them separate means each can be deployed independently.

**Why background tasks in the API instead of synchronous execution?** A full agent run takes 15 to 30 seconds. Default HTTP clients time out at 30 seconds. Returning a job ID immediately (HTTP 202) and polling for results avoids timeout errors and is the correct pattern for long-running operations.

**Why `nest_asyncio` instead of a native async Streamlit solution?** Streamlit does not natively support async execution at the script level. `nest_asyncio` patches the running event loop to allow nested `asyncio.run()` calls. Combined with threading, this is the cleanest solution that does not require restructuring the entire application.

**Why bounded retries instead of dynamic termination?** Dynamic termination (stop when the LLM says it has enough data) requires an additional LLM call per cycle to evaluate completeness. Bounded retries (stop after N attempts regardless) are deterministic, cheaper, and safer. For a system consuming paid API credit, deterministic behaviour is essential.

---

## Companion project

BioAgent is designed to work with the Biomarker Concordance Pipeline:

**[github.com/gbadedata/biomarker-concordance-pipeline](https://github.com/gbadedata/biomarker-concordance-pipeline)**

That project provides:
- The Nextflow DSL2 germline variant calling pipeline
- The concordance and reproducibility analysis engine
- The FastAPI REST API that BioAgent uses as its primary data source
- The PostgreSQL database seeded with HG001 benchmark data

---

## Licence

MIT
