# BioAgent

An autonomous bioinformatics AI analyst built with LangGraph and Claude. Given a sample ID,
BioAgent fetches live concordance and reproducibility data from the Biomarker Concordance
Pipeline API, reasons about the findings, conditionally searches PubMed for relevant
literature, and produces a structured clinical-grade quality report, streamed in real time
through a Streamlit chat interface.

[![CI](https://github.com/gbadedata/bioagent/actions/workflows/ci.yml/badge.svg)](https://github.com/gbadedata/bioagent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-purple)
![Claude](https://img.shields.io/badge/Claude-Sonnet%204.6-orange)
![Tests](https://img.shields.io/badge/tests-45%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Table of contents

- [What this project does](#what-this-project-does)
- [What a run produces](#what-a-run-produces)
- [Why LangGraph and not a simple agent](#why-langgraph-and-not-a-simple-agent)
- [Architecture](#architecture)
- [A run, end to end](#a-run-end-to-end)
- [The graph in detail](#the-graph-in-detail)
- [Tools](#tools)
- [PubMed keyword strategy](#pubmed-keyword-strategy)
- [Graceful degradation](#graceful-degradation)
- [Streaming and the async and Streamlit challenge](#streaming-and-the-async-and-streamlit-challenge)
- [Quick start](#quick-start)
- [API reference](#api-reference)
- [Dashboard](#dashboard)
- [Continuous integration](#continuous-integration)
- [Testing](#testing)
- [Design decisions](#design-decisions)
- [Companion project](#companion-project)
- [Licence](#licence)

---

## What this project does

BioAgent does three things autonomously, without human intervention at each step:

**Quality report generation.** Fetches concordance and reproducibility data from the
Biomarker Concordance Pipeline API, interprets metrics against GIAB HG001 v4.2.1 benchmarks,
and writes a structured Markdown QC report.

**Anomaly detection and explanation.** Identifies runs below threshold, reasons about likely
causes using Claude, and recommends specific remediation actions.

**Literature-contextualised interpretation.** Constructs targeted PubMed search queries
derived from the actual metric values, retrieves the papers and their abstracts, and grounds
the report's literature section in what those abstracts actually say.

## What a run produces

A single `analyse HG001` produces a full quality report with executive summary, concordance
and reproducibility analysis, grounded literature context, numbered recommendations, and a
data-provenance footer, alongside an expandable trace of every tool the agent called.

![BioAgent quality report generated from a live pipeline run](docs/evidence/bioagent_full_report.png)

---

## Why LangGraph and not a simple agent

A simple LangChain agent with a tools list processes one question and stops. BioAgent needs
to do multi-step conditional reasoning:

- Fetch API data, then decide based on what it finds whether to search literature.
- If PubMed returns irrelevant results, reformulate the query and retry.
- If the pipeline API is unreachable, degrade gracefully rather than inventing data.

This requires a stateful graph with cycles and conditional routing, which is exactly what
LangGraph provides. LangGraph models the agent as a directed graph where each node is a
reasoning or action step, and conditional edges decide which node runs next based on the
current state.

The key property is that the graph is **bounded**. Every cycle has a maximum retry count and
the graph cannot loop indefinitely, which matters for a system that consumes paid API credit.

---

## Architecture

One typed state object flows through five nodes. Deterministic tools gather data; the LLM
reasons, and a conditional router decides the path, including a bounded retry cycle and a
graceful-degradation exit.

```mermaid
flowchart TD
    START([sample_id]) --> FETCH["fetch_data<br/>call 5 pipeline API tools"]
    FETCH -->|data collected| ANALYSE["analyse<br/>LLM builds a targeted PubMed query"]
    FETCH -->|"critical tools failed,<br/>retry budget remains"| FETCH
    FETCH -->|"critical tools failed,<br/>retries spent"| DEGRADE["graceful_degradation<br/>report what failed, invent nothing"]
    ANALYSE --> SEARCH["search_literature<br/>query PubMed, broaden and retry if empty"]
    SEARCH --> REPORT["synthesise_report<br/>LLM writes the QC report"]
    REPORT --> DONE([END])
    DEGRADE --> DONE
```

The report synthesis step is pure LLM reasoning, not a tool call. Tools do deterministic
things (call APIs, query databases); the LLM interprets, explains, and writes.

## A run, end to end

```mermaid
sequenceDiagram
    participant U as User / API
    participant G as LangGraph
    participant P as Pipeline API
    participant C as Claude
    participant L as PubMed
    U->>G: analyse(sample_id)
    G->>P: runs, concordance, reproducibility, alerts
    P-->>G: metrics (or structured errors)
    alt critical data still missing after a retry
        G-->>U: graceful-degradation report (no invented data)
    else data collected
        G->>C: build a PubMed query from the metric values
        C-->>G: query
        G->>L: search, broaden and retry if empty
        L-->>G: citations and abstracts
        G->>C: synthesise the QC report from data and abstracts
        C-->>G: structured report
        G-->>U: report, citations, and tool trace
    end
```

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
    pubmed_citations:     list | None    # PMIDs and URLs from search_pubmed
    pubmed_query:         str | None     # Query the analyse node constructed
    pubmed_abstracts:     str | None     # Abstract text used to ground the report
    fetch_retries:        int            # Bounded at MAX_FETCH_RETRIES (1)
    pubmed_retries:       int            # Bounded at MAX_PUBMED_RETRIES (2)
    failed_tools:         list[str]      # Tools that returned errors
    tools_called:         list[str]      # Audit trail
    report:               str | None     # Final output
    status:               str            # running | complete | degraded
```

### Conditional routing

After `fetch_data`, the router inspects state to decide the next node. `fetch_data`
increments `fetch_retries` on every attempt, so the retry budget is finite and the
degradation exit is always reachable when the API stays down:

```python
def route_after_fetch(state: AgentState) -> str:
    critical = {"get_concordance_summary", "get_pipeline_runs"}
    critical_failed = critical.intersection(set(state["failed_tools"]))

    if critical_failed and state["fetch_retries"] > MAX_FETCH_RETRIES:
        return "graceful_degradation"
    if critical_failed and state["fetch_retries"] <= MAX_FETCH_RETRIES:
        return "fetch_data"          # retry, bounded by MAX_FETCH_RETRIES
    return "analyse"
```

---

## Tools

Six deterministic functions. Each returns a structured dict with a `success` flag, and the
agent uses those flags for routing decisions. No tool does any reasoning; that is the LLM's
job.

| Tool | Endpoint | On failure |
|---|---|---|
| `get_pipeline_runs` | `GET /api/v1/runs?sample_id={id}` | Returns `{"success": False, "error": "..."}` |
| `get_concordance_summary` | `GET /api/v1/concordance/summary/{id}` | Returns structured error |
| `get_concordance_results` | `GET /api/v1/concordance?limit=50` | Returns structured error |
| `get_reproducibility` | `GET /api/v1/reproducibility/{id}/latest` | Returns structured error |
| `get_active_alerts` | `GET /api/v1/alerts?unresolved_only=true` | Returns structured error |
| `search_pubmed` | NCBI Entrez esearch and efetch | Returns `{"success": True, "data": []}` with an explanation |

All pipeline tools use a 10-second timeout. Connection errors, timeouts, and HTTP errors are
caught and returned as structured errors rather than raised as exceptions. `search_pubmed`
returns the PMIDs, URLs, and abstract text, and that abstract text is what the report's
literature section is grounded in.

---

## PubMed keyword strategy

The `analyse` node constructs PubMed search queries from actual metric values, not generic
terms, so citations are relevant to the specific findings.

| Finding | Query constructed |
|---|---|
| SNV F1 below threshold | `germline variant calling sensitivity specificity GIAB` |
| Indel F1 below threshold | `indel calling accuracy short read sequencing` |
| ICC below 0.90 | `intraclass correlation coefficient sequencing reproducibility` |
| VAF CV above 15% | `variant allele frequency technical variation replicate` |
| All metrics passing | `germline variant calling quality validation clinical` |

**Fallback strategy.** If the constructed query returns zero results, the agent retries with
a broader `germline variant calling sequencing quality`. If that also returns nothing, it
proceeds without citations rather than citing irrelevant papers. Two retries at most.

---

## Graceful degradation

If the Biomarker Concordance Pipeline API is unreachable, BioAgent does not invent data. The
retry budget is spent, and the graph routes to the `graceful_degradation` node, which:

1. Reports exactly which tools failed and why.
2. Lists what data could not be retrieved.
3. Tells the user how to start the API, with the exact command.
4. Exits cleanly.

The report never contains invented metric values. A system that generates clinical-looking
reports from no data is dangerous, so this path is covered by a regression test that asserts
the degraded report contains no fabricated metrics.

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

## Streaming and the async and Streamlit challenge

Streamlit reruns the entire script on each user interaction. Running an async LangGraph
stream inside Streamlit hits `RuntimeError: This event loop is already running` immediately.

The solution is `nest_asyncio` combined with running the agent in a separate thread:

```python
import nest_asyncio
nest_asyncio.apply()

thread = threading.Thread(target=_run_agent, daemon=True)
thread.start()

# Animate progress while the agent runs
while thread.is_alive():
    status_placeholder.info(f"Running: {steps[step_idx % len(steps)]}...")
    time.sleep(1.5)
    step_idx += 1
```

The progress animation runs in the main Streamlit thread while the agent runs in a daemon
thread. When the agent completes, the result is written to a shared dict and the main thread
renders it.

---

## Quick start

### Prerequisites

- Python 3.12
- Biomarker Concordance Pipeline API running on port 8000 (see the companion project)
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

**Authentication note.** No authentication is implemented; this is intentional for local
portfolio use. A production deployment would require an `X-API-Key` header validated against
a secrets store.

---

## Dashboard

The Streamlit dashboard supports three conversation modes.

**Analyse a sample.**
```
User:  analyse HG001
Agent: [progress animation while the agent runs]
       [full quality report with PubMed citations]
       [expandable tool call trace]
```

**Ask a follow-up.**
```
User:  why is the indel F1 lower than the SNV F1?
Agent: [retrieves concordance data, reasons about indel calling difficulty, cites literature]
```

**Threshold query.**
```
User:  what would happen if I set the minimum indel F1 to 0.97?
Agent: [reasons about the current indel F1 of 0.9656, explains it would fail, recommends action]
```

**Cost guardrail.** The dashboard tracks runs per session and warns after 20 analyses
(roughly one dollar in API credit), which prevents accidental credit drain during a demo.

**Tool call trace.** Every response includes an expandable panel showing each tool called and
a human-readable summary of what it returned. Raw API responses, including internal UUIDs and
timestamps, are never shown to the user.

---

## Continuous integration

Every push runs lint and the full test suite (agent graph, tools, and API, with the LLM, the
pipeline API, and PubMed all mocked), then builds the Docker image and smoke-tests that it
imports. The graph tests run alongside the rest, since they are fully mocked and need no API
key or network.

![Continuous integration passing](docs/evidence/bioagent_ci_green.png)

---

## Testing

```bash
pytest tests/ -v --tb=short        # 45 tests: graph, tools, API
pytest tests/test_tools.py -v      # tools only, no mocked LLM needed
pytest tests/test_graph.py -v      # the agent graph, LLM mocked
ruff check agent/ api/ tests/      # lint
```

The suite covers the tools and their error handling, the FastAPI endpoints, and the graph
end to end with a mocked model, including the bounded retry cycle and a regression test that
the API-down path degrades gracefully rather than looping.

---

## Design decisions

**Why LangGraph over a plain LangChain agent?** Plain agents process one turn and stop.
LangGraph supports cycles, conditional routing, and bounded retries. For an agent that needs
to retry failed tool calls and reformulate search queries based on results, a graph is the
correct abstraction.

**Why separate the API from the dashboard?** The FastAPI endpoint is what a production
pipeline scheduler would call: a programmatic interface with job IDs and polling. The
Streamlit dashboard is what a scientist would use interactively. Keeping them separate lets
each be deployed independently.

**Why background tasks in the API instead of synchronous execution?** A full agent run takes
15 to 30 seconds, and default HTTP clients time out at 30 seconds. Returning a job ID
immediately (HTTP 202) and polling for the result avoids timeout errors and is the right
pattern for long-running operations.

**Why `nest_asyncio` instead of a native async Streamlit solution?** Streamlit does not
natively support async execution at the script level. `nest_asyncio` patches the running
event loop to allow nested `asyncio.run()` calls; combined with threading, this is the
cleanest solution that does not require restructuring the application.

**Why bounded retries instead of dynamic termination?** Dynamic termination (stop when the
LLM decides it has enough data) needs an extra LLM call per cycle to judge completeness.
Bounded retries stop after a fixed number of attempts, which is deterministic, cheaper, and
safer. For a system consuming paid API credit, deterministic behaviour is essential.

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

Released under the MIT License.
