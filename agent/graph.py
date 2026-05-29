"""
BioAgent LangGraph state machine.

Graph structure:
    START -> fetch_data -> analyse -> search_literature -> synthesise_report -> END
                ^                          |
                |  (retry, max 1)          | (retry, max 2)
                +-- data_incomplete        +-- results_irrelevant
                                           v
                                     graceful_degradation -> END

Key design decisions:
- Cycles are bounded: max 1 fetch retry, max 2 PubMed retries
- Graceful degradation fires when the pipeline API is unreachable
- Report synthesis is LLM reasoning, not a tool call
- State is typed via TypedDict for clarity and safety
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()
import logging
from datetime import datetime, timezone
from typing import Annotated, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from .prompts import ANALYSE_PROMPT, DEGRADATION_MESSAGE, KEYWORD_STRATEGY, SYSTEM_PROMPT
from .tools import ALL_TOOLS

logger = logging.getLogger(__name__)

MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "4096"))
MAX_PUBMED_RETRIES = int(os.environ.get("AGENT_MAX_PUBMED_RETRIES", "2"))
MAX_FETCH_RETRIES  = int(os.environ.get("AGENT_MAX_FETCH_RETRIES",  "1"))


# ─── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Conversation messages — add_messages merges rather than overwrites
    messages: Annotated[list, add_messages]

    # Core inputs
    sample_id:    str
    task:         str

    # Collected data
    runs_data:            dict | None
    concordance_summary:  dict | None
    concordance_details:  list | None
    reproducibility_data: dict | None
    alerts_data:          list | None
    pubmed_citations:     list | None

    # Control flow
    fetch_retries:   int
    pubmed_retries:  int
    failed_tools:    list[str]
    tools_called:    list[str]

    # Output
    report:          str | None
    status:          str   # running | complete | degraded


# ─── LLM setup ────────────────────────────────────────────────────────────────

llm = ChatAnthropic(
    model=MODEL,
    max_tokens=MAX_TOKENS,
    temperature=0,
    api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
)

llm_with_tools = llm.bind_tools(ALL_TOOLS)


# ─── Node: fetch_data ─────────────────────────────────────────────────────────

def fetch_data(state: AgentState) -> dict:
    """
    Call all five pipeline API tools to collect data for the sample.
    Tracks which tools failed for potential graceful degradation.
    """
    sample_id   = state["sample_id"]
    failed      = list(state.get("failed_tools", []))
    tools_called = list(state.get("tools_called", []))

    from .tools import (
        get_active_alerts,
        get_concordance_results,
        get_concordance_summary,
        get_pipeline_runs,
        get_reproducibility,
    )

    runs_result    = get_pipeline_runs.invoke({"sample_id": sample_id})
    conc_summary   = get_concordance_summary.invoke({"sample_id": sample_id})
    conc_details   = get_concordance_results.invoke({"sample_id": sample_id})
    repro_result   = get_reproducibility.invoke({"sample_id": sample_id})
    alerts_result  = get_active_alerts.invoke({"sample_id": sample_id})

    tools_called += ["get_pipeline_runs", "get_concordance_summary",
                     "get_concordance_results", "get_reproducibility", "get_active_alerts"]

    # Track failures
    new_failed = []
    if not runs_result.get("success"):   new_failed.append("get_pipeline_runs")
    if not conc_summary.get("success"):  new_failed.append("get_concordance_summary")
    if not conc_details.get("success"):  new_failed.append("get_concordance_results")
    if not repro_result.get("success"):  new_failed.append("get_reproducibility")
    if not alerts_result.get("success"): new_failed.append("get_active_alerts")

    return {
        "runs_data":            runs_result   if runs_result.get("success")   else None,
        "concordance_summary":  conc_summary  if conc_summary.get("success")  else None,
        "concordance_details":  conc_details.get("data") if conc_details.get("success") else None,
        "reproducibility_data": repro_result  if repro_result.get("success")  else None,
        "alerts_data":          alerts_result.get("data") if alerts_result.get("success") else [],
        "failed_tools":         failed + new_failed,
        "tools_called":         tools_called,
        "fetch_retries":        state.get("fetch_retries", 0),
    }


def route_after_fetch(state: AgentState) -> str:
    """
    After fetching data:
    - If critical tools all failed -> graceful_degradation
    - If some failed and retries remain -> fetch_data (retry)
    - Otherwise -> analyse
    """
    failed        = state.get("failed_tools", [])
    fetch_retries = state.get("fetch_retries", 0)

    critical = {"get_concordance_summary", "get_pipeline_runs"}
    critical_failed = critical.intersection(set(failed))

    if critical_failed and fetch_retries >= MAX_FETCH_RETRIES:
        return "graceful_degradation"

    if critical_failed and fetch_retries < MAX_FETCH_RETRIES:
        return "fetch_data"

    return "analyse"


# ─── Node: analyse ────────────────────────────────────────────────────────────

def analyse(state: AgentState) -> dict:
    """
    LLM reasons about collected data to construct a targeted PubMed query.
    Does not generate the final report -- that happens in synthesise_report.
    """
    conc = state.get("concordance_summary", {})
    repro = state.get("reproducibility_data", {})

    findings_parts = []
    if conc and conc.get("data"):
        d = conc["data"]
        findings_parts.append(f"SNV F1={d.get('snv_f1_mean', 'N/A')}")
        findings_parts.append(f"Indel F1={d.get('indel_f1_mean', 'N/A')}")
    if repro and repro.get("data"):
        d = repro["data"]
        findings_parts.append(f"ICC={d.get('icc', 'N/A')}")
        findings_parts.append(f"CV={d.get('median_cv', 'N/A')}%")

    findings = ", ".join(findings_parts) if findings_parts else "No data retrieved"

    keyword_prompt = KEYWORD_STRATEGY.format(findings=findings)
    response = llm.invoke([HumanMessage(content=keyword_prompt)])
    pubmed_query = response.content.strip().strip('"')

    logger.info("analyse_complete", pubmed_query=pubmed_query, findings=findings)

    return {
        "messages": [AIMessage(content=f"Analysis complete. PubMed query: {pubmed_query}")],
        "tools_called": state.get("tools_called", []) + ["analyse"],
        "_pubmed_query": pubmed_query,
    }


# ─── Node: search_literature ─────────────────────────────────────────────────

def search_literature(state: AgentState) -> dict:
    """
    Search PubMed using the query constructed by the analyse node.
    Retries with a broader query if results are irrelevant or empty.
    Bounded by MAX_PUBMED_RETRIES.
    """
    from .tools import search_pubmed

    pubmed_retries = state.get("pubmed_retries", 0)

    # Extract query from last AIMessage
    query = "germline variant calling quality validation clinical"
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and "PubMed query:" in msg.content:
            query = msg.content.split("PubMed query:")[-1].strip()
            break

    result = search_pubmed.invoke({"query": query, "max_results": 5})

    tools_called = state.get("tools_called", []) + ["search_pubmed"]

    if result.get("success") and result.get("data"):
        return {
            "pubmed_citations": result["data"],
            "pubmed_retries":   pubmed_retries,
            "tools_called":     tools_called,
            "messages": [AIMessage(content=f"PubMed: {result['summary']}")],
        }

    # No results or failure -- try broader query on retry
    if pubmed_retries < MAX_PUBMED_RETRIES:
        broader = "germline variant calling sequencing quality"
        result2 = search_pubmed.invoke({"query": broader, "max_results": 5})
        if result2.get("success") and result2.get("data"):
            return {
                "pubmed_citations": result2["data"],
                "pubmed_retries":   pubmed_retries + 1,
                "tools_called":     tools_called + ["search_pubmed_retry"],
                "messages": [AIMessage(content=f"PubMed retry: {result2['summary']}")],
            }

    # Proceed without citations
    return {
        "pubmed_citations": [],
        "pubmed_retries":   pubmed_retries + 1,
        "tools_called":     tools_called,
        "messages": [AIMessage(content="No relevant PubMed results found. Proceeding without citations.")],
    }


# ─── Node: synthesise_report ─────────────────────────────────────────────────

def synthesise_report(state: AgentState) -> dict:
    """
    LLM synthesises all collected data into a structured quality report.
    This is pure LLM reasoning -- not a tool call.
    """
    sample_id = state["sample_id"]

    conc_s   = state.get("concordance_summary")
    repro    = state.get("reproducibility_data")
    runs     = state.get("runs_data")
    alerts   = state.get("alerts_data") or []
    citations = state.get("pubmed_citations") or []

    runs_summary = runs.get("summary", "No run data") if runs else "No run data retrieved"
    conc_summary = conc_s.get("summary", "No concordance data") if conc_s else "No concordance data"

    details_list = state.get("concordance_details") or []
    conc_details = "\n".join([
        f"{r.get('variant_type')} F1={r.get('f1_score'):.4f}"
        for r in details_list if isinstance(r, dict)
    ]) or "No per-run detail available"

    repro_summary = repro.get("summary", "No reproducibility data") if repro else "No reproducibility data"

    if alerts:
        alerts_summary = "\n".join([f"[{a.get('severity','?').upper()}] {a.get('message','')}" for a in alerts])
    else:
        alerts_summary = "No active alerts."

    if citations:
        cit_lines = [f"PMID {c['pmid']}: {c['url']}" for c in citations]
        citations_text = "\n".join(cit_lines)
    else:
        citations_text = "No PubMed citations retrieved."

    run_count = runs.get("count", 0) if runs else 0
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tools_called = ", ".join(state.get("tools_called", []))

    prompt = ANALYSE_PROMPT.format(
        sample_id=sample_id,
        runs_summary=runs_summary,
        concordance_summary=conc_summary,
        concordance_details=conc_details,
        reproducibility_summary=repro_summary,
        alerts_summary=alerts_summary,
        citations=citations_text,
        run_count=run_count,
        timestamp=timestamp,
        tools_called=tools_called,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    report = response.content

    return {
        "report":  report,
        "status":  "complete",
        "messages": [AIMessage(content=report)],
    }


# ─── Node: graceful_degradation ───────────────────────────────────────────────

def graceful_degradation(state: AgentState) -> dict:
    """
    Called when critical API tools are unreachable.
    Produces a clear, actionable error message without hallucinating data.
    """
    failed  = state.get("failed_tools", [])
    message = DEGRADATION_MESSAGE.format(
        sample_id=state["sample_id"],
        failed_tools="\n".join(f"- {t}" for t in failed),
    )
    return {
        "report":  message,
        "status":  "degraded",
        "messages": [AIMessage(content=message)],
    }


# ─── Graph construction ───────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("fetch_data",           fetch_data)
    graph.add_node("analyse",              analyse)
    graph.add_node("search_literature",    search_literature)
    graph.add_node("synthesise_report",    synthesise_report)
    graph.add_node("graceful_degradation", graceful_degradation)

    graph.add_edge(START, "fetch_data")

    graph.add_conditional_edges(
        "fetch_data",
        route_after_fetch,
        {
            "fetch_data":          "fetch_data",
            "analyse":             "analyse",
            "graceful_degradation": "graceful_degradation",
        },
    )

    graph.add_edge("analyse",           "search_literature")
    graph.add_edge("search_literature", "synthesise_report")
    graph.add_edge("synthesise_report", END)
    graph.add_edge("graceful_degradation", END)

    return graph.compile()


# Compiled graph — imported by API and dashboard
compiled_graph = build_graph()


def run_agent(sample_id: str, task: str = "full_analysis") -> dict:
    """
    Synchronous entry point for the FastAPI endpoint.
    Returns the complete agent result including report, citations, and tool trace.
    """
    initial_state: AgentState = {
        "messages":           [HumanMessage(content=f"Analyse sample {sample_id}. Task: {task}")],
        "sample_id":          sample_id,
        "task":               task,
        "runs_data":          None,
        "concordance_summary": None,
        "concordance_details": None,
        "reproducibility_data": None,
        "alerts_data":        None,
        "pubmed_citations":   None,
        "fetch_retries":      0,
        "pubmed_retries":     0,
        "failed_tools":       [],
        "tools_called":       [],
        "report":             None,
        "status":             "running",
    }

    final_state = compiled_graph.invoke(initial_state)

    return {
        "sample_id":       sample_id,
        "status":          final_state.get("status", "unknown"),
        "report":          final_state.get("report", ""),
        "pubmed_citations": final_state.get("pubmed_citations", []),
        "tools_called":    final_state.get("tools_called", []),
        "alerts_found":    len(final_state.get("alerts_data") or []),
    }
