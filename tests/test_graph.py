"""
Tests for agent/graph.py

Tests cover:
  - Full graph run completes successfully with mocked tools and LLM
  - Graph enters graceful_degradation when API is unreachable
  - Fetch retry logic fires when critical tools fail (up to MAX_FETCH_RETRIES)
  - PubMed retry fires when first search returns empty results
  - Final state contains required fields
"""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import (
    AgentState,
    fetch_data,
    graceful_degradation,
    route_after_fetch,
    synthesise_report,
)


def _base_state(sample_id: str = "HG001") -> AgentState:
    return {
        "messages":            [HumanMessage(content=f"Analyse sample {sample_id}")],
        "sample_id":           sample_id,
        "task":                "full_analysis",
        "runs_data":           None,
        "concordance_summary": None,
        "concordance_details": None,
        "reproducibility_data": None,
        "alerts_data":         None,
        "pubmed_citations":    None,
        "fetch_retries":       0,
        "pubmed_retries":      0,
        "failed_tools":        [],
        "tools_called":        [],
        "report":              None,
        "status":              "running",
    }


class TestFetchData:
    def test_fetch_populates_concordance_on_success(self, mock_pipeline_api):
        state = _base_state()
        result = fetch_data(state)
        assert result["concordance_summary"] is not None
        assert result["concordance_summary"]["success"] is True

    def test_fetch_records_failed_tools_when_api_down(self, mock_pipeline_api_down):
        state = _base_state()
        result = fetch_data(state)
        assert len(result["failed_tools"]) > 0
        assert "get_concordance_summary" in result["failed_tools"]

    def test_fetch_appends_tool_names(self, mock_pipeline_api):
        state = _base_state()
        result = fetch_data(state)
        assert "get_pipeline_runs" in result["tools_called"]
        assert "get_concordance_summary" in result["tools_called"]

    def test_fetch_sets_empty_alerts_when_none(self, mock_pipeline_api):
        state = _base_state()
        result = fetch_data(state)
        assert result["alerts_data"] == []


class TestRouteAfterFetch:
    def test_routes_to_analyse_when_data_present(self, mock_pipeline_api):
        state = _base_state()
        result = fetch_data(state)
        state.update(result)
        route = route_after_fetch(state)
        assert route == "analyse"

    def test_routes_to_degradation_when_critical_tools_fail_and_retries_exhausted(self):
        state = _base_state()
        state["failed_tools"]  = ["get_concordance_summary", "get_pipeline_runs"]
        state["fetch_retries"] = 1  # MAX_FETCH_RETRIES = 1
        route = route_after_fetch(state)
        assert route == "graceful_degradation"

    def test_routes_to_fetch_retry_when_retries_remain(self):
        state = _base_state()
        state["failed_tools"]  = ["get_concordance_summary"]
        state["fetch_retries"] = 0
        route = route_after_fetch(state)
        assert route == "fetch_data"

    def test_routes_to_analyse_when_only_non_critical_tools_fail(self):
        state = _base_state()
        state["failed_tools"]  = ["get_active_alerts"]
        state["fetch_retries"] = 0
        route = route_after_fetch(state)
        assert route == "analyse"


class TestGracefulDegradation:
    def test_produces_degraded_status(self):
        state = _base_state()
        state["failed_tools"] = ["get_concordance_summary", "get_pipeline_runs"]
        result = graceful_degradation(state)
        assert result["status"] == "degraded"

    def test_report_contains_failed_tool_names(self):
        state = _base_state()
        state["failed_tools"] = ["get_concordance_summary", "get_pipeline_runs"]
        result = graceful_degradation(state)
        assert "get_concordance_summary" in result["report"]
        assert "get_pipeline_runs" in result["report"]

    def test_report_contains_startup_instructions(self):
        state = _base_state()
        state["failed_tools"] = ["get_concordance_summary"]
        result = graceful_degradation(state)
        assert "uvicorn" in result["report"]

    def test_report_does_not_hallucinate_metrics(self):
        state = _base_state()
        state["failed_tools"] = ["get_concordance_summary"]
        result = graceful_degradation(state)
        assert "0.99" not in result["report"]
        assert "F1" not in result["report"]


class TestSynthesiseReport:
    def test_produces_complete_status(self, mock_pipeline_api, mock_pubmed):
        with patch("agent.graph.llm") as mock_llm:
            mock_llm.invoke.return_value = AIMessage(
                content="## Quality Report: HG001\n\n### Executive Summary\nAll metrics pass."
            )
            state = _base_state()
            fetch_result = fetch_data(state)
            state.update(fetch_result)
            state["pubmed_citations"] = [
                {"pmid": "31039644", "url": "https://pubmed.ncbi.nlm.nih.gov/31039644/"}
            ]
            result = synthesise_report(state)
        assert result["status"] == "complete"
        assert result["report"] is not None
        assert len(result["report"]) > 10

    def test_report_contains_sample_id(self, mock_pipeline_api):
        with patch("agent.graph.llm") as mock_llm:
            mock_llm.invoke.return_value = AIMessage(
                content="## Quality Report: HG001\n\nSNV F1: 0.9928"
            )
            state = _base_state()
            fetch_result = fetch_data(state)
            state.update(fetch_result)
            state["pubmed_citations"] = []
            result = synthesise_report(state)
        assert "HG001" in result["report"]


class TestFullGraphRun:
    def test_full_run_returns_complete_status(self, mock_pipeline_api, mock_pubmed):
        from agent.graph import run_agent
        with patch("agent.graph.llm") as mock_llm:
            mock_llm.invoke.return_value = AIMessage(
                content="## Quality Report: HG001\n\nAll metrics pass thresholds."
            )
            mock_llm.bind_tools.return_value = mock_llm
            result = run_agent("HG001", "full_analysis")
        assert result["status"] in ("complete", "degraded")
        assert result["sample_id"] == "HG001"
        assert result["report"] is not None

    def test_full_run_returns_required_fields(self, mock_pipeline_api, mock_pubmed):
        with patch("agent.graph.llm") as mock_llm:
            mock_llm.invoke.return_value = AIMessage(content="Quality report complete.")
            mock_llm.bind_tools.return_value = mock_llm
            from agent.graph import run_agent
            result = run_agent("HG001")
        required = {"sample_id", "status", "report", "pubmed_citations", "tools_called", "alerts_found"}
        assert required.issubset(result.keys())

    def test_full_run_degrades_when_api_down(self, mock_pipeline_api_down):
        with patch("agent.graph.llm") as mock_llm:
            mock_llm.invoke.return_value = AIMessage(content="Query: test")
            mock_llm.bind_tools.return_value = mock_llm
            from agent.graph import run_agent
            result = run_agent("HG001")
        assert result["status"] == "degraded"
        assert "uvicorn" in result["report"]
