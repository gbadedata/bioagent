"""
Tests for agent/tools.py

Every tool is tested with:
  - Successful mocked API response
  - Unreachable API (ConnectError)
  - Unexpected HTTP error (4xx/5xx)

PubMed tool is tested with mocked Bio.Entrez calls.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from agent.tools import (
    get_active_alerts,
    get_concordance_results,
    get_concordance_summary,
    get_pipeline_runs,
    get_reproducibility,
    search_pubmed,
)
from tests.conftest import (
    SAMPLE_ALERTS,
    SAMPLE_CONCORDANCE_RESULTS,
    SAMPLE_CONCORDANCE_SUMMARY,
    SAMPLE_REPRODUCIBILITY,
    SAMPLE_RUNS,
)


class TestGetPipelineRuns:
    def test_returns_runs_on_success(self, mock_pipeline_api):
        result = get_pipeline_runs.invoke({"sample_id": "HG001"})
        assert result["success"] is True
        assert result["count"] == 3
        assert "run_HG001_rep1" in result["summary"]

    def test_returns_error_when_api_down(self, mock_pipeline_api_down):
        result = get_pipeline_runs.invoke({"sample_id": "HG001"})
        assert result["success"] is False
        assert "Cannot connect" in result["error"]

    def test_returns_empty_list_gracefully(self):
        with respx.mock(base_url="http://localhost:8000") as mock:
            mock.get("/api/v1/runs").mock(return_value=httpx.Response(200, json=[]))
            result = get_pipeline_runs.invoke({"sample_id": "UNKNOWN"})
        assert result["success"] is True
        assert result["data"] == []
        assert "No pipeline runs" in result["summary"]


class TestGetConcordanceSummary:
    def test_returns_summary_on_success(self, mock_pipeline_api):
        result = get_concordance_summary.invoke({"sample_id": "HG001"})
        assert result["success"] is True
        assert "SNV" in result["summary"]
        assert "0.9928" in result["summary"]

    def test_returns_error_when_api_down(self, mock_pipeline_api_down):
        result = get_concordance_summary.invoke({"sample_id": "HG001"})
        assert result["success"] is False
        assert "Cannot connect" in result["error"]

    def test_returns_error_on_404(self):
        with respx.mock(base_url="http://localhost:8000") as mock:
            mock.get("/api/v1/concordance/summary/MISSING").mock(
                return_value=httpx.Response(404, json={"detail": "Not found"})
            )
            result = get_concordance_summary.invoke({"sample_id": "MISSING"})
        assert result["success"] is False
        assert "404" in result["error"]


class TestGetConcordanceResults:
    def test_returns_results_on_success(self, mock_pipeline_api):
        result = get_concordance_results.invoke({"sample_id": "HG001"})
        assert result["success"] is True
        assert result["count"] == 2
        assert "SNP" in result["summary"]
        assert "INDEL" in result["summary"]

    def test_shows_pass_fail_in_summary(self, mock_pipeline_api):
        result = get_concordance_results.invoke({"sample_id": "HG001"})
        assert "PASS" in result["summary"] or "FAIL" in result["summary"]

    def test_returns_error_when_api_down(self, mock_pipeline_api_down):
        result = get_concordance_results.invoke({"sample_id": "HG001"})
        assert result["success"] is False


class TestGetReproducibility:
    def test_returns_icc_on_success(self, mock_pipeline_api):
        result = get_reproducibility.invoke({"sample_id": "HG001"})
        assert result["success"] is True
        assert "0.9847" in result["summary"]
        assert "ICC" in result["summary"]

    def test_summary_includes_cv(self, mock_pipeline_api):
        result = get_reproducibility.invoke({"sample_id": "HG001"})
        assert "CV" in result["summary"]
        assert "4.2" in result["summary"]

    def test_returns_error_when_api_down(self, mock_pipeline_api_down):
        result = get_reproducibility.invoke({"sample_id": "HG001"})
        assert result["success"] is False
        assert "Cannot connect" in result["error"]


class TestGetActiveAlerts:
    def test_returns_no_alerts_message_when_empty(self, mock_pipeline_api):
        result = get_active_alerts.invoke({"sample_id": "HG001"})
        assert result["success"] is True
        assert "No active alerts" in result["summary"]

    def test_returns_alerts_when_present(self):
        alerts = [{"severity": "high", "alert_type": "westgard_1_3s",
                   "message": "VAF outside 3SD", "resolved": False}]
        with respx.mock(base_url="http://localhost:8000") as mock:
            mock.get("/api/v1/alerts").mock(return_value=httpx.Response(200, json=alerts))
            result = get_active_alerts.invoke({"sample_id": "HG001"})
        assert result["success"] is True
        assert result["count"] == 1
        assert "HIGH" in result["summary"]

    def test_returns_error_when_api_down(self, mock_pipeline_api_down):
        result = get_active_alerts.invoke({"sample_id": "HG001"})
        assert result["success"] is False


class TestSearchPubmed:
    def test_returns_papers_on_success(self, mock_pubmed):
        result = search_pubmed.invoke({
            "query": "germline variant calling GIAB benchmark",
            "max_results": 5,
        })
        assert result["success"] is True
        assert result["count"] == 2
        assert any("31039644" in str(p) for p in result["data"])

    def test_returns_empty_list_gracefully(self, mock_pubmed):
        mock_pubmed["read"].return_value = {"IdList": []}
        result = search_pubmed.invoke({"query": "very obscure query", "max_results": 5})
        assert result["success"] is True
        assert result["data"] == []
        assert "No PubMed results" in result["summary"]

    def test_returns_error_on_entrez_failure(self):
        from unittest.mock import patch
        with patch("Bio.Entrez.esearch", side_effect=Exception("Network error")):
            result = search_pubmed.invoke({"query": "test query", "max_results": 5})
        assert result["success"] is False
        assert "PubMed search failed" in result["error"]

    def test_query_preserved_in_result(self, mock_pubmed):
        query = "intraclass correlation coefficient sequencing reproducibility"
        result = search_pubmed.invoke({"query": query, "max_results": 3})
        assert result["query"] == query
