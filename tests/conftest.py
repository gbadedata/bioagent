"""
Shared test fixtures for BioAgent.

All external calls are mocked:
- Pipeline API calls via respx (httpx mock)
- LLM calls via monkeypatching ChatAnthropic
- PubMed calls via monkeypatching Bio.Entrez
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

# ─── Sample API response fixtures ─────────────────────────────────────────────

SAMPLE_RUNS = [
    {
        "id": "aa000000-0000-0000-0000-000000000001",
        "run_id": "run_HG001_rep1",
        "sample_id": "HG001",
        "replicate": 1,
        "status": "completed",
        "created_at": "2026-05-28T13:26:41.109699Z",
        "completed_at": "2026-05-28T13:30:00.000000Z",
        "nextflow_run": "lonely_pike_rep1",
    },
    {
        "id": "aa000000-0000-0000-0000-000000000002",
        "run_id": "run_HG001_rep2",
        "sample_id": "HG001",
        "replicate": 2,
        "status": "completed",
        "created_at": "2026-05-28T13:26:41.508785Z",
        "completed_at": "2026-05-28T13:30:00.000000Z",
        "nextflow_run": "lonely_pike_rep2",
    },
    {
        "id": "aa000000-0000-0000-0000-000000000003",
        "run_id": "run_HG001_rep3",
        "sample_id": "HG001",
        "replicate": 3,
        "status": "completed",
        "created_at": "2026-05-28T13:26:41.752280Z",
        "completed_at": "2026-05-28T13:30:00.000000Z",
        "nextflow_run": "lonely_pike_rep3",
    },
]

SAMPLE_CONCORDANCE_SUMMARY = {
    "sample_id": "HG001",
    "n_runs": 3,
    "snv_f1_mean": 0.9928,
    "snv_f1_min": 0.9925,
    "snv_precision_mean": 0.9921,
    "snv_recall_mean": 0.9934,
    "indel_f1_mean": 0.9656,
    "indel_precision_mean": 0.9612,
    "indel_recall_mean": 0.9703,
    "runs_passing": 3,
    "runs_failing": 0,
}

SAMPLE_CONCORDANCE_RESULTS = [
    {
        "id": "bb000000-0000-0000-0000-000000000001",
        "run_id": "aa000000-0000-0000-0000-000000000001",
        "variant_type": "SNP",
        "true_positives": 412431.0,
        "false_positives": 3245.0,
        "false_negatives": 2701.0,
        "precision": 0.9921,
        "recall": 0.9934,
        "f1_score": 0.9928,
        "specificity": 0.9999,
        "cohen_kappa": 0.9921,
        "precision_pass": True,
        "recall_pass": True,
        "f1_pass": True,
        "created_at": "2026-05-28T13:26:41.109699Z",
    },
    {
        "id": "bb000000-0000-0000-0000-000000000002",
        "run_id": "aa000000-0000-0000-0000-000000000001",
        "variant_type": "INDEL",
        "true_positives": 45231.0,
        "false_positives": 1820.0,
        "false_negatives": 1540.0,
        "precision": 0.9612,
        "recall": 0.9703,
        "f1_score": 0.9657,
        "specificity": 0.9998,
        "cohen_kappa": 0.9598,
        "precision_pass": True,
        "recall_pass": True,
        "f1_pass": True,
        "created_at": "2026-05-28T13:26:41.109699Z",
    },
]

SAMPLE_REPRODUCIBILITY = {
    "id": "cc000000-0000-0000-0000-000000000001",
    "sample_id": "HG001",
    "run_ids": '["run_HG001_rep1","run_HG001_rep2","run_HG001_rep3"]',
    "n_variants": 200,
    "icc": 0.9847,
    "icc_ci_lower": 0.9801,
    "icc_ci_upper": 0.9889,
    "icc_p_value": 0.0001,
    "icc_pass": True,
    "median_cv": 4.2,
    "p90_cv": 8.1,
    "overall_pass": True,
    "alerts": "[]",
    "created_at": "2026-05-28T13:26:41.109699Z",
}

SAMPLE_ALERTS: list = []

SAMPLE_PUBMED_RESULTS = [
    {"pmid": "31039644", "url": "https://pubmed.ncbi.nlm.nih.gov/31039644/"},
    {"pmid": "30542152", "url": "https://pubmed.ncbi.nlm.nih.gov/30542152/"},
]


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_pipeline_api():
    """Mock all pipeline API endpoints using respx."""
    with respx.mock(base_url="http://localhost:8000", assert_all_called=False) as mock:
        mock.get("/api/v1/runs").mock(
            return_value=httpx.Response(200, json=SAMPLE_RUNS)
        )
        mock.get("/api/v1/concordance/summary/HG001").mock(
            return_value=httpx.Response(200, json=SAMPLE_CONCORDANCE_SUMMARY)
        )
        mock.get("/api/v1/concordance").mock(
            return_value=httpx.Response(200, json=SAMPLE_CONCORDANCE_RESULTS)
        )
        mock.get("/api/v1/reproducibility/HG001/latest").mock(
            return_value=httpx.Response(200, json=SAMPLE_REPRODUCIBILITY)
        )
        mock.get("/api/v1/alerts").mock(
            return_value=httpx.Response(200, json=SAMPLE_ALERTS)
        )
        yield mock


@pytest.fixture
def mock_pipeline_api_down():
    """Mock all pipeline API endpoints as unreachable."""
    with respx.mock(base_url="http://localhost:8000", assert_all_called=False) as mock:
        mock.get("/api/v1/runs").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        mock.get("/api/v1/concordance/summary/HG001").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        mock.get("/api/v1/concordance").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        mock.get("/api/v1/reproducibility/HG001/latest").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        mock.get("/api/v1/alerts").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        yield mock


@pytest.fixture
def mock_pubmed():
    """Mock Bio.Entrez calls to avoid real network requests in tests."""
    with patch("Bio.Entrez.esearch") as mock_search, \
         patch("Bio.Entrez.efetch") as mock_fetch, \
         patch("Bio.Entrez.read") as mock_read:

        mock_read.return_value = {"IdList": ["31039644", "30542152"]}
        mock_fetch.return_value = MagicMock(read=MagicMock(
            return_value="Abstract text for PMID 31039644...\nAbstract for 30542152..."
        ))
        yield {"search": mock_search, "fetch": mock_fetch, "read": mock_read}


@pytest.fixture
def mock_llm():
    """Mock ChatAnthropic to avoid real API calls in graph tests."""
    with patch("agent.graph.llm") as mock:
        from langchain_core.messages import AIMessage
        mock.invoke.return_value = AIMessage(
            content='PubMed query: germline variant calling GIAB benchmark validation'
        )
        mock.bind_tools.return_value = mock
        yield mock
