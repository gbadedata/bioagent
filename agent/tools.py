"""
BioAgent tools - six deterministic functions that call external APIs.

Each tool returns a structured dict with a 'success' flag.
On failure, 'error' contains a human-readable explanation.
The agent uses these flags to decide whether to retry or degrade gracefully.

Tools are NOT responsible for reasoning about their results.
That is the LLM's job.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from Bio import Entrez
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

API_BASE = os.environ.get("PIPELINE_API_BASE", "http://localhost:8000")
TIMEOUT  = 10.0

# NCBI requires an email for Entrez access
Entrez.email = "bioagent@biomarker-concordance-pipeline.dev"


def _get(path: str, params: dict | None = None) -> dict[str, Any]:
    """Make a GET request to the pipeline API. Returns structured result."""
    url = f"{API_BASE}{path}"
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
            return {"success": True, "data": r.json(), "status_code": r.status_code}
    except httpx.ConnectError:
        return {
            "success": False,
            "error": f"Cannot connect to pipeline API at {API_BASE}. Is it running?",
            "data": None,
        }
    except httpx.TimeoutException:
        return {
            "success": False,
            "error": f"Request to {url} timed out after {TIMEOUT}s.",
            "data": None,
        }
    except httpx.HTTPStatusError as e:
        return {
            "success": False,
            "error": f"API returned {e.response.status_code} for {url}.",
            "data": None,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error calling {url}: {type(e).__name__}: {e}",
            "data": None,
        }


@tool
def get_pipeline_runs(sample_id: str) -> dict[str, Any]:
    """
    Fetch recent pipeline runs for a sample from the concordance API.

    Returns run IDs, statuses, replicates, and timestamps.
    Use this first to understand how many runs exist for the sample.
    """
    result = _get("/api/v1/runs", params={"sample_id": sample_id, "limit": 20})
    if not result["success"]:
        return result

    runs = result["data"]
    if not runs:
        return {
            "success": True,
            "data": [],
            "summary": f"No pipeline runs found for sample '{sample_id}'.",
        }

    summary_lines = [
        f"Run {r['run_id']}: replicate {r['replicate']}, status={r['status']}, created={r['created_at'][:10]}"
        for r in runs
    ]
    return {
        "success": True,
        "data": runs,
        "count": len(runs),
        "summary": "\n".join(summary_lines),
    }


@tool
def get_concordance_summary(sample_id: str) -> dict[str, Any]:
    """
    Fetch aggregated concordance metrics for a sample across all runs.

    Returns SNV and Indel precision, recall, F1 means and minimums.
    Use this to get an overall picture before drilling into per-run detail.
    """
    result = _get(f"/api/v1/concordance/summary/{sample_id}")
    if not result["success"]:
        return result

    d = result["data"]
    summary = (
        f"SNV:   F1={d['snv_f1_mean']:.4f} (min {d['snv_f1_min']:.4f}), "
        f"Prec={d['snv_precision_mean']:.4f}, Recall={d['snv_recall_mean']:.4f}\n"
        f"Indel: F1={d['indel_f1_mean']:.4f}, "
        f"Prec={d['indel_precision_mean']:.4f}, Recall={d['indel_recall_mean']:.4f}\n"
        f"Runs passing: {d['runs_passing']}/{d['n_runs']}"
    )
    return {"success": True, "data": d, "summary": summary}


@tool
def get_concordance_results(sample_id: str) -> dict[str, Any]:
    """
    Fetch per-run concordance results (SNV and Indel metrics for each run).

    Use this after get_concordance_summary when you need run-level detail,
    for example to identify which specific run failed a threshold.
    """
    result = _get("/api/v1/concordance", params={"limit": 50})
    if not result["success"]:
        return result

    all_results = result["data"]
    if not all_results:
        return {"success": True, "data": [], "summary": "No concordance results found."}

    lines = []
    for r in all_results:
        pass_str = "PASS" if (r["f1_pass"] and r["precision_pass"] and r["recall_pass"]) else "FAIL"
        lines.append(
            f"{r['variant_type']} F1={r['f1_score']:.4f} "
            f"Prec={r['precision']:.4f} Rec={r['recall']:.4f} [{pass_str}]"
        )

    return {
        "success": True,
        "data": all_results,
        "count": len(all_results),
        "summary": "\n".join(lines),
    }


@tool
def get_reproducibility(sample_id: str) -> dict[str, Any]:
    """
    Fetch the most recent reproducibility analysis for a sample.

    Returns ICC(A,1), confidence interval, median CV, and overall pass/fail.
    ICC >= 0.90 indicates excellent VAF reproducibility across replicate runs.
    """
    result = _get(f"/api/v1/reproducibility/{sample_id}/latest")
    if not result["success"]:
        return result

    d = result["data"]
    summary = (
        f"ICC(A,1)={d['icc']:.4f} (95% CI [{d['icc_ci_lower']:.4f}, {d['icc_ci_upper']:.4f}])\n"
        f"Median CV={d['median_cv']:.1f}%\n"
        f"Variants analysed={d['n_variants']}\n"
        f"ICC pass (>=0.90): {'YES' if d['icc_pass'] else 'NO'}\n"
        f"Overall pass: {'YES' if d['overall_pass'] else 'NO'}"
    )
    return {"success": True, "data": d, "summary": summary}


@tool
def get_active_alerts(sample_id: str) -> dict[str, Any]:
    """
    Fetch unresolved quality alerts from the monitoring system.

    Returns Westgard rule violations and concordance threshold breaches.
    An empty list means all monitored metrics are within tolerance.
    """
    result = _get("/api/v1/alerts", params={"unresolved_only": "true", "limit": 50})
    if not result["success"]:
        return result

    alerts = result["data"]
    if not alerts:
        return {
            "success": True,
            "data": [],
            "summary": "No active alerts. All monitored metrics within tolerance.",
        }

    lines = [
        f"[{a['severity'].upper()}] {a['alert_type']}: {a['message']}"
        for a in alerts
    ]
    return {
        "success": True,
        "data": alerts,
        "count": len(alerts),
        "summary": "\n".join(lines),
    }


@tool
def search_pubmed(query: str, max_results: int = 5) -> dict[str, Any]:
    """
    Search PubMed for papers relevant to specific quality findings.

    The query should be constructed from actual metric values and findings,
    not generic terms. Returns PMIDs, titles, and abstracts.
    """
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        search_results = Entrez.read(handle)
        handle.close()

        pmids = search_results.get("IdList", [])
        if not pmids:
            return {
                "success": True,
                "data": [],
                "query": query,
                "summary": f"No PubMed results for query: '{query}'",
            }

        handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="abstract", retmode="text")
        abstracts_raw = handle.read()
        handle.close()

        papers = []
        for pmid in pmids:
            papers.append({
                "pmid": pmid,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })

        summary_lines = [f"PMID {p['pmid']}: {p['url']}" for p in papers]

        return {
            "success": True,
            "data": papers,
            "query": query,
            "count": len(papers),
            "abstracts_preview": abstracts_raw[:2000],
            "summary": f"Found {len(papers)} papers for '{query}':\n" + "\n".join(summary_lines),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"PubMed search failed: {type(e).__name__}: {e}",
            "data": [],
            "query": query,
        }


# All tools as a list for LangGraph binding
ALL_TOOLS = [
    get_pipeline_runs,
    get_concordance_summary,
    get_concordance_results,
    get_reproducibility,
    get_active_alerts,
    search_pubmed,
]
