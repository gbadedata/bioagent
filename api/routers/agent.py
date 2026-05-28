"""
BioAgent API router.

Uses BackgroundTasks pattern: POST /analyse returns a job_id immediately.
GET /results/{job_id} polls for the completed report.

This avoids HTTP timeouts on long agent runs (15-30s with Claude Sonnet).
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

router   = APIRouter()
executor = ThreadPoolExecutor(max_workers=4)

# In-memory job store — sufficient for portfolio/local use
# Production would use Redis or a database
_jobs: dict[str, dict] = {}


class AnalyseRequest(BaseModel):
    sample_id: str
    task:      Literal["full_analysis", "concordance_only", "reproducibility_only"] = "full_analysis"


class JobResponse(BaseModel):
    job_id:    str
    status:    Literal["queued", "running", "complete", "degraded", "error"]
    message:   str


class ResultResponse(BaseModel):
    job_id:          str
    sample_id:       str
    status:          str
    report:          str | None
    pubmed_citations: list
    tools_called:    list[str]
    alerts_found:    int
    duration_seconds: float | None


def _run_agent_job(job_id: str, sample_id: str, task: str) -> None:
    """Execute the agent in a thread pool. Updates the job store when complete."""
    import os
    from dotenv import load_dotenv
    load_dotenv()

    _jobs[job_id]["status"] = "running"
    start = time.time()

    try:
        from agent.graph import run_agent
        result = run_agent(sample_id=sample_id, task=task)
        _jobs[job_id].update({
            "status":           result.get("status", "complete"),
            "report":           result.get("report"),
            "pubmed_citations": result.get("pubmed_citations", []),
            "tools_called":     result.get("tools_called", []),
            "alerts_found":     result.get("alerts_found", 0),
            "duration_seconds": round(time.time() - start, 2),
        })
    except Exception as e:
        _jobs[job_id].update({
            "status": "error",
            "report": f"Agent failed with unexpected error: {type(e).__name__}: {e}",
            "duration_seconds": round(time.time() - start, 2),
        })


@router.post("/analyse", response_model=JobResponse, status_code=202)
def analyse(payload: AnalyseRequest, background_tasks: BackgroundTasks):
    """
    Trigger an autonomous analysis run.

    Returns a job_id immediately (HTTP 202 Accepted).
    Poll GET /results/{job_id} for the completed report.

    Note: No authentication is implemented. This is intentional for local
    portfolio use. Production deployment requires an X-API-Key header.
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id":           job_id,
        "sample_id":        payload.sample_id,
        "status":           "queued",
        "report":           None,
        "pubmed_citations": [],
        "tools_called":     [],
        "alerts_found":     0,
        "duration_seconds": None,
    }

    executor.submit(_run_agent_job, job_id, payload.sample_id, payload.task)

    return JobResponse(
        job_id=job_id,
        status="queued",
        message=f"Analysis queued for sample '{payload.sample_id}'. Poll GET /api/v1/agent/results/{job_id}",
    )


@router.get("/results/{job_id}", response_model=ResultResponse)
def get_results(job_id: str):
    """Poll for the result of a previously submitted analysis job."""
    if job_id not in _jobs:
        raise HTTPException(404, f"Job '{job_id}' not found.")

    job = _jobs[job_id]
    return ResultResponse(
        job_id=job_id,
        sample_id=job["sample_id"],
        status=job["status"],
        report=job.get("report"),
        pubmed_citations=job.get("pubmed_citations", []),
        tools_called=job.get("tools_called", []),
        alerts_found=job.get("alerts_found", 0),
        duration_seconds=job.get("duration_seconds"),
    )


@router.get("/jobs")
def list_jobs():
    """List all jobs and their statuses. Useful for monitoring."""
    return [
        {"job_id": jid, "sample_id": j["sample_id"], "status": j["status"]}
        for jid, j in _jobs.items()
    ]
