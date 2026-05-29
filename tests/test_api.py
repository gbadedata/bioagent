"""
Tests for api/routers/agent.py

Tests cover:
  - POST /api/v1/agent/analyse returns 202 with job_id
  - GET /api/v1/agent/results/{job_id} returns correct structure
  - GET /api/v1/agent/results/{unknown_id} returns 404
  - GET /api/v1/agent/jobs lists all jobs
  - GET /health returns ok
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["service"] == "bioagent"


class TestAnalyseEndpoint:
    def test_returns_202_with_job_id(self):
        r = client.post("/api/v1/agent/analyse", json={"sample_id": "HG001"})
        assert r.status_code == 202
        body = r.json()
        assert "job_id" in body
        assert body["status"] == "queued"

    def test_job_id_is_uuid_format(self):
        import uuid
        r = client.post("/api/v1/agent/analyse", json={"sample_id": "HG001"})
        job_id = r.json()["job_id"]
        uuid.UUID(job_id)  # raises if not valid UUID

    def test_message_contains_poll_instructions(self):
        r = client.post("/api/v1/agent/analyse", json={"sample_id": "HG001"})
        assert "Poll" in r.json()["message"] or "poll" in r.json()["message"]

    def test_accepts_task_parameter(self):
        r = client.post(
            "/api/v1/agent/analyse",
            json={"sample_id": "HG001", "task": "concordance_only"},
        )
        assert r.status_code == 202


class TestResultsEndpoint:
    def test_returns_404_for_unknown_job(self):
        r = client.get("/api/v1/agent/results/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 404

    def test_queued_job_has_correct_structure(self):
        r = client.post("/api/v1/agent/analyse", json={"sample_id": "HG001"})
        job_id = r.json()["job_id"]
        result = client.get(f"/api/v1/agent/results/{job_id}")
        assert result.status_code == 200
        body = result.json()
        assert body["job_id"] == job_id
        assert body["sample_id"] == "HG001"
        assert body["status"] in ("queued", "running", "complete", "degraded", "error")
        assert "tools_called" in body
        assert "pubmed_citations" in body
        assert "alerts_found" in body


class TestJobsEndpoint:
    def test_lists_submitted_jobs(self):
        client.post("/api/v1/agent/analyse", json={"sample_id": "HG001"})
        r = client.get("/api/v1/agent/jobs")
        assert r.status_code == 200
        jobs = r.json()
        assert isinstance(jobs, list)
        assert len(jobs) >= 1
        assert "job_id" in jobs[0]
        assert "status" in jobs[0]
