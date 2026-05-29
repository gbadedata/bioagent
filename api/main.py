"""BioAgent FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import agent

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("bioagent_api_startup")
    yield
    logger.info("bioagent_api_shutdown")


app = FastAPI(
    title="BioAgent API",
    description="Autonomous bioinformatics AI analyst — LangGraph + Claude",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent.router, prefix="/api/v1/agent", tags=["agent"])


@app.get("/health")
def health():
    return {"status": "ok", "service": "bioagent", "version": "1.0.0"}
