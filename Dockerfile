# Stage 1: build
FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml .

RUN pip install --upgrade pip setuptools wheel && \
    pip install \
        langchain langchain-anthropic langchain-core langgraph anthropic \
        biopython httpx fastapi uvicorn pydantic pydantic-settings \
        structlog tenacity python-dotenv nest-asyncio

# Stage 2: runtime
FROM python:3.12-slim

# Non-root user — security best practice
RUN useradd --create-home --shell /bin/bash bioagent

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY agent/    ./agent/
COPY api/      ./api/

RUN chown -R bioagent:bioagent /app
USER bioagent

EXPOSE 8001

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8001/health').raise_for_status()"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
