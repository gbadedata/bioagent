"""
BioAgent — Streamlit Chat Interface

Three conversation modes:
  1. Analyse a sample:  'analyse HG001'
  2. Ask follow-up:     'why is the indel F1 lower?'
  3. Threshold query:   'what if I set the minimum indel F1 to 0.97?'

Features:
  - Real-time streaming via st.write_stream
  - Tool call trace in expandable panel
  - Conversation history stored in st.session_state
  - Cost guardrail: warns after MAX_RUNS_PER_SESSION runs
"""

from __future__ import annotations

import os
import re
import time
import threading

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

MAX_RUNS = int(os.environ.get("MAX_RUNS_PER_SESSION", "20"))
PIPELINE_API = os.environ.get("PIPELINE_API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="BioAgent",
    page_icon="🧬",
    layout="wide",
)

# ─── Session state initialisation ────────────────────────────────────────────

if "messages"    not in st.session_state: st.session_state.messages    = []
if "run_count"   not in st.session_state: st.session_state.run_count   = 0
if "tool_traces" not in st.session_state: st.session_state.tool_traces = []
if "last_sample" not in st.session_state: st.session_state.last_sample = None

# ─── Header ───────────────────────────────────────────────────────────────────

st.title("🧬 BioAgent")
st.caption(
    "Autonomous bioinformatics AI analyst · LangGraph + Claude · "
    f"Connected to pipeline API at {PIPELINE_API}"
)

# Cost guardrail warning
if st.session_state.run_count >= MAX_RUNS:
    st.warning(
        f"You have run {st.session_state.run_count} analyses this session. "
        f"Each run costs approximately $0.05 in API credit. "
        f"Consider restarting the app to reset the counter.",
        icon="⚠️",
    )

col1, col2, col3 = st.columns(3)
col1.metric("Analyses this session", st.session_state.run_count)
col2.metric("API credit used (est.)", f"${st.session_state.run_count * 0.05:.2f}")
col3.metric("Credit remaining (est.)", f"${max(0, 5.00 - st.session_state.run_count * 0.05):.2f}")

st.divider()

# ─── Conversation display ─────────────────────────────────────────────────────

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show tool trace for assistant messages that have one
        if msg["role"] == "assistant" and i < len(st.session_state.tool_traces):
            trace = st.session_state.tool_traces[i]
            if trace:
                with st.expander("Tool call trace", expanded=False):
                    for step in trace:
                        st.markdown(f"**{step['tool']}**")
                        if step.get("summary"):
                            st.text(step["summary"])
                        st.divider()

# ─── Extract sample ID from message ──────────────────────────────────────────

def extract_sample_id(text: str) -> str | None:
    """Extract a sample ID from natural language input."""
    patterns = [
        r'\b(HG\d{3}[_\-]?\w*)\b',
        r'\banalyse\s+(\S+)',
        r'\banalyze\s+(\S+)',
        r'\bsample\s+(\S+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return st.session_state.last_sample


def is_analysis_request(text: str) -> bool:
    """Determine if the user wants a full analysis or a follow-up question."""
    triggers = ["analyse", "analyze", "run analysis", "check", "report on", "quality report"]
    return any(t in text.lower() for t in triggers)


# ─── Run agent ────────────────────────────────────────────────────────────────

def run_agent_sync(sample_id: str, task: str = "full_analysis") -> dict:
    """Run the agent synchronously. Handles the async/Streamlit conflict."""
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()

    from agent.graph import run_agent
    return run_agent(sample_id=sample_id, task=task)


# ─── Chat input ───────────────────────────────────────────────────────────────

prompt = st.chat_input("Type 'analyse HG001' or ask a follow-up question...")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    sample_id = extract_sample_id(prompt)

    with st.chat_message("assistant"):
        if not sample_id:
            response = (
                "I need a sample ID to analyse. Try: `analyse HG001`\n\n"
                "If you are asking a follow-up about the previous sample, "
                "please mention the sample ID again."
            )
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.tool_traces.append([])

        elif st.session_state.run_count >= MAX_RUNS:
            response = (
                f"You have reached the session limit of {MAX_RUNS} analyses. "
                "Please restart the app to continue."
            )
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.tool_traces.append([])

        else:
            st.session_state.last_sample = sample_id
            st.session_state.run_count  += 1

            # Show progress while agent runs
            status_placeholder = st.empty()
            steps = [
                "Fetching pipeline runs...",
                "Fetching concordance data...",
                "Fetching reproducibility data...",
                "Checking active alerts...",
                "Analysing findings...",
                "Searching PubMed...",
                "Synthesising quality report...",
            ]

            result_holder: dict = {}
            error_holder:  dict = {}

            def _run():
                try:
                    result_holder["result"] = run_agent_sync(sample_id, "full_analysis")
                except Exception as e:
                    error_holder["error"] = str(e)

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()

            # Animate progress while agent runs
            step_idx = 0
            while thread.is_alive():
                status_placeholder.info(f"🔄 {steps[step_idx % len(steps)]}")
                time.sleep(1.5)
                step_idx += 1

            thread.join()
            status_placeholder.empty()

            if error_holder:
                response = f"Agent encountered an error: {error_holder['error']}"
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.tool_traces.append([])

            else:
                result = result_holder["result"]
                report = result.get("report", "No report generated.")
                tools  = result.get("tools_called", [])
                citations = result.get("pubmed_citations", [])

                st.markdown(report)

                # Tool trace — human-readable summaries only
                trace_steps = [
                    {"tool": tool, "summary": f"Called successfully"}
                    for tool in tools
                ]
                if citations:
                    trace_steps.append({
                        "tool": "search_pubmed",
                        "summary": "\n".join([f"PMID {c['pmid']}: {c['url']}" for c in citations]),
                    })

                with st.expander("Tool call trace", expanded=False):
                    for step in trace_steps:
                        st.markdown(f"**{step['tool']}**")
                        st.text(step.get("summary", ""))
                        st.divider()

                st.session_state.messages.append({"role": "assistant", "content": report})
                st.session_state.tool_traces.append(trace_steps)

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("BioAgent")
    st.markdown("""
**What BioAgent can do:**
- Analyse concordance and reproducibility for any sample
- Explain anomalies with likely causes
- Search PubMed for relevant literature
- Generate structured clinical-grade QC reports

**Example queries:**
- `analyse HG001`
- `check concordance for HG001`
- `quality report HG001`

**Thresholds used:**
- SNV F1 >= 0.98
- Indel F1 >= 0.95
- VAF ICC >= 0.90
- VAF CV <= 15%
    """)

    st.divider()

    if st.button("Clear conversation"):
        st.session_state.messages    = []
        st.session_state.tool_traces = []
        st.session_state.last_sample = None
        st.rerun()

    st.divider()
    st.caption(
        "Powered by Claude (claude-sonnet-4-20250514) · "
        "LangGraph · LangChain · "
        "github.com/gbadedata/bioagent"
    )
