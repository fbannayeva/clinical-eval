"""
Clinical-Eval — Streamlit UI

Two-panel interface:
  Left  — ingest new data + system stats
  Right — query interface + structured report output
"""
from __future__ import annotations

import os
import time

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Clinical-Eval",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        border-left: 4px solid #0066cc;
    }
    .trial-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
    }
    .confidence-high  { color: #28a745; font-weight: 600; }
    .confidence-med   { color: #ffc107; font-weight: 600; }
    .confidence-low   { color: #dc3545; font-weight: 600; }
    .tag {
        display: inline-block;
        background: #e8f0fe;
        color: #1a56db;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 12px;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar — Ingest + Stats ─────────────────────────────────────────────────
with st.sidebar:
    st.title("🧬 Clinical-Eval")
    st.caption("AI-powered clinical trial analysis")
    st.divider()

    st.subheader("Data Ingestion")
    ingest_query = st.text_input(
        "Search query",
        placeholder="e.g. KRAS G12C NSCLC Phase 3",
    )
    max_per_source = st.slider("Max results per source", 5, 30, 10)

    if st.button("Ingest data", type="primary", use_container_width=True):
        if ingest_query:
            with st.spinner("Fetching from PubMed + ClinicalTrials.gov..."):
                try:
                    resp = httpx.post(
                        f"{API_URL}/ingest",
                        params={"query": ingest_query, "max_per_source": max_per_source},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    st.success(
                        f"✓ Ingested {result['ingested']} new documents "
                        f"({result['skipped']} skipped)"
                    )
                except httpx.HTTPError as e:
                    st.error(f"Ingestion failed: {e}")
        else:
            st.warning("Enter a search query first")

    st.divider()
    st.subheader("System stats")

    if st.button("Refresh stats", use_container_width=True):
        try:
            resp = httpx.get(f"{API_URL}/health", timeout=5)
            st.success("API online")
        except Exception:
            st.error("API offline")

    st.caption("Source: PubMed · ClinicalTrials.gov")
    st.caption("Models: Claude + BioBERT")
    st.caption("Vector DB: Qdrant · Metadata: PostgreSQL")


# ─── Main — Query Interface ───────────────────────────────────────────────────
st.header("Clinical Trial Intelligence")
st.caption("Ask questions about clinical trials. The system retrieves, structures, and synthesizes evidence.")

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "Query",
        placeholder="e.g. What are the primary endpoints of KRAS G12C inhibitor trials in NSCLC?",
        label_visibility="collapsed",
    )
with col2:
    max_results = st.selectbox("Results", [3, 5, 10], index=1, label_visibility="collapsed")

example_queries = [
    "KRAS G12C inhibitor Phase 3 trials",
    "CDK4/6 inhibitors breast cancer endpoints",
    "PD-1 checkpoint inhibitor toxicity profiles",
]
st.caption("Examples: " + " · ".join(f"`{q}`" for q in example_queries))

if st.button("Analyse", type="primary") and query:
    with st.spinner("Running multi-agent pipeline..."):
        start = time.time()
        try:
            resp = httpx.post(
                f"{API_URL}/query",
                json={"query": query, "max_results": max_results},
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.time() - start

        except httpx.HTTPError as e:
            st.error(f"Query failed: {e}")
            st.stop()

    report = data["report"]

    # ── Summary ──
    st.subheader("Executive summary")
    st.write(report["executive_summary"])

    if report.get("key_findings"):
        st.subheader("Key findings")
        for finding in report["key_findings"]:
            st.markdown(f"- {finding}")

    st.divider()

    # ── Trials ──
    trials = report.get("trials_analyzed", [])
    st.subheader(f"Trials analysed ({len(trials)})")

    for trial in trials:
        conf = trial.get("confidence_score", 0)
        conf_class = (
            "confidence-high" if conf >= 0.7
            else "confidence-med" if conf >= 0.4
            else "confidence-low"
        )

        with st.expander(
            f"{'🟢' if conf >= 0.7 else '🟡' if conf >= 0.4 else '🔴'} "
            f"{trial.get('nct_number', 'No NCT')} — {trial.get('title', '')[:80]}"
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("Phase",  trial.get("phase", "N/A"))
            c2.metric("Status", trial.get("status", "Unknown"))
            c3.metric("Confidence", f"{conf:.0%}")

            if trial.get("sponsor"):
                st.caption(f"Sponsor: {trial['sponsor']}")

            if trial.get("conditions"):
                st.markdown("**Conditions**")
                st.markdown(" ".join(
                    f'<span class="tag">{c}</span>'
                    for c in trial["conditions"]
                ), unsafe_allow_html=True)

            if trial.get("interventions"):
                st.markdown("**Interventions**")
                st.markdown(" ".join(
                    f'<span class="tag">{i}</span>'
                    for i in trial["interventions"]
                ), unsafe_allow_html=True)

            endpoints = trial.get("endpoints", [])
            if endpoints:
                st.markdown("**Endpoints**")
                for ep in endpoints:
                    badge = "🎯" if ep.get("type") == "primary" else "○"
                    st.markdown(
                        f"{badge} **{ep['name']}** ({ep.get('type', '')}) "
                        f"— {ep.get('timeframe', '')}"
                    )

            pop = trial.get("population", {})
            if pop.get("sample_size"):
                st.metric("Sample size", pop["sample_size"])

            citations = trial.get("source_citations", [])
            if citations:
                st.markdown("**Sources**")
                for cit in citations:
                    if cit.startswith("NCT"):
                        st.markdown(f"[{cit}](https://clinicaltrials.gov/study/{cit})")
                    else:
                        st.markdown(f"[{cit}](https://pubmed.ncbi.nlm.nih.gov/{cit}/)")

    # ── Limitations ──
    st.divider()
    with st.expander("⚠️ Limitations & disclaimers"):
        for lim in report.get("limitations", []):
            st.markdown(f"- {lim}")
        if report.get("langfuse_trace_id"):
            st.caption(f"Trace ID: {report['langfuse_trace_id']}")

    st.caption(
        f"Generated by {report.get('generated_by_model', 'unknown')} · "
        f"{data.get('processing_time_ms', 0)}ms · "
        f"Request: {data.get('request_id', '')}"
    )
