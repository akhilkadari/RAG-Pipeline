"""Streamlit dashboard for the Helix RAG API.

Run with: streamlit run frontend/app.py
"""
from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

API_BASE = os.environ.get("RAG_API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="Helix RAG Console",
    page_icon=":mag:",
    layout="wide",
)


# ─── helpers ──────────────────────────────────────────────
def call_ask(question: str, mode: str, use_rerank: bool) -> dict[str, Any]:
    resp = requests.post(
        f"{API_BASE}/v1/ask",
        json={
            "question": question,
            "mode": mode,
            "use_rerank": use_rerank,
            "verify": True,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def call_documents() -> dict[str, Any]:
    resp = requests.get(f"{API_BASE}/v1/documents", timeout=30)
    resp.raise_for_status()
    return resp.json()


def call_health() -> dict[str, Any]:
    resp = requests.get(f"{API_BASE}/v1/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


def render_answer(payload: dict[str, Any]) -> None:
    answer = payload.get("answer", "")
    is_idk = payload.get("is_idk", False)

    if is_idk:
        st.warning(answer)
    else:
        st.markdown("### Answer")
        st.markdown(answer)

    fallback = payload.get("fallback")
    if fallback:
        st.info(
            f"Fallback reason: **{fallback.get('reason')}** | "
            f"retrieval strength: {fallback.get('retrieval_strength', 0):.2f} "
            f"(threshold {fallback.get('threshold', 0):.2f})"
        )
        cands = fallback.get("candidate_documents") or []
        if cands:
            st.markdown(
                "Documents worth checking manually: " + ", ".join(f"`{c}`" for c in cands)
            )

    st.divider()
    st.markdown("### Confidence")
    conf = payload.get("confidence", {})
    cols = st.columns(5)
    cols[0].metric("Composite", f"{conf.get('composite', 0):.2f}")
    cols[1].metric("Retrieval", f"{conf.get('retrieval', 0):.2f}")
    cols[2].metric("Citation accuracy", f"{conf.get('citation_accuracy', 0):.2f}")
    cols[3].metric("Citation coverage", f"{conf.get('citation_coverage', 0):.2f}")
    cols[4].metric("Completeness", f"{conf.get('completeness', 0):.2f}")
    if conf.get("completeness_missing"):
        st.caption("Possibly missing parts: " + ", ".join(conf["completeness_missing"]))


def render_chunks(payload: dict[str, Any]) -> None:
    st.markdown("### Retrieved chunks (final, post-rerank)")
    chunks = payload.get("chunks", [])
    if not chunks:
        st.caption("No chunks retrieved.")
        return
    for chunk in chunks:
        meta = chunk.get("metadata", {}) or {}
        title = (
            f"[{chunk.get('index')}] {meta.get('filename', 'unknown')}"
            f" — section: {meta.get('section_heading') or '—'}"
        )
        scores = []
        for k in ("rerank_score", "rrf_score", "dense_score", "sparse_score"):
            v = chunk.get(k)
            if v is not None:
                scores.append(f"{k}={v:.3f}")
        with st.expander(title + ("  |  " + ", ".join(scores) if scores else "")):
            st.text(chunk.get("text", "")[:4000])
            st.caption(f"chunk_id={chunk.get('id')}")


def render_citation_report(payload: dict[str, Any]) -> None:
    st.markdown("### Citation verification")
    report = payload.get("citation_report", {})
    cols = st.columns(3)
    cols[0].metric("Coverage", f"{report.get('coverage', 0):.2f}")
    cols[1].metric("Accuracy", f"{report.get('accuracy', 0):.2f}")
    cols[2].metric(
        "Verified / total",
        f"{report.get('supported_count', 0)} / {report.get('total_checks', 0)}",
    )
    for check in report.get("checks", []):
        icon = ":white_check_mark:" if check.get("supported") else ":warning:"
        st.markdown(
            f"{icon} **Claim:** {check.get('claim', '')}  \n"
            f"  Cites: {check.get('cited_indexes', [])} — _{check.get('reason', '')}_"
        )


def render_retrieval_breakdown(payload: dict[str, Any]) -> None:
    st.markdown("### Retrieval breakdown")
    info = payload.get("retrieval", {})
    cols = st.columns(4)
    cols[0].metric("Dense hits", info.get("dense_count", 0))
    cols[1].metric("Sparse hits", info.get("sparse_count", 0))
    cols[2].metric("Fused", info.get("fused_count", 0))
    cols[3].metric("Final", info.get("final_count", 0))


# ─── sidebar / state ──────────────────────────────────────
with st.sidebar:
    st.title("Helix RAG Console")

    try:
        health = call_health()
        st.caption(
            f"API: {API_BASE}  \n"
            f"Chroma chunks: {health.get('chroma_chunks', 0)}  \n"
            f"BM25 chunks: {health.get('bm25_chunks', 0)}"
        )
    except Exception as exc:
        st.error(f"API unreachable at {API_BASE}\n{exc}")

    st.divider()
    st.subheader("Retrieval mode")
    mode_choice = st.radio(
        "Mode",
        options=["hybrid", "dense", "sparse"],
        index=0,
        horizontal=True,
    )
    use_rerank = st.toggle("Enable LLM reranker", value=True)
    compare = st.toggle("Side-by-side: hybrid vs dense-only", value=False)

    st.divider()
    st.subheader("Indexed documents")
    try:
        docs = call_documents()
        for d in docs.get("documents", []):
            st.markdown(
                f"- `{d['filename']}` ({d['chunk_count']} chunks, "
                f"strategies: {', '.join(d['chunking_strategies']) or '—'})"
            )
    except Exception as exc:
        st.error(f"Could not list documents: {exc}")


# ─── main column ──────────────────────────────────────────
st.title("Ask the documentation")

question = st.text_area(
    "Question", placeholder="e.g. What's the rate limit per token on the Helix API?", height=80
)
ask_button = st.button("Ask", type="primary", use_container_width=False)

if ask_button and question.strip():
    with st.spinner("Retrieving and generating..."):
        try:
            if compare:
                left, right = st.columns(2)
                with left:
                    st.markdown(f"#### Mode: **hybrid** (rerank={use_rerank})")
                    payload_a = call_ask(question, "hybrid", use_rerank)
                    render_answer(payload_a)
                    render_chunks(payload_a)
                    render_citation_report(payload_a)
                    render_retrieval_breakdown(payload_a)
                with right:
                    st.markdown(f"#### Mode: **dense-only** (rerank={use_rerank})")
                    payload_b = call_ask(question, "dense", use_rerank)
                    render_answer(payload_b)
                    render_chunks(payload_b)
                    render_citation_report(payload_b)
                    render_retrieval_breakdown(payload_b)
            else:
                payload = call_ask(question, mode_choice, use_rerank)
                render_answer(payload)
                render_chunks(payload)
                render_citation_report(payload)
                render_retrieval_breakdown(payload)
        except requests.HTTPError as exc:
            st.error(f"API error: {exc.response.status_code}\n{exc.response.text}")
        except Exception as exc:
            st.error(f"Failed: {exc}")
