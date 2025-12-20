from __future__ import annotations

import sys
import inspect
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st

# -----------------------------
# Fix imports when running from /app
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../ragnroll
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rag import rag_answer  # type: ignore

Doc = Dict[str, Any]


def _normalize_doc(d: Doc) -> Doc:
    """Make doc keys predictable. Supports docs where metadata is nested under d['meta']."""
    meta = d.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}

    title = meta.get("title") or d.get("title") or d.get("Title")
    source = meta.get("source") or d.get("source") or d.get("Source")
    url = meta.get("url") or d.get("url") or d.get("URL")
    published = meta.get("published") or d.get("published")  # ✅ NEW

    snippet = d.get("snippet") or d.get("Snippet") or d.get("text") or d.get("content") or d.get("chunk")
    score = d.get("score") or d.get("Score") or d.get("similarity") or d.get("distance")

    if not title and isinstance(url, str) and url.strip():
        title = url.strip().split("/")[-1].replace("-", " ").strip() or "Untitled"
    if not source:
        source = "Unknown source"

    nd = dict(d)
    nd["title"] = title or "Untitled"
    nd["source"] = source
    nd["url"] = url
    nd["published"] = published  # ✅ NEW
    nd["snippet"] = snippet
    nd["score"] = score
    return nd

def clean_snippet(snippet: str) -> str:
    if not snippet:
        return ""

    # Remove inline "Most Popular - A - B - C - ..." even if it appears mid-paragraph
    snippet = re.sub(
        r"Most Popular\s*-\s*(?:[^-\n]+\s*-\s*){3,}[^-\n]+",
        "",
        snippet,
        flags=re.IGNORECASE,
    )

    # Remove generic long dash-chains like: "- A - B - C - D"
    snippet = re.sub(
        r"(?:\s*-\s*[^-\n]{3,80}){4,}",
        "",
        snippet,
    )

    # Clean extra spaces
    snippet = re.sub(r"\s+", " ", snippet).strip()
    return snippet

def _call_rag_answer(question: str, top_k: int) -> Tuple[str, List[Doc]]:
    candidates = ["top_k", "k", "topk", "top_n", "topK", "K"]

    try:
        sig = inspect.signature(rag_answer)
        params = sig.parameters

        for name in candidates:
            if name in params:
                ans, docs = rag_answer(question, **{name: top_k})
                return ans, docs

        if len(params) >= 2:
            ans, docs = rag_answer(question, top_k)
            return ans, docs

        ans, docs = rag_answer(question)
        return ans, docs

    except TypeError:
        try:
            ans, docs = rag_answer(question, top_k)
            return ans, docs
        except Exception:
            ans, docs = rag_answer(question)
            return ans, docs


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _relevance_at_k(question: str, docs: List[Doc], k: int) -> float:
    """
    Simple relevance@k: average keyword overlap between question and each retrieved snippet.
    Returns 0..1 (roughly).
    """
    if k <= 0 or not docs:
        return 0.0

    q = _tokenize(question)
    if not q:
        return 0.0

    total = 0.0
    used = 0
    for d in docs[:k]:
        snippet = d.get("snippet") or ""
        s = _tokenize(snippet)
        if not s:
            continue
        overlap = len(q & s) / float(len(q))
        total += overlap
        used += 1

    return total / used if used else 0.0


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="RAG’n’Roll", page_icon="🎸", layout="wide")

st.title("🎸 RAG’n’Roll")
st.caption("Local RAG: FAISS + Sentence-Transformers + Ollama")

st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top-k sources", min_value=1, max_value=10, value=5, step=1)

st.sidebar.write("")
rebuild = st.sidebar.button("🔄 Refresh News (Rebuild RAG)")

if rebuild:
    import subprocess

    with st.sidebar:
        with st.spinner("Refreshing daily snapshot (ingest → embed)..."):
            p1 = subprocess.run(
                [sys.executable, "-m", "core.ingest"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            p2 = None
            if p1.returncode == 0:
                p2 = subprocess.run(
                    [sys.executable, "-m", "core.embed"],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )

        if p1.returncode != 0:
            st.error("❌ Ingest failed")
        elif p2 and p2.returncode != 0:
            st.error("❌ Embed failed")
        else:
            st.success("✅ Refresh complete! New FAISS index built.")

        with st.expander("Show logs"):
            st.code("INGEST OUTPUT:\n" + (p1.stdout or "") + "\n" + (p1.stderr or ""))
            if p2 is not None:
                st.code("EMBED OUTPUT:\n" + (p2.stdout or "") + "\n" + (p2.stderr or ""))

# Keep state
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

st.subheader("Ask a question")
question = st.text_input(
    " ",
    value="What is happening in technology news today?",
    label_visibility="collapsed",
)

ask = st.button("Ask 🚀")

if ask:
    if not question.strip():
        st.warning("Type a question first.")
    else:
        with st.spinner("Running RAG..."):
            try:
                answer, docs = _call_rag_answer(question.strip(), top_k)
                docs = docs or []
                norm_docs = [_normalize_doc(d) for d in docs]

                st.session_state.last_answer = answer or ""
                st.session_state.last_docs = norm_docs
                st.session_state.last_question = question.strip()

            except Exception as e:
                st.error("RAG execution failed (see error below):")
                st.exception(e)

if st.session_state.last_question:
    st.metric(
        label=f"Relevance@{top_k}",
        value=f"{_relevance_at_k(st.session_state.last_question, st.session_state.last_docs, top_k):.2f}",
    )

    st.header("Question")
    st.write(st.session_state.last_question)

    st.header("Answer")
    st.write(st.session_state.last_answer or "No answer returned.")

    st.header("Sources")
    docs = st.session_state.last_docs or []
    if not docs:
        st.info("No sources returned.")
    else:
        for i, d in enumerate(docs[:top_k], start=1):
            title = d.get("title", "Untitled")
            source = d.get("source", "Unknown source")
            score = d.get("score", None)
            url = d.get("url", None)
            snippet = d.get("snippet", None)
            published = d.get("published")  # ✅ NEW

            score_txt = ""
            if isinstance(score, (int, float)):
                score_txt = f" (score: {float(score):.3f})"

            if isinstance(url, str) and url.strip():
                st.markdown(f"**[{i}] [{title}]({url}) — _{source}_**{score_txt}")
            else:
                st.markdown(f"**[{i}] {title} — _{source}_**{score_txt}")

            # ✅ Show published
            if isinstance(published, str) and published.strip():
                st.caption(f"Published: {published.strip()}")

            if isinstance(snippet, str) and snippet.strip():
                st.caption(clean_snippet(snippet))

            st.divider()
