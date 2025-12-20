# core/rag.py
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from core.retrieve import retrieve

# -----------------------------
# CONFIG
# -----------------------------
OLLAMA_MODEL = "llama3.2"

# Your actual Ollama install path (confirmed by dir %LOCALAPPDATA%\Programs\Ollama)
OLLAMA_EXE = os.environ.get(
    "OLLAMA_EXE",
    r"C:\Users\krish\AppData\Local\Programs\Ollama\ollama.exe",
)

SYSTEM_PROMPT = """You are RAG’n’Roll, a research assistant.
Answer the question ONLY using the provided sources.

Rules:
- If the sources do not contain enough information, say so.
- If the question asks for "today", "latest", "current", or similar, and the sources do not clearly mention dates or recent updates, you MUST say you cannot verify what's happening today from these sources.
- Always cite sources using [1], [2], etc. matching the provided sources.
- Do not invent URLs or facts not present in sources.
- Keep the answer concise and structured.
"""

# -----------------------------
# HELPERS
# -----------------------------
def looks_like_fresh_query(q: str) -> bool:
    q = q.lower()
    return any(
        w in q
        for w in [
            "today",
            "latest",
            "current",
            "right now",
            "this week",
            "breaking",
            "recent",
            "news today",
        ]
    )

# -----------------------------
# RAG PIPELINE
# -----------------------------
def build_prompt(question: str, docs: List[Dict]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        meta = d.get("meta", {}) if isinstance(d.get("meta"), dict) else {}
        blocks.append(
            f"[{i}] Title: {meta.get('title','')}\n"
            f"Source: {meta.get('source','')}\n"
            f"URL: {meta.get('url','')}\n"
            f"Published: {meta.get('published') or meta.get('date') or meta.get('publish_date') or ''}\n"
            f"Excerpt:\n{d.get('text','')}"
        )

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question:\n{question}\n\n"
        f"Sources:\n" + "\n\n".join(blocks) + "\n\n"
        f"Answer (with citations like [1], [2]):"
    )

def ask_ollama(prompt: str) -> str:
    exe = Path(OLLAMA_EXE)
    if not exe.exists():
        raise FileNotFoundError(
            f"Ollama executable not found at:\n  {OLLAMA_EXE}\n"
            "Fix: set env var OLLAMA_EXE to the correct path."
        )

    # Force UTF-8 for the subprocess on Windows to avoid cp1252 decode issues.
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.run(
        [str(exe), "run", OLLAMA_MODEL],
        input=prompt,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env=env,
    )

    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        return f"ERROR calling Ollama:\n{err}"

    return (proc.stdout or "").strip()

def rag_answer(question: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
    docs = retrieve(question, top_k=top_k)

    # Freshness guard: if user asks for "today/latest/current" but sources have no date signals,
    # force the model to be explicit that it cannot verify "today" from these sources.
    if looks_like_fresh_query(question):
        has_any_date = False
        for d in docs:
            meta = d.get("meta", {}) if isinstance(d.get("meta"), dict) else {}
            if meta.get("date") or meta.get("published") or meta.get("publish_date"):
                has_any_date = True
                break

        if not has_any_date:
            prompt = build_prompt(question, docs) + "\n\n" + (
                "Important: The provided sources do NOT include publication dates or explicit 'today' updates. "
                "Do NOT claim this reflects today's news. Clearly say you cannot verify what's happening today from these sources, "
                "then summarize what the sources DO contain."
            )
            answer = ask_ollama(prompt)
            return answer, docs

    prompt = build_prompt(question, docs)
    answer = ask_ollama(prompt)
    return answer, docs

# -----------------------------
# CLI
# -----------------------------
def format_sources(docs: List[Dict]) -> str:
    lines = []
    for i, d in enumerate(docs, 1):
        meta = d.get("meta", {}) if isinstance(d.get("meta"), dict) else {}
        lines.append(f"[{i}] {meta.get('title','Untitled')} — {meta.get('url','')}")
    return "\n".join(lines)

if __name__ == "__main__":
    q = input("Ask RAG’n’Roll: ").strip()
    answer, docs = rag_answer(q, top_k=5)

    print("\nAnswer:\n" + "-" * 60)
    print(answer)

    print("\nSources:\n" + "-" * 60)
    print(format_sources(docs))
