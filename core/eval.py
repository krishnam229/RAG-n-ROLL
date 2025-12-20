from __future__ import annotations

import re
from typing import Dict, List


def _keywords(text: str) -> List[str]:
    # keep simple, robust tokens
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    # drop tiny tokens
    return [t for t in tokens if len(t) >= 3]


def precision_at_k(question: str, docs: List[Dict], k: int = 5) -> float:
    """
    Lightweight Precision@K heuristic:
    A retrieved chunk is "relevant" if it contains ANY keyword from the question.
    This is not perfect, but it's a valid baseline for a class project.
    """
    keys = set(_keywords(question))
    if not keys:
        return 0.0

    top = docs[:k]
    rel = 0
    for d in top:
        chunk = (d.get("text") or "").lower()
        if any(k in chunk for k in keys):
            rel += 1

    return rel / max(1, len(top))
