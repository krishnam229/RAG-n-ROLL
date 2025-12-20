import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]  # project root
FAISS_PATH = BASE_DIR / "data" / "faiss.index"
CHUNKS_PATH = BASE_DIR / "data" / "chunks.jsonl"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_chunks() -> List[Dict]:
    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def retrieve(query: str, top_k: int = 5) -> List[Dict]:
    if not FAISS_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError("Missing FAISS index or chunks.jsonl. Run: python -m core.embed")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    index = faiss.read_index(str(FAISS_PATH))
    chunks = load_chunks()

    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        rec = chunks[int(idx)]
        results.append({
            "score": float(score),
            "meta": rec["meta"],
            "text": rec["text"]
        })
    return results

if __name__ == "__main__":
    q = input("Ask RAG’n’Roll a question: ").strip()
    hits = retrieve(q, top_k=5)

    print("\nTop evidence:\n" + "="*60)
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        print(f"\n[{i}] Score: {h['score']:.3f}")
        print(f"Source: {m.get('source')} | Title: {m.get('title')}")
        print(f"URL: {m.get('url')}")
        print(f"Snippet: {h['text'][:300]}...")
