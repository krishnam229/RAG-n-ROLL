import json
import sqlite3
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from core.db import DB_PATH
from core.chunk import chunk_text

CHUNKS_PATH = Path("data/chunks.jsonl")
FAISS_PATH = Path("data/faiss.index")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_articles(conn: sqlite3.Connection, limit: int = 200) -> List[Tuple]:
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id, title, url, source, published, text FROM articles ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return rows


def build_index(article_limit: int = 200, chunk_size: int = 1200, overlap: int = 200):
    Path("data").mkdir(parents=True, exist_ok=True)

    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    conn = sqlite3.connect(DB_PATH)
    articles = load_articles(conn, article_limit)
    conn.close()

    texts = []
    metas = []

    for aid, title, url, source, published, text in articles:
        chunks = chunk_text(text, chunk_size, overlap)
        for i, ch in enumerate(chunks):
            texts.append(ch)
            metas.append(
                {
                    "article_id": aid,
                    "chunk_idx": i,
                    "title": title,
                    "url": url,
                    "source": source,
                    "published": published,   # ✅ NEW
                }
            )

    if not texts:
        print("No chunks found to embed.")
        return

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for i, (meta, text) in enumerate(zip(metas, texts)):
            meta["chunk_id"] = i
            f.write(json.dumps({"meta": meta, "text": text}, ensure_ascii=False) + "\n")

    faiss.write_index(index, str(FAISS_PATH))

    print(f"Built FAISS index | Articles: {len(articles)} | Chunks: {len(texts)}")
    print("Saved files:")
    print(f"- {FAISS_PATH}")
    print(f"- {CHUNKS_PATH}")


if __name__ == "__main__":
    build_index(article_limit=200)
