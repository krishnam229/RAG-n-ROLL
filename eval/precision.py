import json
import numpy as np

# Simple keyword-based relevance heuristic
def is_relevant(question: str, chunk_text: str) -> bool:
    keywords = question.lower().split()
    text = chunk_text.lower()
    return any(k in text for k in keywords)

def precision_at_k(question, retrieved_chunks, k=5):
    top_k = retrieved_chunks[:k]
    relevant = sum(is_relevant(question, c["text"]) for c in top_k)
    return relevant / k

if __name__ == "__main__":
    question = "technology news today"

    with open("data/chunks.jsonl", "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    score = precision_at_k(question, chunks, k=5)
    print(f"Precision@5: {score:.2f}")
