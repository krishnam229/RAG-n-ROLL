import re
from typing import List

def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Simple character-based chunking.
    chunk_size and overlap are in characters.
    """
    text = normalize_whitespace(text)
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += max(chunk_size - overlap, 1)

    return chunks
