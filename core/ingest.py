import re
import feedparser
import requests
from trafilatura import extract
from core.db import get_conn

FEEDS = [
    "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "https://www.theverge.com/rss/index.xml",
]

def fetch_article_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "RAGnRoll/1.0"})
        r.raise_for_status()
        text = extract(r.text)
        return text or ""
    except Exception:
        return ""

def clean_text(text: str) -> str:
    """Remove common site boilerplate / junk lines to improve embeddings + snippets."""
    if not text:
        return ""
    # Remove inline "Most Popular - ..." blocks even if they appear mid-paragraph
    text = re.sub(r"Most Popular\s*-\s*(?:[^-\n]+\s*-\s*){3,}[^-\n]+", "", text, flags=re.IGNORECASE)

    bad_markers = [
        "Most Popular",
        "Sign up",
        "Subscribe",
        "Advertisement",
        "Cookie",
        "Privacy Policy",
        "Terms of Service",
        "All rights reserved",
        "©",
    ]

    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        low = line.lower()
        if any(m.lower() in low for m in bad_markers):
            continue

        # Remove "headline list" lines like: "Most Popular - A - B - C - D"
        if " - " in line and line.count(" - ") >= 4:
            continue
        
        # Remove very long "nav-like" lines (often a list of headlines)
        if len(line) > 200 and line.count(" - ") >= 3:
            continue

        # Collapse extra whitespace
        line = re.sub(r"\s+", " ", line).strip()
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def ingest(limit_per_feed: int = 20) -> int:
    conn = get_conn()
    cur = conn.cursor()
    inserted = 0

    for feed_url in FEEDS:
        feed = feedparser.parse(feed_url)
        source = feed.feed.get("title", "Unknown Source")

        for entry in feed.entries[:limit_per_feed]:
            url = entry.get("link")
            title = entry.get("title", "")
            published = entry.get("published", "") or entry.get("updated", "")

            if not url:
                continue

            text = clean_text(fetch_article_text(url))  # ✅ CLEANING ADDED

            # skip if extraction failed / too short
            if len(text) < 400:
                continue

            cur.execute(
                "INSERT OR IGNORE INTO articles(url, title, source, published, text) VALUES (?, ?, ?, ?, ?)",
                (url, title, source, published, text),
            )
            if cur.rowcount > 0:
                inserted += 1

    conn.commit()
    conn.close()
    return inserted

if __name__ == "__main__":
    n = ingest(limit_per_feed=20)
    print(f"RAGnRoll ingestion complete. Inserted {n} new articles.")
