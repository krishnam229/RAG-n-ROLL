import sqlite3
from pathlib import Path

DB_PATH = Path("data/ragnroll.db")

def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT UNIQUE,
        title TEXT,
        source TEXT,
        published TEXT,
        text TEXT,
        added_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    return conn
