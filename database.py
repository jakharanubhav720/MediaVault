import os
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("MEDIAVAULT_DB_PATH", BASE_DIR / "mediavault.db"))

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS media (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL UNIQUE,
                media_type TEXT NOT NULL CHECK(media_type IN ('Image', 'Video')),
                mime_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                embedding TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_media_type ON media(media_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_media_filename ON media(filename)")

        # Additive migrations preserve existing SQLite rows and databases.
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(media)").fetchall()}
        if "tags" not in columns:
            conn.execute("ALTER TABLE media ADD COLUMN tags TEXT NOT NULL DEFAULT '[]'")
        if "sha256" not in columns:
            conn.execute("ALTER TABLE media ADD COLUMN sha256 TEXT")
        if "phash" not in columns:
            conn.execute("ALTER TABLE media ADD COLUMN phash TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_media_sha256 ON media(sha256)")
