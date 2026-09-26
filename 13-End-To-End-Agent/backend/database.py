import sqlite3
from pathlib import Path
from threading import RLock

from langgraph.checkpoint.sqlite import SqliteSaver


# ------------------------------------------------------------
# Database path
# ------------------------------------------------------------

# backend/database.py
#        ↓
# parent = backend
#        ↓
# parent = 13-End-To-End-Agent

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_PATH = BASE_DIR / "chatbot.db"


# ------------------------------------------------------------
# SQLite connection
# ------------------------------------------------------------

conn = sqlite3.connect(
    database=str(DATABASE_PATH),
    check_same_thread=False,
    timeout=30,
)


# ------------------------------------------------------------
# SQLite configuration
# ------------------------------------------------------------

conn.execute(
    "PRAGMA busy_timeout = 30000"
)


# ------------------------------------------------------------
# Database lock
# ------------------------------------------------------------

database_lock = RLock()


# ------------------------------------------------------------
# LangGraph checkpoint storage
# ------------------------------------------------------------

checkpoint = SqliteSaver(conn)