"""Conversation history: last N turns per session, stored in SQLite.

One "turn" is a user message plus the assistant's reply, so the cap of
``max_turns`` turns is stored as at most ``2 * max_turns`` rows. Old rows
are pruned on write, so the table never grows without bound.
"""
import sqlite3
import threading
from pathlib import Path


class HistoryStore:
    def __init__(self, db_path: str = "./history.db", max_turns: int = 10):
        self.db_path = str(db_path)
        self.max_turns = max_turns
        self._lock = threading.Lock()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS turns (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       session_id TEXT NOT NULL,
                       role TEXT NOT NULL,
                       content TEXT NOT NULL,
                       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                   )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_turns_session "
                "ON turns (session_id, id)"
            )

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        """Append one message and prune the session to the turn cap."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO turns (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )
            # Keep the newest 2 * max_turns rows for this session.
            conn.execute(
                """DELETE FROM turns
                   WHERE session_id = ?
                     AND id NOT IN (
                         SELECT id FROM turns
                         WHERE session_id = ?
                         ORDER BY id DESC
                         LIMIT ?
                     )""",
                (session_id, session_id, 2 * self.max_turns),
            )

    def get_history(self, session_id: str, limit: int | None = None) -> list[dict]:
        """Return the last ``limit`` turns (default: the store's cap),
        oldest first, as [{'role': ..., 'content': ...}, ...]."""
        n_turns = self.max_turns if limit is None else limit
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """SELECT role, content FROM turns
                   WHERE session_id = ?
                   ORDER BY id DESC
                   LIMIT ?""",
                (session_id, 2 * n_turns),
            ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def clear(self, session_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
