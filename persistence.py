"""
persistence.py
---------------
SQLite-Speicher für Channel-Historien statt eines flüchtigen In-Memory-Dicts.

Wichtige Einschränkung: Render Free-Tier hat KEIN persistentes Volume.
Diese Datei liegt im Container-Filesystem und wird bei jedem Deploy/Neustart gelöscht.
Vorteile trotzdem:
  - Der Bot überlebt temporäre Prozess-Crashes ohne Session-Verlust.
  - Bereit für Postgres/Supabase-Erweiterung über get_connection().
"""

import sqlite3
import json
import time
import threading
from contextlib import contextmanager
from google.genai import types

DB_PATH = "/tmp/hermes_state.db"
MAX_HISTORY_PER_CHANNEL = 12

_local = threading.local()


@contextmanager
def get_connection():
    """Thread-lokale Connection, da discord.py und FastAPI unterschiedliche Threads nutzen."""
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield _local.conn
    finally:
        pass


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                role TEXT NOT NULL,
                parts_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_channel ON messages(channel_id, created_at)")
        conn.commit()


def _serialize_parts(parts) -> str:
    """Serialisiert Gemini-Parts in JSON."""
    serializable = []
    for part in parts:
        if isinstance(part, str):
            serializable.append({"type": "text", "value": part})
        elif hasattr(part, "text") and part.text is not None:
            serializable.append({"type": "text", "value": part.text})
        elif hasattr(part, "function_call") and part.function_call:
            fc = part.function_call
            serializable.append({
                "type": "function_call",
                "name": fc.name,
                "args": dict(fc.args) if fc.args else {},
            })
        elif hasattr(part, "function_response") and part.function_response:
            fr = part.function_response
            serializable.append({
                "type": "function_response",
                "name": fr.name,
                "response": dict(fr.response) if fr.response else {},
            })
        elif hasattr(part, "inline_data"):
            serializable.append({"type": "image_placeholder"})
        else:
            serializable.append({"type": "text", "value": str(part)})
    return json.dumps(serializable)


def _deserialize_parts(parts_json: str) -> list:
    """Deserialisiert JSON strikt in valide types.Part-Objekte für das Google GenAI SDK."""
    raw = json.loads(parts_json)
    parts = []
    for item in raw:
        if item["type"] == "text":
            parts.append(types.Part.from_text(text=item["value"]))
        elif item["type"] == "function_call":
            parts.append(types.Part(function_call=types.FunctionCall(name=item["name"], args=item["args"])))
        elif item["type"] == "function_response":
            parts.append(types.Part(function_response=types.FunctionResponse(name=item["name"], response=item["response"])))
        elif item["type"] == "image_placeholder":
            parts.append(types.Part.from_text(text="[Bild aus vorheriger Nachricht]"))
    return parts


def append_message(channel_id: int, role: str, parts) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (channel_id, role, parts_json, created_at) VALUES (?, ?, ?, ?)",
            (str(channel_id), role, _serialize_parts(parts), time.time()),
        )
        conn.commit()
        _trim_history(conn, channel_id)


def _trim_history(conn, channel_id: int) -> None:
    conn.execute("""
        DELETE FROM messages
        WHERE channel_id = ? AND id NOT IN (
            SELECT id FROM messages WHERE channel_id = ?
            ORDER BY created_at DESC LIMIT ?
        )
    """, (str(channel_id), str(channel_id), MAX_HISTORY_PER_CHANNEL))
    conn.commit()


def get_history(channel_id: int) -> list:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT role, parts_json FROM messages WHERE channel_id = ? ORDER BY created_at ASC",
            (str(channel_id),),
        ).fetchall()
    return [{"role": role, "parts": _deserialize_parts(parts_json)} for role, parts_json in rows]


def clear_history(channel_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM messages WHERE channel_id = ?", (str(channel_id),))
        conn.commit()
