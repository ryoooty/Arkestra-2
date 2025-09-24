"""
SQLite DB helper + migrations.
Provides: get_conn(), migrate(), simple CRUD helpers used by core/* and tools/*.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List

DB_PATH = Path("arkestra.db")

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS messages(
  id INTEGER PRIMARY KEY,
  user_id TEXT,
  role TEXT CHECK(role IN ('user','assistant')),
  text TEXT,
  ts DATETIME DEFAULT CURRENT_TIMESTAMP,
  approved INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS facts(
  id INTEGER PRIMARY KEY,
  user_id TEXT, key TEXT, value TEXT,
  ts DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS notes(
  id INTEGER PRIMARY KEY,
  user_id TEXT, text TEXT, tags TEXT,
  ts DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS aliases(
  id INTEGER PRIMARY KEY,
  user_id TEXT, alias TEXT,
  is_primary INTEGER DEFAULT 0,
  short_desc TEXT,
  ts DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(user_id, alias)
);
CREATE TABLE IF NOT EXISTS feedback(
  id INTEGER PRIMARY KEY,
  msg_id INTEGER,
  kind TEXT CHECK(kind IN ('up','down','text')),
  text TEXT,
  ts DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS neuro_logs(
  id INTEGER PRIMARY KEY,
  session_id TEXT,
  levels_json TEXT,
  preset_json TEXT,
  ts DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS bandit_stats(
  intent TEXT, kind TEXT,
  wins REAL DEFAULT 1, plays REAL DEFAULT 2,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(intent, kind)
);
CREATE TABLE IF NOT EXISTS tools(
  name TEXT PRIMARY KEY,
  title TEXT, description TEXT,
  instruction TEXT, entrypoint TEXT,
  enabled INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS env_sessions(
  id INTEGER PRIMARY KEY,
  channel TEXT, chat_id TEXT,
  participants_json TEXT,
  started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(channel, chat_id)
);
CREATE TABLE IF NOT EXISTS env_facts(
  id INTEGER PRIMARY KEY,
  env_id INTEGER,
  key TEXT, value TEXT,
  importance REAL DEFAULT 0.5,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(env_id) REFERENCES env_sessions(id)
);
CREATE TABLE IF NOT EXISTS sleep_batches(
  id TEXT PRIMARY KEY,
  started_at DATETIME,
  finished_at DATETIME,
  from_seen_at DATETIME,
  to_seen_at DATETIME,
  processed_count INTEGER,
  status TEXT
);
CREATE TABLE IF NOT EXISTS reminders(
  id INTEGER PRIMARY KEY,
  user_id TEXT,
  title TEXT,
  when_ts DATETIME,
  channel TEXT DEFAULT 'cli',
  status TEXT DEFAULT 'scheduled',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def migrate() -> None:
    with get_conn() as c:
        c.executescript(SCHEMA_SQL)


def get_tool_instructions(names: list[str]) -> dict:
    if not names:
        return {}
    qmarks = ",".join("?" for _ in names)
    with get_conn() as c:
        cur = c.execute(f"SELECT name,instruction FROM tools WHERE enabled=1 AND name IN ({qmarks})", names)
        return {r["name"]: r["instruction"] for r in cur.fetchall()}


def upsert_bandit(intent: str, kind: str, wins_delta: float, plays_delta: float) -> None:
    with get_conn() as c:
        cur = c.execute("SELECT wins,plays FROM bandit_stats WHERE intent=? AND kind=?", (intent, kind))
        row = cur.fetchone()
        if row:
            wins = row["wins"] + wins_delta
            plays = row["plays"] + plays_delta
            c.execute("UPDATE bandit_stats SET wins=?, plays=?, updated_at=CURRENT_TIMESTAMP WHERE intent=? AND kind=?",
                      (wins, plays, intent, kind))
        else:
            c.execute("INSERT INTO bandit_stats(intent,kind,wins,plays) VALUES (?,?,?,?)",
                      (intent, kind, max(1.0+wins_delta, 0.0), max(2.0+plays_delta, 0.0)))


def decay_bandit(factor: float = 0.995) -> None:
    with get_conn() as c:
        c.execute("UPDATE bandit_stats SET wins=wins*?, plays=plays*?, updated_at=CURRENT_TIMESTAMP", (factor, factor))


def insert_message(user_id: str, role: str, text: str, approved: int = 0) -> int:
    with get_conn() as c:
        cur = c.execute("INSERT INTO messages(user_id, role, text, approved) VALUES (?,?,?,?)",
                        (user_id, role, text, approved))
        return cur.lastrowid


def add_feedback(msg_id: int, kind: str, text: str | None = None) -> int:
    assert kind in ("up", "down", "text")
    with get_conn() as c:
        cur = c.execute(
            "INSERT INTO feedback(msg_id, kind, text) VALUES (?,?,?)",
            (msg_id, kind, text),
        )
        return cur.lastrowid


def mark_approved(msg_id: int, approved: int = 1) -> None:
    with get_conn() as c:
        c.execute("UPDATE messages SET approved=? WHERE id=?", (approved, msg_id))


def last_assistant_msg_id(user_id: str) -> int | None:
    with get_conn() as c:
        cur = c.execute(
            "SELECT id FROM messages WHERE user_id=? AND role='assistant' ORDER BY id DESC LIMIT 1",
            (user_id,),
        )
        row = cur.fetchone()
        return int(row["id"]) if row else None


def get_last_messages(user_id: str, n: int = 6) -> list[dict]:
    with get_conn() as c:
        cur = c.execute(
            "SELECT role,text,ts FROM messages WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, n),
        )
        rows = [dict(r) for r in cur.fetchall()]
        return list(reversed(rows))


def upsert_env_session(channel: str, chat_id: str, participants_json: str) -> int:
    with get_conn() as c:
        c.execute(
            "INSERT INTO env_sessions(channel,chat_id,participants_json) VALUES (?,?,?) "
            "ON CONFLICT(channel,chat_id) DO UPDATE SET last_seen=CURRENT_TIMESTAMP",
            (channel, chat_id, participants_json)
        )
        cur = c.execute("SELECT id FROM env_sessions WHERE channel=? AND chat_id=?", (channel, chat_id))
        return int(cur.fetchone()["id"])


def get_env_facts(env_id: int) -> List[Dict[str, Any]]:
    with get_conn() as c:
        cur = c.execute("SELECT key,value,importance FROM env_facts WHERE env_id=? ORDER BY importance DESC, updated_at DESC", (env_id,))
        return [dict(r) for r in cur.fetchall()]


def set_env_fact(env_id: int, key: str, value: str, importance: float = 0.5) -> None:
    with get_conn() as c:
        cur = c.execute("SELECT id FROM env_facts WHERE env_id=? AND key=?", (env_id, key))
        row = cur.fetchone()
        if row:
            c.execute("UPDATE env_facts SET value=?, importance=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                      (value, importance, row["id"]))
        else:
            c.execute("INSERT INTO env_facts(env_id,key,value,importance) VALUES (?,?,?,?)",
                      (env_id, key, value, importance))
