"""note.create tool implementation."""

from __future__ import annotations

from typing import Any, Dict

from app.memory.db import get_conn


def main(args: Dict[str, Any] | None) -> Dict[str, Any]:
    """Create a note entry in the database.

    Args:
        args: Input mapping with required ``text`` field and optional ``tags``
            field (list or string).

    Returns:
        A dictionary with ``ok`` flag. When successful it contains the
        ``note_id`` of the created note; otherwise an ``error`` message.
    """

    payload = args or {}
    text = str(payload.get("text", "")).strip()
    tags_value = payload.get("tags", [])

    if not text:
        return {"ok": False, "error": "text is required"}

    if isinstance(tags_value, list):
        tags = ",".join(str(tag) for tag in tags_value)
    else:
        tags = str(tags_value)

    with get_conn() as conn:
        cursor = conn.execute(
            "INSERT INTO notes(user_id, text, tags) VALUES (?,?,?)",
            ("self", text, tags),
        )
        note_id = cursor.lastrowid

    return {"ok": True, "note_id": note_id}
