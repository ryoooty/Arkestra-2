"""messages.search_by_date tool implementation."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict

from app.memory.db import get_conn


def main(args: Dict[str, Any] | None) -> Dict[str, Any]:
    """Fetch messages stored within the provided date range."""

    payload = args or {}
    date_str = payload.get("date")
    span = int(payload.get("span_days", 1) or 1)

    if not date_str:
        return {"ok": False, "error": "date (YYYY-MM-DD) required"}

    try:
        start_dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return {"ok": False, "error": "invalid date format"}

    start = start_dt.strftime("%Y-%m-%d 00:00:00")
    end_dt = start_dt + timedelta(days=max(span, 1))
    end = end_dt.strftime("%Y-%m-%d 00:00:00")

    with get_conn() as conn:
        cursor = conn.execute(
            "SELECT role,text,ts FROM messages WHERE ts>=? AND ts<? ORDER BY id",
            (start, end),
        )
        rows = [dict(row) for row in cursor.fetchall()]

    return {"ok": True, "items": rows, "count": len(rows)}
