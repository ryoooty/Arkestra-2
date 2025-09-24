"""Reminder tool for creating scheduled notifications."""

from __future__ import annotations

from datetime import datetime
from typing import Dict

from app.core.reminders import create_reminder


def main(args: Dict) -> Dict:
    """Create a reminder by scheduling it via :mod:`app.core.reminders`."""
    title = (args or {}).get("title", "").strip()
    when = (args or {}).get("when", "").strip()
    channel = (args or {}).get("channel", "cli")
    if not title or not when:
        return {"ok": False, "error": "title and when are required"}
    try:
        dt = datetime.strptime(when, "%Y-%m-%d %H:%M")
    except ValueError:
        return {"ok": False, "error": "when format must be YYYY-MM-DD HH:MM"}
    iso = dt.strftime("%Y-%m-%d %H:%M:00")
    rem_id = create_reminder("self", title, iso, channel)
    return {"ok": True, "reminder_id": rem_id, "when": iso}
