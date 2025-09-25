"""Reminder tool for creating scheduled notifications."""

from __future__ import annotations

import datetime as dt
from typing import Dict

import pytz
from dateutil import parser as dp

from app.core.reminders import create_reminder


def _to_utc(dt_str: str) -> dt.datetime:
    parsed = dp.parse(dt_str)
    if parsed.tzinfo is None:
        tz = pytz.timezone("Europe/Amsterdam")
        parsed = tz.localize(parsed)
    return parsed.astimezone(pytz.UTC)


def main(args: Dict) -> Dict:
    """Create a reminder by scheduling it via :mod:`app.core.reminders`."""
    title = (args or {}).get("title", "").strip()
    when = (args or {}).get("when", "").strip()
    channel = (args or {}).get("channel", "cli")
    if not title or not when:
        return {"ok": False, "error": "title and when are required"}
    try:
        when_utc = _to_utc(when).replace(microsecond=0)
    except (ValueError, TypeError, OverflowError):
        return {"ok": False, "error": "bad_datetime"}
    iso = when_utc.isoformat()
    rem_id = create_reminder("self", title, iso, channel)
    return {"ok": True, "reminder_id": rem_id, "when": iso}
