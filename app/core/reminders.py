"""Reminder scheduling and persistence via APScheduler."""
from __future__ import annotations

from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler

from app.core.logs import log
from app.memory.db import get_conn, insert_message

_sched: BackgroundScheduler | None = None


def init_scheduler() -> BackgroundScheduler:
    """Initialize the background scheduler once and restore persisted jobs."""
    global _sched
    if _sched:
        return _sched
    scheduler = BackgroundScheduler(daemon=True)
    scheduler.start()
    _sched = scheduler
    _restore_jobs()
    return scheduler


def _restore_jobs() -> None:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT id,title,when_ts,channel FROM reminders WHERE status='scheduled'"
        )
        for row in cur.fetchall():
            _schedule_job(int(row["id"]), row["title"], row["when_ts"], row["channel"])


def _schedule_job(rem_id: int, title: str, when_ts: str, channel: str) -> None:
    assert _sched is not None
    dt = datetime.fromisoformat(when_ts.replace("Z", "").replace(" ", "T"))
    _sched.add_job(
        _fire,
        "date",
        run_date=dt,
        args=[rem_id, title, channel],
        id=f"rem-{rem_id}",
        replace_existing=True,
    )
    log.info(f"Reminder scheduled {rem_id} at {dt}")


def _fire(rem_id: int, title: str, channel: str) -> None:
    insert_message("self", "assistant", f"🔔 Напоминание: {title}")
    with get_conn() as conn:
        conn.execute("UPDATE reminders SET status='sent' WHERE id=?", (rem_id,))
    log.info(f"Reminder fired {rem_id}")


def create_reminder(user_id: str, title: str, when_ts: str, channel: str = "cli") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO reminders(user_id,title,when_ts,channel) VALUES (?,?,?,?)",
            (user_id, title, when_ts, channel),
        )
        rem_id = cur.lastrowid
    init_scheduler()
    _schedule_job(rem_id, title, when_ts, channel)
    return int(rem_id)

