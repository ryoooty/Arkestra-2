"""
Processes ALL unconsolidated records since last batch, groups by day, writes temp (day_summaries + curated_facts),
pushes aged temp -> long, updates RAG corpora (replace with real indices), resets neuro, and prepares SFT/LoRA sets.
"""

from typing import List, Dict
from datetime import datetime, timedelta
from app.core import neuro
from app.core.summarize import summarize_day
from app.memory.db import get_conn

from scripts.export_sft import export_sft

from scripts.export_junior_lora import export_junior_lora



def _last_batch(conn) -> Dict:
    cur = conn.execute("SELECT id, to_seen_at FROM sleep_batches WHERE status='ok' ORDER BY finished_at DESC LIMIT 1")
    row = cur.fetchone()
    if not row:
        return {"to_seen_at": "1970-01-01 00:00:00"}
    return dict(row)


def run_sleep_batch():
    with get_conn() as c:
        c.execute("BEGIN")
        last = _last_batch(c)
        from_seen = last["to_seen_at"]
        to_seen = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        batch_id = f"batch-{to_seen}"

        # collect unconsolidated messages/notes/facts since last watermark
        cur = c.execute("SELECT id, text, ts FROM messages WHERE ts > ? ORDER BY ts ASC", (from_seen,))
        msgs = [dict(r) for r in cur.fetchall()]
        # bucket by date (UTC date from ts)
        buckets: Dict[str, List[str]] = {}
        for m in msgs:
            day = m["ts"][:10]
            buckets.setdefault(day, []).append(m["text"])

        processed = 0
        # temp store: create tables if not exist
        c.executescript("""
        CREATE TABLE IF NOT EXISTS day_summaries(id INTEGER PRIMARY KEY, date TEXT, text TEXT, salience REAL DEFAULT 0.5, batch_id TEXT);
        CREATE TABLE IF NOT EXISTS facts_curated(id INTEGER PRIMARY KEY, key TEXT, value TEXT, first_seen TEXT, last_seen TEXT, importance REAL DEFAULT 0.5, batch_id TEXT);
        CREATE TABLE IF NOT EXISTS long_days(id INTEGER PRIMARY KEY, date TEXT, summary TEXT, key_events TEXT);
        """)
        # write summaries
        for day, chunks in buckets.items():
            if not chunks:
                continue
            summary = summarize_day(chunks, max_tokens=400)
            c.execute("INSERT INTO day_summaries(date,text,salience,batch_id) VALUES (?,?,?,?)", (day, summary, 0.6, batch_id))
            processed += len(chunks)

        # temp TTL -> long (7 days)
        ttl_day = (datetime.utcnow()-timedelta(days=7)).strftime("%Y-%m-%d")
        cur = c.execute("SELECT date, GROUP_CONCAT(text,' ') AS all_text FROM day_summaries WHERE date <= ? GROUP BY date", (ttl_day,))
        to_promote = [dict(r) for r in cur.fetchall()]
        for row in to_promote:
            c.execute("INSERT OR REPLACE INTO long_days(date, summary, key_events) VALUES (?,?,?)", (row["date"], row["all_text"][:2000], "[]"))
            # (Optional) delete promoted rows from day_summaries
            c.execute("DELETE FROM day_summaries WHERE date = ?", (row["date"],))

        # write batch
        c.execute("INSERT INTO sleep_batches(id,started_at,finished_at,from_seen_at,to_seen_at,processed_count,status) VALUES (?,?,?,?,?,?,?)",
                  (batch_id, datetime.utcnow(), datetime.utcnow(), from_seen, to_seen, processed, "ok"))
        c.execute("COMMIT")

    export_sft()
    # neuro reset after batch
    neuro.sleep_reset()
    export_junior_lora()
    print(f"Sleep batch {batch_id} done. processed={processed}")
