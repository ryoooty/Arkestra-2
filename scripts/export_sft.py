"""
Export approved assistant replies to SFT staging JSONL.
"""

import json
from pathlib import Path

from app.memory.db import get_conn


def export_sft(path: str = "data/sft_staging.jsonl"):
    with get_conn() as c:
        cur = c.execute(
            "SELECT text FROM messages WHERE role='assistant' AND approved=1 ORDER BY id"
        )
        rows = [dict(r) for r in cur.fetchall()]
    Path("data").mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            rec = {"prompt": "", "completion": r["text"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Exported {len(rows)} replies to {path}")


if __name__ == "__main__":
    export_sft()
