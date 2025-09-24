"""
Build dataset for micro-LoRA tuning junior:
Input: history_tail, user_text, neuro_snapshot, env_brief
Output: intent, tools_hint, rag_query?, style_directive, neuro_update.levels
"""

import json
from pathlib import Path
from app.memory.db import get_conn


def export_junior_lora(path: str = "data/junior_lora.jsonl"):
    with get_conn() as c:
        cur = c.execute(
            "SELECT m1.text as user_text, m2.text as reply, m2.approved as approved "
            "FROM messages m1 JOIN messages m2 ON m2.id=m1.id+1 "
            "WHERE m1.role='user' AND m2.role='assistant' AND m2.approved=1"
        )
        rows = [dict(r) for r in cur.fetchall()]

    Path("data").mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            # here we fake junior_json for demo; replace with real logs
            sample = {
                "input": {
                    "history_tail": [],
                    "user_text": r["user_text"],
                    "neuro_snapshot": {"dopamine": 7, "serotonin": 5},
                    "env_brief": {},
                },
                "output": {
                    "intent": "task",
                    "tools_hint": ["note.create"],
                    "rag_query": None,
                    "style_directive": "тёплый, поддерживающий тон",
                    "neuro_update": {"levels": {"dopamine": 8, "serotonin": 6}},
                },
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Exported {len(rows)} samples to {path}")


if __name__ == "__main__":
    export_junior_lora()
