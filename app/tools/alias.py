"""
alias.add — добавить алиас (с коротким описанием).
alias.set_primary — сделать алиас основным.
"""

from typing import Dict

from app.memory.db import get_conn


def add(args: Dict) -> Dict:
    user_id = args.get("user_id", "self")
    alias = (args.get("alias") or "").strip()
    short_desc = (args.get("short_desc") or "").strip()
    if not alias:
        return {"ok": False, "error": "alias required"}
    with get_conn() as c:
        try:
            c.execute(
                "INSERT INTO aliases(user_id,alias,short_desc,is_primary) VALUES (?,?,?,0)",
                (user_id, alias, short_desc),
            )
        except Exception as e:  # pragma: no cover - sqlite3 errors vary
            return {"ok": False, "error": str(e)}
    return {"ok": True, "alias": alias}


def set_primary(args: Dict) -> Dict:
    user_id = args.get("user_id", "self")
    alias = (args.get("alias") or "").strip()
    if not alias:
        return {"ok": False, "error": "alias required"}
    with get_conn() as c:
        c.execute("UPDATE aliases SET is_primary=0 WHERE user_id=?", (user_id,))
        cur = c.execute(
            "UPDATE aliases SET is_primary=1 WHERE user_id=? AND alias=?",
            (user_id, alias),
        )
        updated = cur.rowcount or 0
        c.commit()
    return {"ok": bool(updated), "updated": int(updated)}
