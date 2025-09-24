"""
budget.py — токен-бюджет промпта.
Приоритет включения: history → RAG → junior_meta.
Если не помещаемся — режем в порядке обратного приоритета: junior_meta, затем RAG (низший score), затем хвост истории.
"""

from typing import Dict, List, Any
from app.core.tokens import count_tokens


def trim(history: List[Dict], rag_hits: List[Dict], junior_meta: Dict, max_tokens: int) -> Dict[str, Any]:
    # naive packing: we only return what fits roughly by token counts
    out = {"history": [], "rag_hits": [], "junior_meta": junior_meta}
    used = 0

    # pack history first
    for m in reversed(history):  # start from latest
        t = count_tokens(f"{m.get('role','')}:{m.get('text','')}")
        if used + t > max_tokens * 0.6:
            break
        out["history"].insert(0, m)
        used += t

    # then RAG
    for d in rag_hits:
        t = count_tokens(d.get("text", ""))
        if used + t > max_tokens * 0.9:
            break
        out["rag_hits"].append(d)
        used += t

    # junior_meta as string size estimate
    jm = str(junior_meta)[:1000]
    if used + len(jm) // 4 < max_tokens:
        out["junior_meta"] = junior_meta
    else:
        out["junior_meta"] = {}

    return out
