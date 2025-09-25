"""
budget.py — токен-бюджет промпта.
Приоритет включения: history → RAG → junior_meta.
Если не помещаемся — режем в порядке обратного приоритета: junior_meta, затем RAG (низший score), затем хвост истории.
"""

from typing import Dict, List, Any
import json

from app.core.tokens import count_tokens


def trim(history: List[Dict], rag_hits: List[Dict], junior_meta: Dict, max_tokens: int) -> Dict[str, Any]:
    """Pack the prompt budget prioritising history → junior_meta → rag."""

    out: Dict[str, Any] = {"history": [], "rag_hits": [], "junior_meta": {}}
    used = 0
    history_cap = max_tokens - 128 if max_tokens > 128 else max_tokens
    history_cap = max(300, min(600, history_cap))

    # Always keep the latest messages, respecting the token cap and chronological order.
    min_messages = min(10, len(history))
    max_messages = 16
    for message in reversed(history):
        tokens = count_tokens(f"{message.get('role', '')}:{message.get('text', '')}")
        if out["history"] and len(out["history"]) >= max_messages:
            break
        if (
            out["history"]
            and len(out["history"]) >= min_messages
            and used + tokens > history_cap
        ):
            break
        out["history"].insert(0, message)
        used += tokens

    remaining = max_tokens - used if max_tokens > used else 0
    if remaining > 0 and junior_meta:
        jm_serialized = json.dumps(junior_meta, ensure_ascii=False)
        jm_tokens = count_tokens(jm_serialized)
        if jm_tokens <= remaining:
            out["junior_meta"] = junior_meta
            used += jm_tokens
        else:
            out["junior_meta"] = {}

    remaining = max_tokens - used if max_tokens > used else 0
    if remaining <= 0:
        return out

    for doc in rag_hits:
        tokens = count_tokens(doc.get("text", ""))
        if tokens > remaining:
            break
        out["rag_hits"].append(doc)
        remaining -= tokens

    if not out["junior_meta"]:
        out["junior_meta"] = junior_meta if junior_meta and used == 0 else out["junior_meta"]

    return out
