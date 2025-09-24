"""
budget.py — токен-бюджет промпта.
Приоритет включения: history → RAG → junior_meta.
Если не помещаемся — режем в порядке обратного приоритета: junior_meta, затем RAG (низший score), затем хвост истории.
"""

from typing import Dict, List, Any


def trim(history: List[Dict], rag_hits: List[Dict], junior_meta: Dict, max_tokens: int) -> Dict[str, Any]:
    raise NotImplementedError
