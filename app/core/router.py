"""
router.py — RAG-маршрутизация + реранк (e5-small).
Шаги: primary search (FAISS/BM25) → concat → rerank_e5(query, docs) → trim_to_budget().
Настройки в config/router.yaml: targets, window_days, top_k, rerank_top_k.
"""

from typing import List, Dict


def search(query: str, intent: str) -> List[Dict]:
    # 1) выбрать индексы по intent (short/temp/long/jokes)
    # 2) получить первичные кандидаты (top_k)
    # 3) rerank_e5() и вернуть rerank_top_k
    raise NotImplementedError


def rerank_e5(query: str, docs: List[Dict], top_k: int) -> List[Dict]:
    raise NotImplementedError
