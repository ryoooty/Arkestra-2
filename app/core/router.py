from typing import List, Dict


# Заглушка индексов: возвращаем фиктивные документы с полем 'score'
# Позже подключите FAISS/BM25 + e5-small.
def search(query: str, intent: str) -> List[Dict]:
    if not query:
        return []
    # pretend candidates
    candidates = [
        {"id": "doc1", "text": "Вчера обсуждали авто-сон и переносы памяти.", "score": 0.62},
        {"id": "doc2", "text": "Планируем добавить rerank e5-small для RAG.", "score": 0.71},
        {"id": "doc3", "text": "Junior генерирует style_directive и neuro_update.", "score": 0.55},
    ]
    reranked = rerank_e5(query, candidates, top_k=2)
    return reranked


def rerank_e5(query: str, docs: List[Dict], top_k: int) -> List[Dict]:
    # Заглушка: просто сортировка по 'score' у кандидатов.
    docs_sorted = sorted(docs, key=lambda d: d.get("score", 0), reverse=True)
    return docs_sorted[:top_k]
