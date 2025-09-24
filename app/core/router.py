from typing import List, Dict

from app.rag.encoders import encode, cos_sim

# TEMP corpus: replace with your FAISS/BM25 store
_CORPUS = [
    {"id": "doc1", "text": "Мы добавили авто-сон: перенос short->temp и temp->long.", "date": "2025-09-20"},
    {"id": "doc2", "text": "Включён e5-small rerank для RAG, top_k=6 -> rerank_top_k=3.", "date": "2025-09-23"},
    {"id": "doc3", "text": "Junior отдаёт style_directive и neuro_update уровни.", "date": "2025-09-24"},
]


def search(query: str, intent: str) -> List[Dict]:
    if not query:
        return []

    # primary: naive BM25-like by term overlap score
    q_terms = set(query.lower().split())
    cands: List[Dict] = []
    for doc in _CORPUS:
        score = sum(1 for term in q_terms if term in doc["text"].lower())
        if score > 0:
            cands.append({**doc, "score": float(score)})

    if not cands:
        cands = [{**doc, "score": 0.1} for doc in _CORPUS]  # weak fallback

    # rerank
    reranked = rerank_e5(query, cands, top_k=3)
    return reranked


def rerank_e5(query: str, docs: List[Dict], top_k: int) -> List[Dict]:
    qv = encode([query])[0]
    dv = encode([doc["text"] for doc in docs])

    rescored = []
    for doc, vec in zip(docs, dv):
        sim = cos_sim(qv, vec)
        rescored.append({**doc, "rerank": float(sim)})

    rescored.sort(key=lambda item: item["rerank"], reverse=True)
    return rescored[:top_k]
