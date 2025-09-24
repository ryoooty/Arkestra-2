from typing import List, Dict
from app.rag.index import search as idx_search
from app.rag.encoders import encode, cos_sim

def search(query: str, intent: str) -> List[Dict]:
    if not query: return []
    cands = idx_search(query, k=6) or []
    if not cands:
        return []
    # rerank
    reranked = rerank_e5(query, cands, top_k=3)
    return reranked

def rerank_e5(query: str, docs: List[Dict], top_k: int) -> List[Dict]:
    qv = encode([query])[0]
    dv = encode([d["text"] for d in docs])
    rescored = []
    for d, vec in zip(docs, dv):
        sim = cos_sim(qv, vec)
        rescored.append({**d, "rerank": float(sim)})
    rescored.sort(key=lambda x: x["rerank"], reverse=True)
    return rescored[:top_k]
