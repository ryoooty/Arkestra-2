"""
Simple FAISS-backed index with metadata (fallback to in-memory list).
"""

from typing import List, Dict, Any
import json
from pathlib import Path

try:
    import faiss
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

from app.rag.encoders import encode

DATA_DIR = Path("rag_store"); DATA_DIR.mkdir(exist_ok=True)
META_PATH = DATA_DIR / "meta.jsonl"
INDEX_PATH = DATA_DIR / "index.faiss"

_mem: List[Dict[str, Any]] = []  # fallback corpus

def add_texts(rows: List[Dict[str, Any]]):
    """
    rows: [{"id": "...", "text":"...", "meta": {...}}]
    """
    global _mem
    _mem.extend(rows)
    if not HAVE_FAISS: return
    vecs = encode([r["text"] for r in rows])
    dim = len(vecs[0])
    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
    else:
        index = faiss.IndexFlatIP(dim)
    import numpy as np
    X = np.array(vecs, dtype="float32")
    index.add(X)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "a", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")

def search(query: str, k: int = 6) -> List[Dict[str, Any]]:
    if not HAVE_FAISS:
        # naive fallback
        q = query.lower().split()
        scored = []
        for r in _mem:
            s = sum(1 for t in q if t in r["text"].lower())
            if s>0: scored.append({"id":r["id"], "text": r["text"], "score": float(s), "meta": r.get("meta",{})})
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:k]
    # FAISS search
    from app.rag.encoders import encode
    import numpy as np, json
    if not INDEX_PATH.exists() or not META_PATH.exists(): return []
    index = faiss.read_index(str(INDEX_PATH))
    qv = encode([query])
    D, I = index.search(np.array(qv, dtype="float32"), k)
    # read meta
    meta_rows = [json.loads(line) for line in META_PATH.read_text(encoding="utf-8").splitlines()]
    out = []
    for idx in I[0]:
        if idx<0 or idx>=len(meta_rows): continue
        m = meta_rows[idx]
        out.append({"id": m["id"], "text": m["text"], "meta": m.get("meta",{}), "score": float(1.0)})
    return out
