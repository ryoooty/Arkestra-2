"""Simple FAISS-backed index with metadata (fallback to in-memory list)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency fallback
    np = None  # type: ignore[assignment]

try:
    import faiss
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

from app.rag.encoders import encode

DATA_DIR = Path("data") / "rag"
INDEX_PATH = DATA_DIR / "faiss.index"
ROWS_PATH = DATA_DIR / "rows.jsonl"
INFO_PATH = DATA_DIR / "meta.json"

_mem: List[Dict[str, Any]] = []  # fallback corpus
_index = None


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_index_from_disk():
    if not HAVE_FAISS or not INDEX_PATH.exists():
        return None
    try:
        return faiss.read_index(str(INDEX_PATH))
    except Exception:
        try:
            INDEX_PATH.unlink()
        except FileNotFoundError:  # pragma: no cover - filesystem race
            pass
        return None


def _write_rows(rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    _ensure_data_dir()
    with ROWS_PATH.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_meta(dim: int, encoder: str) -> None:
    _ensure_data_dir()
    INFO_PATH.write_text(
        json.dumps({"dim": int(dim), "encoder": encoder}, ensure_ascii=False),
        encoding="utf-8",
    )


def reset_index() -> None:
    """Remove on-disk FAISS index and metadata."""

    global _index

    _index = None
    for path in (INDEX_PATH, INFO_PATH):
        try:
            path.unlink()
        except FileNotFoundError:
            continue


def add_texts(X: np.ndarray, encoder_name: str | None = None) -> None:
    """Persist embeddings into the FAISS index, recreating when dims change."""

    global _index, _mem

    if np is None:
        raise RuntimeError("NumPy is required for FAISS indexing but is not available")

    rows: Sequence[Dict[str, Any]] | None = None

    if isinstance(X, list) and X and isinstance(X[0], dict):
        rows = X  # type: ignore[assignment]
        _mem.extend(rows)
        if not HAVE_FAISS:
            _write_rows(rows)
            return
        vectors = encode([row["text"] for row in rows])
        X = np.asarray(vectors, dtype=np.float32)
        if encoder_name is None:
            from app.rag.encoders import get_encoder_name  # local import to avoid cycle

            encoder_name = get_encoder_name()
    else:
        X = np.asarray(X, dtype=np.float32)
        if encoder_name is None:
            from app.rag.encoders import get_encoder_name  # local import to avoid cycle

            encoder_name = get_encoder_name()

    if X.ndim != 2:
        raise ValueError("Expected embeddings with shape (n, d)")

    X = np.ascontiguousarray(X, dtype=np.float32)

    if not HAVE_FAISS:
        return

    _ensure_data_dir()

    dim = int(X.shape[1])

    existing_dim = None
    if INFO_PATH.exists():
        try:
            meta_data = json.loads(INFO_PATH.read_text(encoding="utf-8"))
            existing_dim = int(meta_data.get("dim", dim))
        except Exception:
            existing_dim = None

    index = None
    if existing_dim is not None and existing_dim != dim:
        reset_index()
    else:
        index = _index or _load_index_from_disk()
        if index is not None and getattr(index, "d", None) != dim:
            reset_index()
            index = None

    if index is None:
        index = faiss.IndexFlatIP(dim)

    index.add(X)
    faiss.write_index(index, str(INDEX_PATH))
    _write_meta(dim, encoder_name or "unknown")

    if rows:
        _write_rows(rows)

    _index = index

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
    if not INDEX_PATH.exists() or not ROWS_PATH.exists():
        return []
    index = faiss.read_index(str(INDEX_PATH))
    if np is None:
        raise RuntimeError("NumPy is required for FAISS indexing but is not available")

    qv = encode([query])
    Q = np.ascontiguousarray(np.asarray(qv, dtype="float32"))
    D, I = index.search(Q, k)
    # read meta
    meta_rows = [json.loads(line) for line in ROWS_PATH.read_text(encoding="utf-8").splitlines()]
    out = []
    for dist, idx in zip(D[0], I[0]):
        if idx<0 or idx>=len(meta_rows): continue
        m = meta_rows[idx]
        out.append({"id": m["id"], "text": m["text"], "meta": m.get("meta",{}), "score": float(dist)})
    return out
