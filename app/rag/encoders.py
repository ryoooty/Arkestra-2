from typing import List
import math

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    _e5 = SentenceTransformer("intfloat/e5-small-v2")
except Exception:  # pragma: no cover - optional dependency fallback
    _e5 = None
    np = None


def encode(texts: List[str]):
    if _e5 is None:
        # Fallback: simple hashing vector (toy)
        return [[hash(t) % 1000 / 1000.0 for _ in range(32)] for t in texts]
    return _e5.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def cos_sim(a, b):
    if np is None:
        # toy cosine
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb + 1e-8)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
