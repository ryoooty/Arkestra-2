from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import List

try:  # pragma: no cover - optional dependency fallback
    import numpy as np
except Exception:  # pragma: no cover - optional dependency fallback
    np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency fallback
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency fallback
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency fallback
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency fallback
    AutoModel = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency fallback
    from sentence_transformers import SentenceTransformer

    _e5 = SentenceTransformer("intfloat/e5-small-v2")
except Exception:  # pragma: no cover - optional dependency fallback
    _e5 = None


_qwen_tokenizer = None
_qwen_model = None


def encode(texts: List[str]):
    encoder_name = _get_encoder_name()

    if encoder_name == "qwen3-0.6b":
        vectors = _encode_with_qwen(texts)
        if vectors is not None:
            return vectors

    if _e5 is not None:
        return _e5.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    # Fallback: simple hashing vector (toy)
    return [[hash(t) % 1000 / 1000.0 for _ in range(32)] for t in texts]


def cos_sim(a, b):
    if np is None:
        # toy cosine
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb + 1e-8)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _encode_with_qwen(texts: List[str]):
    if torch is None or F is None or AutoTokenizer is None or AutoModel is None:
        return None

    tokenizer, model = _load_qwen()
    if tokenizer is None or model is None:
        return None

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    normalized = F.normalize(cls_embeddings, p=2, dim=1)
    return normalized.detach().to("cpu").float().numpy()


@lru_cache(maxsize=1)
def _get_encoder_name() -> str:
    config = _load_rag_config()
    value = config.get("encoder", "e5-small")
    if isinstance(value, str):
        return value.lower()
    return "e5-small"


@lru_cache(maxsize=1)
def _load_rag_config():
    path = Path("config/rag.yaml")
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")
    try:  # pragma: no cover - optional dependency fallback
        import yaml

        return yaml.safe_load(text) or {}
    except Exception:  # pragma: no cover - optional dependency fallback
        from app.util import simple_yaml

        return simple_yaml.loads(text)


def _load_qwen():
    global _qwen_tokenizer, _qwen_model
    if _qwen_tokenizer is not None and _qwen_model is not None:
        return _qwen_tokenizer, _qwen_model

    try:  # pragma: no cover - optional dependency fallback
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B",
            trust_remote_code=True,
        )
        model = AutoModel.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()
    except Exception:  # pragma: no cover - optional dependency fallback
        return None, None

    _qwen_tokenizer = tokenizer
    _qwen_model = model
    return tokenizer, model
