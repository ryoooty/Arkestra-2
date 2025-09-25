from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

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
except Exception:  # pragma: no cover - optional dependency fallback
    SentenceTransformer = None  # type: ignore[assignment]


_e5_model = None
_e5_failed = False


_qwen_tokenizer = None
_qwen_model = None
_active_encoder: Optional[str] = None


def encode(texts: List[str]):
    encoder_name = _resolve_encoder()

    if encoder_name == "qwen3-0.6b":
        vectors = _encode_with_qwen(texts)
        if vectors is not None:
            return vectors
        _reset_active_encoder()
        encoder_name = _resolve_encoder()

    if encoder_name == "e5-small":
        vectors = _encode_with_e5(texts)
        if vectors is not None:
            return vectors
        _reset_active_encoder()
        encoder_name = _resolve_encoder()

    vectors = _encode_with_hash(texts)
    _set_active_encoder("hash")
    return vectors


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

    vec = outputs.last_hidden_state[:, 0, :]
    vec = F.normalize(vec, dim=-1)
    vec = vec.detach().cpu().to(dtype=torch.float32).numpy()
    return _ensure_numpy(vec)


def _encode_with_e5(texts: List[str]):
    model = _load_e5()
    if model is None or np is None:
        return None

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return _ensure_numpy(embeddings)


def _encode_with_hash(texts: List[str]):
    if np is None:
        return [[hash(t) % 1000 / 1000.0 for _ in range(32)] for t in texts]

    data = np.array(
        [[hash(t) % 1000 / 1000.0 for _ in range(32)] for t in texts],
        dtype=np.float32,
    )
    return _ensure_numpy(data)


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


def _load_e5():
    global _e5_model, _e5_failed
    if _e5_failed:
        return None
    if _e5_model is not None:
        return _e5_model
    if SentenceTransformer is None:
        _e5_failed = True
        return None

    try:  # pragma: no cover - optional dependency fallback
        _e5_model = SentenceTransformer("intfloat/e5-small-v2")
    except Exception:  # pragma: no cover - optional dependency fallback
        _e5_failed = True
        _e5_model = None
    return _e5_model


def _ensure_numpy(array):
    if np is None:
        return array

    if not isinstance(array, np.ndarray):
        array = np.asarray(array)

    array = array.astype(np.float32, copy=False)
    if array.ndim == 2:
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        array = array / norms
    return np.ascontiguousarray(array)


def _resolve_encoder() -> str:
    if _active_encoder is not None:
        return _active_encoder

    encoder = _get_encoder_name()

    if encoder == "qwen3-0.6b":
        tokenizer, model = _load_qwen()
        if tokenizer is not None and model is not None:
            _set_active_encoder("qwen3-0.6b")
            return "qwen3-0.6b"
        encoder = "e5-small"

    if encoder == "e5-small":
        if _load_e5() is not None:
            _set_active_encoder("e5-small")
            return "e5-small"

    if _load_e5() is not None:
        _set_active_encoder("e5-small")
        return "e5-small"

    _set_active_encoder("hash")
    return "hash"


def _set_active_encoder(name: str) -> None:
    global _active_encoder
    _active_encoder = name


def _reset_active_encoder() -> None:
    global _active_encoder
    _active_encoder = None


def get_encoder_name() -> str:
    if _active_encoder is not None:
        return _active_encoder
    return _resolve_encoder()
