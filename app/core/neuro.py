"""Модуль управления нейромедиаторами и стилем общения."""

from typing import Dict
from pathlib import Path

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback
    yaml = None

    from app.util import simple_yaml

    def _load_yaml(path: Path):
        return simple_yaml.loads(path.read_text(encoding='utf-8'))
else:
    def _load_yaml(path: Path):
        return yaml.safe_load(path.read_text(encoding='utf-8'))

_PERSONA = None
_LEVELS: Dict[str, int] = {}
_MIN: Dict[str, int] = {}
_BASE: Dict[str, int] = {}
_MAX: Dict[str, int] = {}
_BIAS: Dict[str, Dict[str, float]] = {}
_SLEEP: Dict[str, int] = {}


def _load_persona():
    global _PERSONA, _MIN, _BASE, _MAX, _BIAS, _SLEEP, _LEVELS
    if _PERSONA is None:
        data = _load_yaml(Path('config/persona.yaml'))
        _PERSONA = data
        _MIN = data["neuro_baselines"]["min"]
        _BASE = data["neuro_baselines"]["base"]
        _MAX = data["neuro_baselines"]["max"]
        _BIAS = data["bias_map"]
        _SLEEP = data["sleep_reset_profile"]
        _LEVELS = _BASE.copy()


def snapshot() -> Dict[str, int]:
    _load_persona()
    return _LEVELS.copy()


def _clip_set(new_levels: Dict[str, int]) -> None:
    for k, v in new_levels.items():
        lo = _MIN.get(k, 1)
        hi = _MAX.get(k, 11)
        _LEVELS[k] = max(lo, min(hi, int(v)))


def set_levels(levels: Dict[str, int]) -> None:
    _load_persona()
    _clip_set(levels)


def apply_delta(deltas: Dict[str, int]) -> None:
    _load_persona()
    merged = {k: _LEVELS.get(k, _BASE.get(k, 5)) + int(v) for k, v in deltas.items()}
    for k in merged:
        merged[k] = merged[k]
    _clip_set({**_LEVELS, **merged})


def bias_to_style() -> Dict[str, float]:
    """
    Суммируем вклад каждого медиатора по bias_map.
    Возвращаем плоский пресет для senior.
    """

    _load_persona()
    out: Dict[str, float] = {
        "temperature": 0.7,
        "max_tokens": 512,
        "structure_bias": 0.0,
        "ask_clarify_bias": 0.0,
        "humor_bias": 0.0,
        "politeness": 0.0,
        "energy": 0.0,
        "assertiveness": 0.0,
        "we_pronouns": 0.0,
        "memory_write_bias": 0.0,
    }
    for med, lvl in _LEVELS.items():
        if med not in _BIAS:
            continue
        coef_map = _BIAS[med]
        for key, delta in coef_map.items():
            out[key] = out.get(key, 0.0) + float(delta) * (
                (lvl - _BASE.get(med, 5)) / max((_MAX.get(med, 11) - _MIN.get(med, 1)), 1)
            )
    out["temperature"] = max(0.1, min(1.2, out.get("temperature", 0.7)))
    out["max_tokens"] = int(max(128, min(1024, 512 + out.get("max_tokens", 0))))
    return out


def sleep_reset() -> None:
    _load_persona()
    set_levels(_SLEEP)
