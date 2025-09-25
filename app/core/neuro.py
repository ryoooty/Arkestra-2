"""Neurotransmitter utilities and conversational style mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union


try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback
    yaml = None

    from app.util import simple_yaml

    def _load_yaml(path: Path) -> Dict[str, object]:
        """Load YAML content from *path* using the lightweight parser."""

        return simple_yaml.loads(path.read_text(encoding="utf-8"))
else:

    def _load_yaml(path: Path) -> Dict[str, object]:
        """Load YAML content from *path* using :mod:`yaml`."""

        return yaml.safe_load(path.read_text(encoding="utf-8"))


_NEUROTRANSMITTERS = (
    "dopamine",
    "serotonin",
    "norepinephrine",
    "acetylcholine",
    "gaba",
    "glutamate",
    "endorphins",
    "oxytocin",
    "vasopressin",
    "histamine",
)

_BASELINE: Dict[str, int] = {}
_LEVELS: Dict[str, int] = {}
_STYLE_BASELINE: Dict[str, float] = {}
_PERSONA_META: Dict[str, Any] = {}
_PERSONA_LOADED = False


def _clamp_level(value: Union[int, float]) -> int:
    """Clamp a neurotransmitter level to the inclusive ``[0, 11]`` range."""

    return max(0, min(11, int(round(float(value)))))


def _ensure_persona_loaded() -> None:
    """Load persona configuration and initialise baselines exactly once."""

    global _PERSONA_LOADED, _BASELINE, _LEVELS, _STYLE_BASELINE, _PERSONA_META
    if _PERSONA_LOADED:
        return

    data = _load_yaml(Path("config/persona.yaml")) or {}
    raw_baseline = data.get("baseline_levels", {}) if isinstance(data, dict) else {}
    raw_bias_map = data.get("bias_map", {}) if isinstance(data, dict) else {}
    raw_persona = data.get("persona", {}) if isinstance(data, dict) else {}

    baseline_items: Iterable[Tuple[str, int]]
    if isinstance(raw_baseline, dict):
        baseline_items = raw_baseline.items()
    else:
        baseline_items = []

    _BASELINE = {
        name: _clamp_level(level)
        for name, level in baseline_items
        if name in _NEUROTRANSMITTERS
    }

    for name in _NEUROTRANSMITTERS:
        _BASELINE.setdefault(name, 0)

    _LEVELS = dict(_BASELINE)

    if isinstance(raw_bias_map, dict):
        _STYLE_BASELINE = {
            "humor": float(raw_bias_map.get("humor", 0.0)),
            "structure": float(raw_bias_map.get("structure", 0.0)),
            "ask_clarify": float(raw_bias_map.get("ask_clarify", 0.0)),
            "brevity": float(raw_bias_map.get("brevity", 0.0)),
            "positivity": float(raw_bias_map.get("positivity", 0.0)),
            "warmth": float(raw_bias_map.get("warmth", 0.0)),
            "alertness": float(raw_bias_map.get("alertness", 0.0)),
        }
    else:
        _STYLE_BASELINE = {
            "humor": 0.0,
            "structure": 0.0,
            "ask_clarify": 0.0,
            "brevity": 0.0,
            "positivity": 0.0,
            "warmth": 0.0,
            "alertness": 0.0,
        }
    if isinstance(raw_persona, dict):
        _PERSONA_META = dict(raw_persona)
    else:
        _PERSONA_META = {}
    _PERSONA_LOADED = True


def snapshot() -> Dict[str, int]:
    """Return the current neurotransmitter levels as ``{name: level}``."""

    _ensure_persona_loaded()
    return {name: _LEVELS[name] for name in _NEUROTRANSMITTERS}


def set_levels(levels: Dict[str, Union[int, float]]) -> None:
    """Overwrite provided neurotransmitter levels (clamped into ``[0, 11]``)."""

    _ensure_persona_loaded()
    for name, value in levels.items():
        if name in _NEUROTRANSMITTERS:
            _LEVELS[name] = _clamp_level(value)


def apply_delta(deltas: Dict[str, Union[int, float]]) -> None:
    """Add additive adjustments from ``deltas`` and clamp the new levels."""

    _ensure_persona_loaded()
    for name, delta in deltas.items():
        if name not in _NEUROTRANSMITTERS:
            continue
        _LEVELS[name] = _clamp_level(_LEVELS.get(name, 0) + float(delta))


def sleep_reset() -> None:
    """Restore all neurotransmitter levels to the persona baseline profile."""

    _ensure_persona_loaded()
    _LEVELS.update(_BASELINE)


def decay_step(rate: float = 0.2) -> None:
    """Move each level a fraction of the gap towards its baseline and clamp."""

    _ensure_persona_loaded()
    if rate <= 0:
        return

    for name in _NEUROTRANSMITTERS:
        baseline = _BASELINE[name]
        current = _LEVELS[name]
        delta = baseline - current
        adjusted = round(current + delta * rate)
        _LEVELS[name] = _clamp_level(adjusted)


def _clamp_float(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp *value* to ``[lower, upper]``."""

    return max(lower, min(upper, value))


def bias_to_style() -> Dict[str, Union[int, float]]:
    """Convert the current neurochemical mix into style-control directives."""

    _ensure_persona_loaded()

    ratios = {name: _LEVELS.get(name, 0) / 11.0 for name in _NEUROTRANSMITTERS}

    temperature = 0.7
    max_tokens = 512.0

    humor = _STYLE_BASELINE.get("humor", 0.0)
    structure = _STYLE_BASELINE.get("structure", 0.0)
    ask_clarify = _STYLE_BASELINE.get("ask_clarify", 0.0)
    brevity = _STYLE_BASELINE.get("brevity", 0.0)
    warmth = _STYLE_BASELINE.get("warmth", 0.0)
    positivity = _STYLE_BASELINE.get("positivity", 0.0)
    alertness = _STYLE_BASELINE.get("alertness", 0.0)

    dopamine = ratios["dopamine"]
    serotonin = ratios["serotonin"]
    norepinephrine = ratios["norepinephrine"]
    acetylcholine = ratios["acetylcholine"]
    gaba = ratios["gaba"]
    glutamate = ratios["glutamate"]
    endorphins = ratios["endorphins"]
    oxytocin = ratios["oxytocin"]
    vasopressin = ratios["vasopressin"]
    histamine = ratios["histamine"]

    temperature += 0.2 * dopamine
    temperature -= 0.2 * gaba

    max_tokens += 200.0 * dopamine
    max_tokens -= 150.0 * norepinephrine
    max_tokens -= 50.0 * histamine

    humor += 0.5 * dopamine
    structure += 0.6 * acetylcholine
    structure += 0.1 * glutamate
    ask_clarify += 0.5 * acetylcholine
    brevity += 0.6 * norepinephrine
    brevity += 0.2 * vasopressin
    warmth += 0.5 * serotonin
    warmth += 0.4 * oxytocin
    positivity += 0.3 * serotonin
    positivity += 0.5 * endorphins
    alertness += 0.6 * histamine

    temperature = max(0.1, min(1.3, temperature))
    max_tokens = max(128.0, min(1024.0, max_tokens))

    return {
        "temperature": float(temperature),
        "max_tokens": int(round(max_tokens)),
        "humor_bias": _clamp_float(humor),
        "structure_bias": _clamp_float(structure),
        "ask_clarify_bias": _clamp_float(ask_clarify),
        "brevity_bias": _clamp_float(brevity),
        "warmth_bias": _clamp_float(warmth),
        "positivity_bias": _clamp_float(positivity),
        "alertness_bias": _clamp_float(alertness),
    }


def persona_brief() -> Dict[str, Any]:
    """Return a lightweight summary of the persona configuration."""

    _ensure_persona_loaded()
    name = str(_PERSONA_META.get("name") or "Arkestra")
    description = str(
        _PERSONA_META.get(
            "description",
            "Тёплый русскоязычный ИИ-компаньон, который поддерживает и помогает собеседнику.",
        )
    )
    return {
        "name": name,
        "description": description,
        "style_bias": dict(_STYLE_BASELINE),
    }


