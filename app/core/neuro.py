"""Neurotransmitter utilities and conversational style mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple, Union


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
_PERSONA_LOADED = False


def _clamp_level(value: int) -> int:
    """Clamp a neurotransmitter level to the inclusive ``[0, 11]`` range."""

    return max(0, min(11, int(value)))


def _ensure_persona_loaded() -> None:
    """Load persona configuration and initialise baselines exactly once."""

    global _PERSONA_LOADED, _BASELINE, _LEVELS
    if _PERSONA_LOADED:
        return

    data = _load_yaml(Path("config/persona.yaml")) or {}
    raw_baseline = data.get("baseline_levels", {}) if isinstance(data, dict) else {}

    baseline_items: Iterable[Tuple[str, int]]
    if isinstance(raw_baseline, dict):
        baseline_items = raw_baseline.items()
    else:
        baseline_items = []

    _BASELINE = {
        name: _clamp_level(int(level))
        for name, level in baseline_items
        if name in _NEUROTRANSMITTERS
    }

    for name in _NEUROTRANSMITTERS:
        _BASELINE.setdefault(name, 0)

    _LEVELS = dict(_BASELINE)
    _PERSONA_LOADED = True


def snapshot() -> Dict[str, int]:
    """Return the current neurotransmitter levels as ``{name: level}``."""

    _ensure_persona_loaded()
    return {name: _LEVELS[name] for name in _NEUROTRANSMITTERS}


def set_levels(levels: Dict[str, int]) -> None:
    """Overwrite provided neurotransmitter levels (clamped into ``[0, 11]``)."""

    _ensure_persona_loaded()
    for name, value in levels.items():
        if name in _NEUROTRANSMITTERS:
            _LEVELS[name] = _clamp_level(value)


def apply_delta(deltas: Dict[str, int]) -> None:
    """Add additive adjustments from ``deltas`` and clamp the new levels."""

    _ensure_persona_loaded()
    for name, delta in deltas.items():
        if name not in _NEUROTRANSMITTERS:
            continue
        _LEVELS[name] = _clamp_level(_LEVELS.get(name, 0) + int(delta))


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


def _ratio(name: str) -> float:
    """Return the current level scaled to the ``[0.0, 1.0]`` interval."""

    return _clamp_float(_LEVELS.get(name, 0) / 11.0)


def _clamp_float(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp *value* to ``[lower, upper]``."""

    return max(lower, min(upper, value))


def bias_to_style() -> Dict[str, Union[int, float]]:
    """Convert the current neurochemical mix into style-control directives."""

    _ensure_persona_loaded()

    dopamine = _ratio("dopamine")
    serotonin = _ratio("serotonin")
    norepinephrine = _ratio("norepinephrine")
    acetylcholine = _ratio("acetylcholine")
    gaba = _ratio("gaba")
    glutamate = _ratio("glutamate")
    endorphins = _ratio("endorphins")
    oxytocin = _ratio("oxytocin")
    vasopressin = _ratio("vasopressin")
    histamine = _ratio("histamine")

    temperature = 0.7 + 0.2 * dopamine - 0.2 * gaba
    temperature = max(0.1, min(1.3, temperature))

    max_tokens = int(round(512 + 200 * dopamine - 150 * norepinephrine - 50 * histamine))
    max_tokens = max(128, min(1024, max_tokens))

    return {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "humor_bias": _clamp_float(0.5 * dopamine),
        "structure_bias": _clamp_float(0.6 * acetylcholine + 0.1 * glutamate),
        "ask_clarify_bias": _clamp_float(0.5 * acetylcholine),
        "brevity_bias": _clamp_float(0.6 * norepinephrine + 0.2 * vasopressin),
        "warmth_bias": _clamp_float(0.5 * serotonin + 0.4 * oxytocin),
        "positivity_bias": _clamp_float(0.3 * serotonin + 0.5 * endorphins),
        "alertness_bias": _clamp_float(0.6 * histamine),
    }


