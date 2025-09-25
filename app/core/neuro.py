"""Utilities for managing neurotransmitter levels and style presets."""

from typing import Dict, List
from pathlib import Path

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback
    yaml = None

    from app.util import simple_yaml

    def _load_yaml(path: Path):
        return simple_yaml.loads(path.read_text(encoding="utf-8"))
else:

    def _load_yaml(path: Path):
        return yaml.safe_load(path.read_text(encoding="utf-8"))


_PERSONA = None
_BASELINE: Dict[str, int] = {}
_LEVELS: Dict[str, int] = {}
_BIAS_MAP: Dict[str, List[str]] = {}

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


def _clamp(value: int) -> int:
    """Clamp neurotransmitter levels to the supported [0, 11] range."""

    return max(0, min(11, int(value)))


def _ensure_levels() -> None:
    """Populate _LEVELS for every known neurotransmitter using the baseline when missing."""

    for name in _NEUROTRANSMITTERS:
        if name not in _LEVELS:
            _LEVELS[name] = _clamp(_BASELINE.get(name, 0))


def _load_persona() -> None:
    """Lazy-load persona configuration and initialise baselines/levels."""

    global _PERSONA, _BASELINE, _LEVELS, _BIAS_MAP
    if _PERSONA is None:
        data = _load_yaml(Path("config/persona.yaml")) or {}
        _PERSONA = data
        baseline = data.get("baseline_levels", {})
        _BASELINE = {name: _clamp(value) for name, value in baseline.items()}
        _BIAS_MAP = data.get("bias_map", {})
        _LEVELS = {name: _clamp(level) for name, level in _BASELINE.items()}
        _ensure_levels()


def snapshot() -> Dict[str, int]:
    """Return a copy of the current neurotransmitter levels."""

    _load_persona()
    _ensure_levels()
    return {name: _LEVELS.get(name, _clamp(0)) for name in _NEUROTRANSMITTERS}


def _set_level(name: str, value: int) -> None:
    """Set a single neurotransmitter level with clamping."""

    if name not in _NEUROTRANSMITTERS:
        return
    _LEVELS[name] = _clamp(value)


def set_levels(levels: Dict[str, int]) -> None:
    """Replace multiple neurotransmitter levels at once."""

    _load_persona()
    for name, value in levels.items():
        _set_level(name, value)
    _ensure_levels()


def apply_delta(deltas: Dict[str, int]) -> None:
    """Apply additive deltas to levels while keeping them inside [0, 11]."""

    _load_persona()
    for name, delta in deltas.items():
        if name not in _NEUROTRANSMITTERS:
            continue
        current = _LEVELS.get(name, _BASELINE.get(name, 0))
        _LEVELS[name] = _clamp(current + int(delta))
    _ensure_levels()


def sleep_reset() -> None:
    """Reset neurotransmitter levels back to their baseline persona profile."""

    global _LEVELS

    _load_persona()
    _LEVELS = {name: _clamp(level) for name, level in _BASELINE.items()}
    _ensure_levels()


def decay_step(step: int = 1) -> None:
    """Move current levels one step closer to the baseline profile."""

    _load_persona()
    if step <= 0:
        return
    for name in _NEUROTRANSMITTERS:
        baseline = _BASELINE.get(name, 0)
        current = _LEVELS.get(name, baseline)
        if current < baseline:
            _LEVELS[name] = min(baseline, current + step)
        elif current > baseline:
            _LEVELS[name] = max(baseline, current - step)


def _ratio(name: str) -> float:
    """Return a level normalised to the [0.0, 1.0] range."""

    level = _LEVELS.get(name, _BASELINE.get(name, 0))
    return max(0.0, min(1.0, level / 11.0))


def _deviation(name: str) -> float:
    """Return the signed deviation from baseline normalised by the max level."""

    baseline = _BASELINE.get(name, 0)
    level = _LEVELS.get(name, baseline)
    return (level - baseline) / 11.0


def bias_to_style() -> Dict[str, float]:
    """Translate current neurotransmitter levels into model-style directives."""

    _load_persona()
    _ensure_levels()

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

    max_tokens = int(
        384
        + 256 * dopamine
        - 160 * norepinephrine
        - 120 * histamine
    )
    max_tokens = max(128, min(1024, max_tokens))

    temperature = 0.45 + 0.5 * dopamine - 0.35 * gaba + 0.15 * histamine
    temperature = max(0.1, min(1.4, temperature))

    style: Dict[str, float] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "humor_bias": min(1.0, 0.1 + 0.9 * dopamine + 0.2 * endorphins),
        "warmth_bias": min(1.0, 0.2 + 0.6 * serotonin + 0.2 * endorphins + 0.2 * oxytocin),
        "politeness": min(1.0, 0.2 + 0.7 * serotonin + 0.1 * gaba),
        "structure_bias": min(1.0, 0.1 + 0.7 * acetylcholine + 0.3 * norepinephrine),
        "ask_clarify_bias": min(1.0, 0.1 + 0.8 * acetylcholine),
        "calm_bias": min(1.0, 0.2 + 0.7 * gaba + 0.2 * serotonin - 0.2 * histamine),
        "memory_bias": min(1.0, 0.2 + 0.8 * glutamate + 0.3 * acetylcholine),
        "positivity_bias": min(1.0, 0.2 + 0.7 * endorphins + 0.2 * dopamine),
        "inclusive_language_bias": min(1.0, 0.2 + 0.7 * oxytocin + 0.2 * endorphins),
        "assertiveness_bias": min(1.0, 0.2 + 0.6 * vasopressin + 0.3 * norepinephrine - 0.2 * serotonin),
        "alertness_bias": min(1.0, 0.2 + 0.7 * histamine + 0.3 * norepinephrine - 0.3 * gaba),
        "sentence_length_bias": max(-1.0, min(1.0, 0.3 * dopamine - 0.6 * histamine - 0.4 * norepinephrine)),
        "detail_bias": min(1.0, 0.2 + 0.7 * norepinephrine + 0.3 * glutamate),
    }

    # Soft tone adjustments derived from deviations allow small swings around neutral baseline.
    style["soft_tone_bias"] = max(0.0, min(1.0, 0.5 + 3.0 * _deviation("serotonin") - 2.0 * _deviation("vasopressin")))
    style["protective_tone_bias"] = max(0.0, min(1.0, 0.4 + 3.0 * _deviation("vasopressin")))

    return style
