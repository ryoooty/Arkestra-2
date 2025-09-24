"""
LLM client wrappers for junior (≤3B) and senior (7B).
Config-driven via config/llm.yaml.
Supports: local HTTP endpoint or python callable (plug your inference).
"""

from typing import Dict, Any, Optional
import yaml
from pathlib import Path

_cfg = yaml.safe_load(Path("config/llm.yaml").read_text(encoding="utf-8"))


def _model_cfg(role: str) -> Dict[str, Any]:
    return _cfg.get(role, {})


def generate(role: str, prompt: str, stop: Optional[list[str]] = None, max_new_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
    """
    Replace with your inference code. For now, returns a stub.
    """
    cfg = _model_cfg(role)
    # TODO: integrate with your inference server / transformers pipeline.
    # This is a hard stub to keep project runnable.
    return '{"stub":"replace with real LLM call"}'
