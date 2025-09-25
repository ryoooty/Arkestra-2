"""LLM client wrappers for junior (≤3B) and senior (7B)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_cfg = yaml.safe_load(Path("config/llm.yaml").read_text(encoding="utf-8"))

_llama_junior: Optional[Any] = None


def _model_cfg(role: str) -> Dict[str, Any]:
    return _cfg.get(role, {})


def generate(
    role: str,
    prompt: str,
    stop: Optional[list[str]] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """Call configured LLM endpoint for the requested role."""

    cfg = _model_cfg(role)
    provider = cfg.get("provider")
    if role == "junior" and provider == "llama-cpp":
        return _generate_with_llama_cpp(cfg, prompt, stop, max_new_tokens, temperature)

    if provider:
        msg = (
            "Only local llama-cpp provider is supported. "
            "Update config/llm.yaml to use provider='llama-cpp'."
        )
        raise ValueError(msg)

    return '{"stub":"replace with real LLM call"}'


def _generate_with_llama_cpp(
    cfg: Dict[str, Any],
    prompt: str,
    stop: Optional[list[str]],
    max_new_tokens: Optional[int],
    temperature: Optional[float],
) -> str:
    """Generate a completion using a cached llama.cpp model instance."""

    global _llama_junior

    if _llama_junior is None:
        from llama_cpp import Llama

        model_path = cfg.get("model_path")
        if not model_path:
            raise ValueError("junior llama-cpp configuration requires 'model_path'")

        _llama_junior = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=12,
        )

    llama = _llama_junior
    response = llama.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature if temperature is not None else cfg.get("temperature", 0.7),
        max_tokens=max_new_tokens if max_new_tokens is not None else cfg.get("max_new_tokens", 256),
        stop=stop if stop is not None else cfg.get("stop"),
    )

    return response["choices"][0]["message"]["content"]

