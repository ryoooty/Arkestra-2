"""Local-only LLM client wrappers for junior (≤3B) and senior (7B) roles."""

from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_CFG: Optional[Dict[str, Any]] = None
_LLAMA_JR = None
_SEN_TOK = None
_SEN_MDL = None


def _load_cfg() -> Dict[str, Any]:
    """Load the LLM configuration file once and reuse it."""

    global _CFG
    if _CFG is None:
        config_path = Path("config/llm.yaml")
        if not config_path.exists():
            _CFG = {}
        else:
            _CFG = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return _CFG


def _model_cfg(role: str) -> Dict[str, Any]:
    """Return the configuration section for a given model role."""

    return _load_cfg().get(role, {})


def _apply_stops(text: str, stops):
    if not stops:
        return text
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


def _generate_with_llama_cpp(
    cfg: Dict[str, Any],
    role: str,
    prompt: str,
    stop: Optional[List[str]],
    max_new_tokens: Optional[int],
    temperature: Optional[float],
    grammar: Optional[Any],
    repeat_penalty: Optional[float],
) -> str:
    global _LLAMA_JR
    from llama_cpp import Llama

    model_path = cfg.get("model_path")
    if not model_path or not Path(model_path).exists():
        raise ValueError(f"Junior GGUF not found: {model_path}")

    if _LLAMA_JR is None:
        _LLAMA_JR = Llama(
            model_path=model_path,
            n_ctx=int(cfg.get("n_ctx", 2048)),
            n_threads=min(12, os.cpu_count() or 8),
            n_gpu_layers=int(cfg.get("n_gpu_layers", -1)),
            chat_format=cfg.get("chat_format", "gemma"),
            verbose=False,
        )

    if role == "junior":
        stops = list(stop or [])
    else:
        stops = list(stop or cfg.get("stop") or [])
    if role in {"junior", "senior"} and "</json>" not in stops:
        stops.append("</json>")

    if role == "junior":
        requested_tokens = max_new_tokens if max_new_tokens is not None else cfg.get("max_new_tokens", 160)
        if requested_tokens is None:
            requested_tokens = 160
        max_tokens = max(160, int(requested_tokens))
    else:
        max_tokens = int(max_new_tokens if max_new_tokens is not None else cfg.get("max_new_tokens", 64))

    temperature_value = float(temperature if temperature is not None else cfg.get("temperature", 0.2))
    repeat_penalty_value = repeat_penalty
    if repeat_penalty_value is None:
        repeat_penalty_value = cfg.get("repeat_penalty")
    if repeat_penalty_value is None and role == "junior":
        repeat_penalty_value = 1.1

    completion_kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "temperature": temperature_value,
        "max_tokens": max_tokens,
    }
    if stops:
        completion_kwargs["stop"] = stops
    if grammar is not None:
        completion_kwargs["grammar"] = grammar
    if repeat_penalty_value is not None:
        completion_kwargs["repeat_penalty"] = float(repeat_penalty_value)

    out = _LLAMA_JR.create_completion(**completion_kwargs)
    text = out["choices"][0]["text"]
    trimmed = _apply_stops(text, stops)
    if stops and "</json>" in stops and "</json>" in text and "</json>" not in trimmed:
        trimmed = f"{trimmed}</json>"
    return trimmed


def _generate_with_transformers(
    cfg: Dict[str, Any],
    role: str,
    prompt: str,
    stop: Optional[List[str]],
    max_new_tokens: Optional[int],
    temperature: Optional[float],
    grammar: Optional[Any],
    repeat_penalty: Optional[float],
) -> str:
    global _SEN_TOK, _SEN_MDL

    model_path = cfg.get("model_path")
    if not model_path or not Path(model_path).exists():
        raise ValueError(f"Senior model path not found: {model_path}")

    if _SEN_TOK is None or _SEN_MDL is None:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        _SEN_TOK = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        _SEN_MDL = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            quantization_config=bnb,
            device_map="auto",
        )

    inputs = _SEN_TOK(prompt, return_tensors="pt").to(_SEN_MDL.device)
    stops = list(stop or cfg.get("stop") or [])
    if role == "senior" and "</json>" not in stops:
        stops.append("</json>")
    gen = _SEN_MDL.generate(
        **inputs,
        do_sample=True,
        temperature=float(temperature if temperature is not None else cfg.get("temperature", 0.7)),
        max_new_tokens=int(max_new_tokens if max_new_tokens is not None else cfg.get("max_new_tokens", 512)),
    )
    text = _SEN_TOK.decode(gen[0], skip_special_tokens=True)
    trimmed = _apply_stops(text, stops)
    if "</json>" in stops and "</json>" in text and "</json>" not in trimmed:
        trimmed = f"{trimmed}</json>"
    return trimmed


def generate(
    role: str,
    prompt: str,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    stop: Optional[List[str]] = None,
    repair: bool = False,
    grammar: Optional[Any] = None,
    repeat_penalty: Optional[float] = None,
) -> str:
    """Generate a local LLM response for the requested role."""

    if repair:
        if temperature is None:
            temperature = 0.1
        else:
            temperature = max(0.0, min(0.2, float(temperature)))

    cfg = _model_cfg(role)
    provider = cfg.get("provider", "")

    if provider == "llama-cpp":
        return _generate_with_llama_cpp(
            cfg,
            role,
            prompt,
            stop,
            max_new_tokens,
            temperature,
            grammar,
            repeat_penalty,
        )
    if provider == "transformers":
        return _generate_with_transformers(
            cfg,
            role,
            prompt,
            stop,
            max_new_tokens,
            temperature,
            grammar,
            repeat_penalty,
        )
    raise ValueError(f"Unsupported provider: {provider}")
