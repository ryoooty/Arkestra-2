"""Local-only LLM client wrappers for junior (≤3B) and senior (7B) roles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import os
import yaml

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_CFG: Optional[Dict[str, Any]] = None
_LLAMA_JUNIOR: Optional[Any] = None
_senior_mdl: Optional[Any] = None
_senior_tok: Optional[Any] = None


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


def _apply_stops(text: str, stop: Optional[list[str]]) -> str:
    """Truncate the text at the first occurrence of any stop sequence."""

    if not stop:
        return text

    earliest_index: Optional[int] = None
    for sequence in stop:
        if not sequence:
            continue
        idx = text.find(sequence)
        if idx != -1 and (earliest_index is None or idx < earliest_index):
            earliest_index = idx

    if earliest_index is None:
        return text
    return text[:earliest_index]


def generate(
    role: str,
    prompt: str,
    stop: Optional[list[str]] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """Generate a local LLM response for the requested role."""

    cfg = _model_cfg(role)
    provider = cfg.get("provider")

    if provider == "llama-cpp":
        return _generate_with_llama_cpp(cfg, prompt, stop, max_new_tokens, temperature)
    if provider == "transformers":
        return _generate_with_transformers(cfg, prompt, stop, max_new_tokens, temperature)

    suggestion = "llama-cpp" if role == "junior" else "transformers"
    raise ValueError(
        f"Unsupported provider '{provider}' for role '{role}'. "
        f"Please configure the '{suggestion}' provider for local inference."
    )


def _generate_with_llama_cpp(
    cfg: Dict[str, Any],
    prompt: str,
    stop: Optional[list[str]],
    max_new_tokens: Optional[int],
    temperature: Optional[float],
) -> str:
    """Generate a completion using a cached llama.cpp model instance."""

    global _LLAMA_JUNIOR
    if _LLAMA_JUNIOR is None:
        from llama_cpp import Llama

        model_path = cfg.get("model_path")
        if not model_path:
            raise ValueError("junior llama-cpp configuration requires 'model_path'")

        _LLAMA_JUNIOR = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=12,
        )

    llama = _LLAMA_JUNIOR
    response = llama.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature if temperature is not None else cfg.get("temperature", 0.7),
        max_tokens=max_new_tokens if max_new_tokens is not None else cfg.get("max_new_tokens", 256),
        stop=stop if stop is not None else cfg.get("stop"),
    )

    message = response["choices"][0]["message"]["content"]
    return _apply_stops(message, stop or cfg.get("stop"))


def _generate_with_transformers(
    cfg: Dict[str, Any],
    prompt: str,
    stop: Optional[list[str]],
    max_new_tokens: Optional[int],
    temperature: Optional[float],
) -> str:
    """Generate a completion using a cached Hugging Face transformers model."""

    global _senior_mdl, _senior_tok
    if _senior_mdl is None or _senior_tok is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_id = cfg.get("model_id") or "mistralai/Mistral-7B-Instruct-v0.3"

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        _senior_tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        _senior_mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
        )

    tokenizer = _senior_tok
    model = _senior_mdl

    default_temp = cfg.get("temperature", 0.7)
    default_max_tokens = cfg.get("max_new_tokens", 256)
    inputs = tokenizer(prompt, return_tensors="pt")

    target_device = None
    if hasattr(model, "device"):
        target_device = model.device
    else:
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            import torch

            target_device = torch.device("cpu")

    if target_device is not None:
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

    input_length = inputs["input_ids"].shape[-1]
    generation = model.generate(
        **inputs,
        temperature=temperature if temperature is not None else default_temp,
        max_new_tokens=max_new_tokens if max_new_tokens is not None else default_max_tokens,
        do_sample=(temperature if temperature is not None else default_temp) > 0,
    )

    generated_tokens = generation[0][input_length:]
    completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return _apply_stops(completion, stop or cfg.get("stop"))

