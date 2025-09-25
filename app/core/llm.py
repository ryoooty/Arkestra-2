"""Local-only LLM client wrappers for junior (≤3B) and senior (7B) roles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if TYPE_CHECKING:  # pragma: no cover - import is heavy but useful for typing
    from llama_cpp import Llama


_CFG: Optional[Dict[str, Any]] = None
_LLAMA_JUNIOR: Optional["Llama"] = None
_SENIOR_MODEL: Optional[Any] = None
_SENIOR_TOKENIZER: Optional[Any] = None


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


def _apply_stops(text: str, stop: Optional[Sequence[str]]) -> str:
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
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    stop: Optional[list[str]] = None,
) -> str:
    """Generate a local LLM response for the requested role."""

    cfg = _model_cfg(role)
    provider = cfg.get("provider")

    if provider == "llama-cpp":
        return _generate_with_llama_cpp(cfg, prompt, max_new_tokens, temperature, stop)
    if provider == "transformers":
        return _generate_with_transformers(cfg, prompt, max_new_tokens, temperature, stop)

    suggestion = "llama-cpp" if role == "junior" else "transformers"
    raise ValueError(
        f"Unsupported provider '{provider}' for role '{role}'. "
        f"Please configure the '{suggestion}' provider for local inference."
    )


def _generate_with_llama_cpp(
    cfg: Dict[str, Any],
    prompt: str,
    max_new_tokens: Optional[int],
    temperature: Optional[float],
    stop: Optional[Sequence[str]],
) -> str:
    """Generate a completion using a cached llama.cpp model instance."""

    global _LLAMA_JUNIOR

    model_path_str = cfg.get("model_path")
    if not model_path_str:
        raise ValueError("junior llama-cpp configuration requires 'model_path'")

    model_path = Path(model_path_str)
    if not model_path.exists():
        raise ValueError(f"junior llama-cpp model not found at '{model_path}'")

    if _LLAMA_JUNIOR is None:
        from llama_cpp import Llama

        _LLAMA_JUNIOR = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_threads=int(cfg.get("n_threads", 12)),
        )

    llama = _LLAMA_JUNIOR
    stop_sequences: list[str] = list(stop or cfg.get("stop") or [])
    response = llama.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature if temperature is not None else cfg.get("temperature", 0.7),
        max_tokens=max_new_tokens if max_new_tokens is not None else cfg.get("max_new_tokens", 256),
        stop=stop_sequences,
    )

    message = response["choices"][0]["message"]["content"]
    return _apply_stops(message, stop_sequences)


def _generate_with_transformers(
    cfg: Dict[str, Any],
    prompt: str,
    max_new_tokens: Optional[int],
    temperature: Optional[float],
    stop: Optional[Sequence[str]],
) -> str:
    """Generate a completion using a cached Hugging Face transformers model."""

    global _SENIOR_MODEL, _SENIOR_TOKENIZER

    model_path_str = cfg.get("model_path")
    if not model_path_str:
        raise ValueError("senior transformers configuration requires 'model_path'")

    model_path = Path(model_path_str)
    if not model_path.exists():
        raise ValueError(f"senior transformers model not found at '{model_path}'")

    if _SENIOR_MODEL is None or _SENIOR_TOKENIZER is None:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        _SENIOR_TOKENIZER = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            use_fast=True,
        )
        _SENIOR_MODEL = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            quantization_config=quant_config,
            device_map="auto",
        )

    tokenizer = _SENIOR_TOKENIZER
    model = _SENIOR_MODEL

    default_temp = cfg.get("temperature", 0.7)
    default_max_tokens = cfg.get("max_new_tokens", 256)
    stop_sequences: list[str] = list(stop or cfg.get("stop") or [])

    inputs = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    if hasattr(inputs, "to"):
        inputs = inputs.to(device)  # type: ignore[assignment]
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}  # type: ignore[assignment]

    generation = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature if temperature is not None else default_temp,
        max_new_tokens=max_new_tokens if max_new_tokens is not None else default_max_tokens,
    )

    completion = tokenizer.decode(generation[0], skip_special_tokens=True)
    if completion.startswith(prompt):
        completion = completion[len(prompt) :]
    return _apply_stops(completion, stop_sequences)

