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
_JR_TOK = None
_JR_MDL = None


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


def _jr_load_transformers(cfg: Dict[str, Any]) -> None:
    """Lazily load the junior transformers model in 4-bit quantization."""

    global _JR_TOK, _JR_MDL

    if _JR_MDL is not None and _JR_TOK is not None:
        return

    model_id = cfg.get("model_id")
    if not model_id:
        raise ValueError("Junior transformers configuration requires 'model_id'.")

    dtype_cfg = cfg.get("torch_dtype", torch.float16)
    if isinstance(dtype_cfg, str):
        torch_dtype = getattr(torch, dtype_cfg, torch.float16)
    else:
        torch_dtype = dtype_cfg if isinstance(dtype_cfg, torch.dtype) else torch.float16

    bnb = BitsAndBytesConfig(
        load_in_4bit=bool(cfg.get("load_in_4bit", True)),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
    )

    _JR_TOK = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if _JR_TOK.pad_token is None and _JR_TOK.eos_token is not None:
        _JR_TOK.pad_token = _JR_TOK.eos_token

    _JR_MDL = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map=cfg.get("device_map", "auto"),
        torch_dtype=torch_dtype,
    )


def _jr_generate_transformers(
    cfg: Dict[str, Any],
    prompt: str,
    stop: Optional[List[str]],
    max_new_tokens: Optional[int],
    temperature: Optional[float],
    do_sample: bool,
    **extra: Any,
) -> str:
    """Generate junior outputs using transformers backend."""

    _jr_load_transformers(cfg)
    tokenizer = _JR_TOK
    model = _JR_MDL

    if tokenizer is None or model is None:
        raise RuntimeError("Junior transformers model failed to load.")

    if cfg.get("use_chat_template", True):
        messages = [
            {"role": "system", "content": "You return ONLY JSON inside <json>...</json>."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
    else:
        inputs = tokenizer(prompt, return_tensors="pt")

    inputs = inputs.to(model.device)
    input_length = inputs["input_ids"].shape[-1]

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        if tokenizer.pad_token_id is not None:
            eos_token_id = tokenizer.pad_token_id
        else:
            raise ValueError("Tokenizer requires an EOS or PAD token for generation.")

    top_p_value = extra.get("top_p")
    if top_p_value is None:
        top_p_value = cfg.get("top_p")
    if top_p_value is None:
        top_p_final = 0.9 if do_sample else 1.0
    else:
        top_p_final = float(top_p_value)

    rep_penalty = extra.get("repeat_penalty")
    if rep_penalty is None:
        rep_penalty = extra.get("repetition_penalty")
    if rep_penalty is None:
        rep_penalty = cfg.get("repetition_penalty")
    if rep_penalty is None:
        rep_penalty = cfg.get("repeat_penalty")
    if rep_penalty is None:
        rep_penalty = 1.05

    sampling_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens if max_new_tokens is not None else cfg.get("max_new_tokens", 160)),
        "do_sample": do_sample,
        "temperature": float(temperature if temperature is not None else cfg.get("temperature", 0.2)),
        "top_p": top_p_final,
        "repetition_penalty": float(rep_penalty),
        "pad_token_id": eos_token_id,
        "eos_token_id": eos_token_id,
    }

    output = model.generate(**inputs, **sampling_kwargs)
    generated_tokens = output[0][input_length:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return _apply_stops(text, stop or [])


def _generate_with_llama_cpp(
    cfg: Dict[str, Any],
    role: str,
    prompt: str,
    stop: Optional[List[str]],
    max_new_tokens: Optional[int],
    temperature: Optional[float],
    grammar: Optional[Any],
    repeat_penalty: Optional[float],
    top_p: Optional[float],
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
    top_p_value = top_p if top_p is not None else cfg.get("top_p")
    if top_p_value is not None:
        completion_kwargs["top_p"] = float(top_p_value)

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
    top_p: Optional[float],
    do_sample: bool,
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
    final_temperature = float(temperature if temperature is not None else cfg.get("temperature", 0.7))
    top_p_value = top_p if top_p is not None else cfg.get("top_p")
    repetition = repeat_penalty if repeat_penalty is not None else cfg.get("repetition_penalty", 1.05)

    gen = _SEN_MDL.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens if max_new_tokens is not None else cfg.get("max_new_tokens", 512)),
        do_sample=do_sample,
        temperature=final_temperature,
        top_p=float(top_p_value) if top_p_value is not None else (0.92 if do_sample else 1.0),
        repetition_penalty=float(repetition),
        pad_token_id=_SEN_TOK.eos_token_id,
        eos_token_id=_SEN_TOK.eos_token_id,
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
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
) -> str:
    """Generate a local LLM response for the requested role."""

    if repair:
        if temperature is None:
            temperature = 0.1
        else:
            temperature = max(0.0, min(0.2, float(temperature)))

    if repetition_penalty is not None:
        repeat_penalty = repetition_penalty

    cfg = _model_cfg(role)
    provider = cfg.get("provider", "")

    default_temp = cfg.get("temperature")
    if default_temp is None:
        default_temp = 0.7 if role != "junior" else 0.2
    effective_temp = temperature if temperature is not None else default_temp
    try:
        temp_value = float(effective_temp)
    except (TypeError, ValueError):
        temp_value = float(default_temp) if isinstance(default_temp, (int, float)) else 0.7
    do_sample = True
    if temp_value <= 0:
        do_sample = False
        temp_value = 1e-6
    temperature = temp_value

    if role == "junior" and provider == "transformers":
        return _jr_generate_transformers(
            cfg,
            prompt,
            stop,
            max_new_tokens,
            temperature,
            do_sample,
            grammar=grammar,
            repeat_penalty=repeat_penalty,
            top_p=top_p,
        )

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
            top_p,
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
            top_p,
            do_sample,
        )
    raise ValueError(f"Unsupported provider: {provider}")
