"""LLM client wrappers for junior (≤3B) and senior (7B)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
import yaml

_cfg = yaml.safe_load(Path("config/llm.yaml").read_text(encoding="utf-8"))

_senior_tok = None
_senior_mdl = None


def _load_senior_transformer(cfg: Dict[str, Any]) -> Tuple[Any, Any]:
    """Load and cache the senior transformer model and tokenizer."""

    global _senior_tok, _senior_mdl

    if _senior_tok is None or _senior_mdl is None:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model_id = cfg["model_id"]
        _senior_tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if _senior_tok.pad_token_id is None and _senior_tok.eos_token_id is not None:
            _senior_tok.pad_token = _senior_tok.eos_token

        _senior_mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
        )
        _senior_mdl.eval()

    return _senior_tok, _senior_mdl


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
    model = cfg.get("model_id")

    if provider == "transformers":
        tokenizer, model_ref = _load_senior_transformer(cfg)
        import torch

        max_tokens = max_new_tokens or cfg.get("max_new_tokens", 256)
        temperature = (
            temperature if temperature is not None else cfg.get("temperature", 0.7)
        )
        stop_sequences = stop or cfg.get("stop") or []

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(model_ref.device)

        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        with torch.no_grad():
            generated = model_ref.generate(**inputs, **generation_kwargs)

        generated_tokens = generated[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        for stop_seq in stop_sequences:
            if stop_seq:
                idx = text.find(stop_seq)
                if idx != -1:
                    text = text[:idx]
                    break

        return text.strip()

    endpoint = cfg.get("endpoint")
    if not endpoint:
        return '{"stub":"replace with real LLM call"}'

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature if temperature is not None else cfg.get("temperature", 0.7),
        "max_tokens": max_new_tokens or cfg.get("max_new_tokens", 256),
        "stop": stop or cfg.get("stop"),
        "stream": False,
    }

    for attempt in range(3):
        try:
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {os.getenv('LLM_API_KEY', '')}",
                },
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            return data.get("text", json.dumps(data, ensure_ascii=False))
        except Exception:
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))

