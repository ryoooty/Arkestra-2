"""LLM client wrappers for junior (≤3B) and senior (7B)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import yaml

_cfg = yaml.safe_load(Path("config/llm.yaml").read_text(encoding="utf-8"))


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
    endpoint = cfg.get("endpoint")
    model = cfg.get("model_id")
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

