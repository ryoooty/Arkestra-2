"""
junior.py — DeepSeek (≤3B). Роль: диспетчер и регулятор.
ВХОД: history_tail, user_text, neuro_snapshot, env_brief, tools_catalog (название+краткое назначение).
ВЫХОД (JSON v2):
- intent
- tools_hint[] (имена инструментов из каталога)
- tools_request[] (если не хватает инструмента)
- rag_query?
- style_directive (короткая подсказка, как окрасить ответ)
- neuro_update.levels (целевые уровни нейро)
!!! Junior НЕ пишет финальный ответ пользователю и НЕ знает схем аргументов инструментов.
"""

from typing import Dict, Any
import json
from pathlib import Path
from jsonschema import validate, ValidationError
from app.core.llm import generate as llm_generate

_SCHEMA = json.loads(Path("config/schemas/junior.v2.schema.json").read_text(encoding="utf-8"))

_SYS = """You are JUNIOR for Arkestra. 
Return ONLY compact JSON v2 with keys: intent, tools_hint, (optional) tools_request, rag_query, style_directive, neuro_update.levels, neuro_update.reason.
Do NOT write the user's final answer. 
Keep under 120 tokens."""


def _build_prompt(payload: Dict[str, Any]) -> str:
    hist = payload.get("history_tail", [])
    user_text = payload.get("user_text","")
    neuro_snapshot = payload.get("neuro_snapshot", {})
    env_brief = payload.get("env_brief", {})
    tools_catalog = payload.get("tools_catalog", [])
    return (
        f"{_SYS}\n"
        f"ENV: {env_brief}\n"
        f"NEURO_SNAPSHOT: {neuro_snapshot}\n"
        f"TOOLS_CATALOG (name+desc only): {tools_catalog}\n"
        f"HISTORY_TAIL: {hist}\n"
        f"USER: {user_text}\n"
        f"Reply with JSON v2 only."
    )


def generate(payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = _build_prompt(payload)
    raw = llm_generate("junior", prompt, max_new_tokens=64, temperature=0.2, stop=["\n\n"])
    # Try to extract JSON
    try:
        data = json.loads(raw)
    except Exception:
        # try to strip code fences or text before/after
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end >= 0 and end > start:
            data = json.loads(raw[start:end+1])
        else:
            raise RuntimeError(f"Junior returned non-JSON: {raw}")
    # Validate
    try:
        validate(instance=data, schema=_SCHEMA)
    except ValidationError as e:
        raise RuntimeError(f"Junior JSON v2 schema validation failed: {e.message}")
    return data
