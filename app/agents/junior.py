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

from typing import Dict, Any, Optional
import json
import re
from pathlib import Path
from jsonschema import validate, ValidationError
from app.core.llm import generate as llm_generate

_SCHEMA = json.loads(Path("config/schemas/junior.v2.schema.json").read_text(encoding="utf-8"))

_SYS = """You are JUNIOR for Arkestra.
Return ONLY compact JSON v2 with keys: intent, tools_hint, (optional) tools_request, rag_query, style_directive, neuro_update.levels, neuro_update.reason.
Do NOT write the user's final answer.
Keep under 120 tokens."""

STRICT_JSON_JR = (
    "You are JUNIOR for Arkestra. Return STRICT JSON v2, wrapped in <json> and </json>."
    " Include keys intent, tools_hint, optional tools_request, rag_query, style_directive,"
    " neuro_update.levels, neuro_update.reason. No prose before or after."
)


def _build_prompt(payload: Dict[str, Any]) -> str:
    hist = payload.get("history_tail", [])
    user_text = payload.get("user_text", "")
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


def _extract_json(raw: str) -> Optional[str]:
    if not raw:
        return None
    match = re.search(r"<json>(.+?)</json>", raw, flags=re.S | re.I)
    if match:
        return match.group(1).strip()
    match = re.search(r"```json\s*(.+?)```", raw, flags=re.S | re.I)
    if match:
        return match.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1].strip()
    return None


def generate(payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = _build_prompt(payload)
    raw = llm_generate("junior", prompt, max_new_tokens=64, temperature=0.2, stop=["\n\n"])
    data: Optional[Dict[str, Any]] = None
    try:
        block = _extract_json(raw)
        data = json.loads(block) if block else None
    except Exception:
        repair_prompt = (
            STRICT_JSON_JR
            + "\n\nPrevious output was invalid. Return ONLY JSON in <json>...</json> now.\n\nOutput:\n<json>"
        )
        raw2 = llm_generate(
            "junior",
            repair_prompt,
            max_new_tokens=128,
            temperature=0.1,
            stop=["</json>"],
        )
        if raw2 and not raw2.strip().endswith("</json>"):
            raw2 = raw2.strip() + "</json>"
        block = _extract_json(raw2)
        data = json.loads(block) if block else None

    if not data:
        raise RuntimeError(f"Junior returned non-JSON: {raw[:2000]}")
    # Validate
    try:
        validate(instance=data, schema=_SCHEMA)
    except ValidationError as e:
        raise RuntimeError(f"Junior JSON v2 schema validation failed: {e.message}")
    return data
