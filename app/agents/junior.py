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
import re, json
from textwrap import dedent
from pathlib import Path
from jsonschema import validate, ValidationError

from app.core.llm import generate as llm_generate

JR_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "suggestions": {"type": "array"},
        "rag_query": {"type": ["string", "null"]},
        "rag_targets": {"type": ["array", "null"]},
        "style_directive": {"type": ["string", "null"]},
        "memory_hint": {"type": "array"},
        "neuro_update": {
            "type": "object",
            "properties": {
                "levels": {
                    "type": "object",
                    "properties": {
                        "dopamine": {"type": "integer"},
                        "serotonin": {"type": "integer"},
                        "norepinephrine": {"type": "integer"},
                        "acetylcholine": {"type": "integer"},
                    },
                    "required": [
                        "dopamine",
                        "serotonin",
                        "norepinephrine",
                        "acetylcholine",
                    ],
                }
            },
            "required": ["levels"],
        },
        "proactive": {
            "type": "object",
            "properties": {
                "allow": {"type": "boolean"},
                "reason": {"type": "string"},
                "cooldown_s": {"type": "integer"},
            },
            "required": ["allow", "reason", "cooldown_s"],
        },
        "tools_hint": {"type": ["string", "null"]},
        "tools_request": {"type": ["array", "null"]},
    },
    "required": ["intent", "suggestions", "memory_hint", "neuro_update", "proactive"],
}

STRICT_JSON_JR = dedent(
    """Return ONLY valid JSON according to the schema.
No markdown, no code fences, no extra text. Output must be inside <json>...</json>."""
)


def _extract_json(raw: str) -> str | None:
    if not raw:
        return None
    m = re.search(r"<json>(.+?)</json>", raw, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"```json\s*(.+?)```", raw, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e != -1 and e > s:
        return raw[s : e + 1].strip()
    return None


_SCHEMA = json.loads(Path("config/schemas/junior.v2.schema.json").read_text(encoding="utf-8"))

_SYS = """You are JUNIOR for Arkestra.
Return ONLY compact JSON v2 with keys: intent, tools_hint, (optional) tools_request, rag_query, style_directive, neuro_update.levels, neuro_update.reason.
Do NOT write the user's final answer.
Suggestions[].text must stay in the user's language, sound warm, and contain at least 12 words.
Keep under 120 tokens."""


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


def generate(payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    prompt = STRICT_JSON_JR + "\n\n" + _build_prompt(payload) + "\n\nOutput:\n<json>"
    requested_tokens = kwargs.get("max_new_tokens")
    if isinstance(requested_tokens, int):
        max_new_tokens = max(160, requested_tokens)
    else:
        max_new_tokens = 160

    stop_sequences = ["</json>"]
    base_temperature = kwargs.get("temperature", 0.2)
    repeat_penalty = kwargs.get("repeat_penalty", 1.1)

    def _run(temp: float) -> str:
        return llm_generate(
            "junior",
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            stop=stop_sequences,
            repeat_penalty=repeat_penalty,
        )

    raw = _run(base_temperature).strip()
    if raw and not raw.endswith("</json>"):
        raw = f"{raw}</json>"

    block = _extract_json(raw)
    if not block:
        retry_raw = _run(0.1).strip()
        if retry_raw and not retry_raw.endswith("</json>"):
            retry_raw = f"{retry_raw}</json>"
        block = _extract_json(retry_raw)
        if not block:
            snippet = retry_raw if retry_raw else raw
            raise RuntimeError(f"Junior returned non-JSON: {snippet[:2000]}")
        raw = retry_raw
    data = json.loads(block)

    if "neuro_update.levels" in data and "neuro_update" not in data:
        data["neuro_update"] = {"levels": data.pop("neuro_update.levels")}

    # Validate
    try:
        validate(instance=data, schema=_SCHEMA)
    except ValidationError as e:
        raise RuntimeError(f"Junior JSON v2 schema validation failed: {e.message}")

    data.setdefault("suggestions", [{"kind": "good", "text": ""}, {"kind": "mischief", "text": ""}])
    data.setdefault("memory_hint", [])
    data.setdefault("rag_query", None)
    data.setdefault("rag_targets", None)
    data.setdefault("style_directive", None)
    data.setdefault("tools_hint", None)
    data.setdefault("tools_request", None)
    data.setdefault("proactive", {"allow": False, "reason": "", "cooldown_s": 0})

    nu = data.get("neuro_update") or {}
    lv = nu.get("levels") or {}
    if not isinstance(lv, dict):
        # Неправильный формат от модели. Сбрасываем в пустой словарь,
        # чтобы последующая нормализация не падала.
        lv = {}
    for k in ("dopamine", "serotonin", "norepinephrine", "acetylcholine"):
        lv.setdefault(k, 0)
    nu["levels"] = lv
    data["neuro_update"] = nu

    return data
