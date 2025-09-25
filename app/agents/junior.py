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

STRICT_JSON_JR = dedent("""Return ONLY valid JSON (double quotes), no markdown, no code fences.
Wrap JSON in <json> and </json>. No extra text.
Schema:
{
  "intent": "smalltalk|ask|task|joke|other",
  "suggestions": [{"kind":"good","text":"..."},{"kind":"mischief","text":"..."}],
  "rag_query": null | string,
  "rag_targets": null | string[],
  "style_directive": string | null,
  "memory_hint": string[],
  "neuro_update": { "levels": { "dopamine":int,"serotonin":int,"norepinephrine":int,"acetylcholine":int } },
  "proactive": {"allow":bool,"reason":string,"cooldown_s":int},
  "tools_hint": string | null,
  "tools_request": null | string[]
}
""")


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


def generate(payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = STRICT_JSON_JR + "\n\n" + _build_prompt(payload) + "\n\nOutput:\n<json>"
    raw = llm_generate(
        "junior",
        prompt,
        max_new_tokens=128,
        temperature=0.2,
        stop=["</json>", "\n```", "\n\n"],
    )
    if raw and not raw.strip().endswith("</json>"):
        raw = raw.strip() + "</json>"

    block = _extract_json(raw)
    if not block:
        raise RuntimeError(f"Junior returned non-JSON: {raw[:2000]}")
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
    for k in ("dopamine", "serotonin", "norepinephrine", "acetylcholine"):
        lv.setdefault(k, 0)
    nu["levels"] = lv
    data["neuro_update"] = nu

    return data
