"""
senior.py — Mistral-7B. Роль: исполнитель и стилист.
ВХОД: history, user_text, rag_hits, junior_json_v2, preset (из нейро), style_directive, env_brief, tool_instructions (полные)
ВЫХОД (Reply JSON):
- text
- tool_calls[] (name, args)
- memory[] (facts/notes)
- plan[]
"""

from typing import Dict, Any
from textwrap import dedent
import re
import json
from jsonschema import validate, ValidationError
from pathlib import Path
import yaml
from json_repair import repair_json
from app.core.llm import generate as llm_generate

_SCHEMA = json.loads(Path("config/schemas/reply.schema.json").read_text(encoding="utf-8"))
_CFG = yaml.safe_load(Path("config/llm.yaml").read_text(encoding="utf-8"))


_SYS = dedent(
    """
    Вы — Arkestra Senior. Верните СТРОГО JSON по схеме reply.schema.json.

    Контекст и требования:
    - Вы SENIOR для Arkestra (Mistral-7B).
    - Поля JSON: text, tool_calls (опциональный массив {name,args}), memory (опциональный массив), plan (опциональный массив).
    - Пользовательский текст выводите в поле "text" без кодовых блоков.
    - Следуйте style_directive, preset, env_brief и инструкциям инструментов.
    - Если JUNIOR.tools_request содержит недостающие инструменты, в "text" вежливо попросите пользователя добавить их и кратко объясните зачем.

    Пример:
    <json>{"text":"ok","tool_calls":[],"memory":[],"plan":[]}</json>

    Отвечай ТОЛЬКО валидным JSON строго по схеме.
    Оберни JSON в теги <json> и </json>.
    Без какого-либо текста до/после.
    """
)


def _build_prompt(payload: Dict[str, Any]) -> str:
    history = payload.get("history", [])
    user_text = payload.get("user_text", "")
    rag_hits = payload.get("rag_hits", [])
    junior_json = payload.get("junior_json", {})
    preset = payload.get("preset", {})
    style_directive = payload.get("style_directive", "")
    env_brief = payload.get("env_brief", {})
    tool_instructions = payload.get("tool_instructions", {})

    payload_serialized = json.dumps(
        {
            "env_brief": env_brief,
            "preset": preset,
            "style_directive": style_directive,
            "tool_instructions": tool_instructions,
            "history": history,
            "rag_hits": rag_hits,
            "junior_json": junior_json,
            "user_text": user_text,
        },
        ensure_ascii=False,
    )

    user = dedent(
        f"""
        {payload_serialized}
        """
    ).strip()

    return f"{_SYS}\n\n{user}"


def _extract_json_block(raw: str) -> str | None:
    # 1) <json>...</json>
    m = re.search(r"<json>(.+?)</json>", raw, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    # 2)
    m = re.search(r"```json\s*(.+?)```", raw, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    # 3) Самые внешние { ... }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1].strip()
    return None


def _parse_reply(raw: str) -> dict:
    block = _extract_json_block(raw)
    if not block:
        raise ValueError("No JSON block found in model output")
    # Сначала строгий json
    try:
        return json.loads(block)
    except Exception:
        # Попробовать поправить мелкие косяки (одинарные кавычки, запятые и т.п.)
        fixed = repair_json(block)
        return json.loads(fixed)


def generate_structured(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a structured reply for the senior agent.

    The ``preset`` parameter originates from ``neuro.bias_to_style()`` and conveys
    sampling overrides (``temperature``), output length caps (``max_tokens``), and
    stylistic bias hints. Temperature and max token limits are forwarded directly
    into ``llm.generate`` so that the preset truly shapes generation dynamics.
    """

    prompt = _build_prompt(payload)
    cfg = _CFG.get("senior", {})
    preset = payload.get("preset") or {}
    temperature = preset.get("temperature")
    if not isinstance(temperature, (int, float)):
        temperature = cfg.get("temperature", 0.7)
    max_tokens = preset.get("max_tokens")
    if not isinstance(max_tokens, int):
        max_tokens = cfg.get("max_new_tokens", 512)
    raw = llm_generate(
        "senior",
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        stop=["</json>"],
    )
    try:
        data = _parse_reply(raw)
    except Exception:
        repair_prompt = dedent(
            f"""
            Предыдущий ответ невалидный. Верни ТОЛЬКО валидный JSON, по тем же правилам, между <json> и </json>.
            Без любого дополнительного текста. Исправь ошибки формата.

            ОРИГИНАЛ:
            {raw[-2000:]}
            """
        )
        raw2 = llm_generate(
            "senior",
            repair_prompt,
            max_new_tokens=256,
            temperature=0.0,
            stop=["</json>"],
        )
        if not raw2.strip().endswith("</json>"):
            raw2 = raw2.strip() + "</json>"
        data = _parse_reply(raw2)
    try:
        validate(instance=data, schema=_SCHEMA)
    except ValidationError:
        # minimal recovery to keep service alive
        data = {"text": data["text"]} if isinstance(data, dict) and "text" in data else {"text": "Извини, у меня сбой формата ответа."}
    # light sanity
    if "text" not in data:
        raise RuntimeError("Senior reply lacks 'text'")
    # Normalize tool_calls
    if "tool_calls" in data and not isinstance(data["tool_calls"], list):
        data["tool_calls"] = []
    return data


def refine_with_results(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload: same as generate_structured + {"tool_results":[...]}
    Returns JSON with at least 'text'.
    """

    base = _build_prompt(payload) + f"\nTOOL_RESULTS: {payload.get('tool_results', [])}\n"
    base += "Now briefly update 'text' to include outcomes. Reply JSON with just {'text': '...'}."
    temp = _CFG.get("senior", {}).get("temperature", 0.7)
    raw = llm_generate("senior", base, max_new_tokens=256, temperature=temp, stop=["\n\n"])
    try:
        data = json.loads(raw)
    except Exception:
        s = raw.find("{"); e = raw.rfind("}")
        data = json.loads(raw[s:e+1]) if s >= 0 and e > s else {"text": raw.strip()}
    if "text" not in data:
        data = {"text": str(raw).strip()}
    try:
        validate(instance=data, schema=_SCHEMA)
    except ValidationError:
        data = {"text": data["text"]} if isinstance(data, dict) and "text" in data else {"text": "Извини, у меня сбой формата ответа."}
    return data
