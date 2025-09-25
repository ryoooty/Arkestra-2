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
import json
from jsonschema import validate, ValidationError
from pathlib import Path
import yaml
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
    raw = raw.strip()
    if "</json>" not in raw:
        raw = f"{raw}</json>"
    start_tag, end_tag = "<json>", "</json>"
    start_idx = raw.find(start_tag)
    end_idx = raw.find(end_tag, start_idx + len(start_tag))
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        raise RuntimeError(f"Senior returned response without <json> wrapper: {raw}")
    json_payload = raw[start_idx + len(start_tag):end_idx].strip()
    # Extract JSON
    try:
        data = json.loads(json_payload)
    except Exception:
        start = json_payload.find("{"); end = json_payload.rfind("}")
        if start >= 0 and end >= 0 and end > start:
            data = json.loads(json_payload[start:end+1])
        else:
            raise RuntimeError(f"Senior returned non-JSON: {raw}")
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
