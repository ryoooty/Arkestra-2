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
from app.core.logs import log

_SCHEMA = json.loads(Path("config/schemas/reply.schema.json").read_text(encoding="utf-8"))
_CFG = yaml.safe_load(Path("config/llm.yaml").read_text(encoding="utf-8"))


_SYS = dedent(
    """
    Вы — Arkestra Senior, тёплый и внимательный русскоязычный ассистент.
    Верни СТРОГО валидный JSON по схеме reply.schema.json.
    Требования:
    1. Единственное, что ты выводишь — JSON между тегами <json> и </json>.
    2. Поля: text (строка, без Markdown-кодовых блоков), опциональные tool_calls[], memory[], plan[].
    3. Соблюдай предоставленные persona, env_brief, style_hint, style_directive и инструкцию инструментов.
    4. Если инструментов не хватает, попроси пользователя добавить их, объясни зачем.
    """
)


def _format_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _context_block(payload: Dict[str, Any]) -> str:
    history = payload.get("history") or []
    persona = payload.get("persona") or {}
    env_brief = payload.get("env_brief") or {}
    style_hint = payload.get("style_hint") or {}
    style_directive = payload.get("style_directive") or ""
    junior_json = payload.get("junior_json") or {}
    jr_suggestions = payload.get("jr_suggestions") or []
    rag_hits = payload.get("rag_hits") or []
    tool_instructions = payload.get("tool_instructions") or {}

    sections = [
        "CONTEXT:",
        "History (latest last):\n" + _format_json(history),
        "Persona:\n" + _format_json(persona),
        "Environment:\n" + _format_json(env_brief),
        "Style hint:\n" + _format_json(style_hint),
        f"Style directive:\n{style_directive or '—'}",
        "Junior suggestions (top-3):\n" + _format_json(jr_suggestions),
        "Junior meta:\n" + _format_json(junior_json),
        "Tool instructions:\n" + _format_json(tool_instructions),
        "RAG hits:\n" + _format_json(rag_hits),
    ]
    return "\n\n".join(sections)


def _task_block(payload: Dict[str, Any]) -> str:
    last_user_text = payload.get("last_user_text") or payload.get("user_text") or ""
    last_user_json = json.dumps(last_user_text, ensure_ascii=False)
    requirements = dedent(
        f"""
        TASK:
        1. Ответь тепло и поддерживающе на последний пользовательский ввод (см. last_user_text).
        2. Минимум 40 слов в ответе.
        3. Заверши ответ одним коротким встречным вопросом.
        4. Используй русский язык.

        last_user_text: {last_user_json}
        """
    ).strip()
    return requirements


def _build_prompt(payload: Dict[str, Any]) -> str:
    context = _context_block(payload)
    task = _task_block(payload)
    output_rules = "OUTPUT: Верни только <json>{...}</json> по заданной схеме."
    return "\n\n".join([_SYS.strip(), context, task, output_rules])


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
    if not block or not block.strip():
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
    log.info("senior prompt preview: %s", prompt[:600].replace("\n", "\\n"))
    cfg = _CFG.get("senior", {})
    preset = payload.get("preset") or {}
    temp = preset.get("temperature")
    if not isinstance(temp, (int, float)):
        temp = cfg.get("temperature", 0.7)
    temp = max(0.05, float(temp))
    max_tokens = preset.get("max_tokens")
    if not isinstance(max_tokens, int):
        max_tokens = cfg.get("max_new_tokens", 512)
    raw = llm_generate(
        "senior",
        prompt,
        max_new_tokens=max_tokens,
        temperature=temp,
        top_p=0.92,
        repetition_penalty=1.08,
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
        try:
            data = _parse_reply(raw2)
        except Exception:
            data = {"text": "Извини, у меня сбой формата ответа."}
    try:
        validate(instance=data, schema=_SCHEMA)
    except ValidationError:
        # minimal recovery to keep service alive
        data = {"text": data["text"]} if isinstance(data, dict) and "text" in data else {"text": "Извини, у меня сбой формата ответа."}
    data = _maybe_refine_smalltalk(payload, data, max_tokens)
    # light sanity
    if "text" not in data:
        raise RuntimeError("Senior reply lacks 'text'")
    # Normalize tool_calls
    if "tool_calls" in data and not isinstance(data["tool_calls"], list):
        data["tool_calls"] = []
    return data


def _maybe_refine_smalltalk(
    payload: Dict[str, Any],
    data: Dict[str, Any],
    max_tokens: int,
) -> Dict[str, Any]:
    jr_intent = str((payload.get("junior_json") or {}).get("intent") or "").lower()
    text = str(data.get("text", ""))
    if jr_intent != "smalltalk" or len(text.split()) >= 20:
        return data

    draft_json = json.dumps({"text": text}, ensure_ascii=False)
    context = _context_block(payload)
    task = dedent(
        """
        TASK UPDATE:
        Ответ получился слишком коротким. Перепиши JSON так, чтобы:
        - поле "text" содержало минимум 45 слов;
        - тон остался тёплым и поддерживающим;
        - в конце оставался один короткий встречный вопрос.
        Верни только JSON между тегами <json> и </json>.
        """
    ).strip()
    refine_prompt = "\n\n".join(
        [
            _SYS.strip(),
            context,
            f"DRAFT:\n<json>{draft_json}</json>",
            task,
        ]
    )
    raw = llm_generate(
        "senior",
        refine_prompt,
        max_new_tokens=max_tokens,
        temperature=0.65,
        top_p=0.9,
        stop=["</json>"],
    )
    try:
        refined = _parse_reply(raw)
        validate(instance=refined, schema=_SCHEMA)
    except Exception:
        refined = data

    refined_text = str(refined.get("text", ""))
    if len(refined_text.split()) < 20:
        refined["text"] = (
            "Привет! Я рядом и с удовольствием поддержу тебя."
            " Расскажи, что сейчас происходит и какое настроение у тебя сегодня?"
        )
    return refined


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
