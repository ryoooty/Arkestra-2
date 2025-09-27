"""
junior.py — DeepSeek (≤3B). Роль: диспетчер и регулятор.
ВХОД: history_tail, user_text, neuro_snapshot, env_brief, tools_catalog (название+краткое назначение).
ВЫХОД: два текстовых блока <ctrl> и <advice> без JSON.
<ctrl> — строки key=value (intent, tools, rag_query, нейроуровни).
<advice> — короткая подсказка для senior о стиле и содержании ответа.
Junior НЕ пишет финальный ответ пользователю и НЕ знает схем аргументов инструментов.
"""

from typing import Dict, Any
import re
from textwrap import dedent

from app.core.llm import generate as llm_generate

_SYS = dedent(
    """
    You are JUNIOR for Arkestra.
    Always answer with exactly two plain text blocks without JSON or Markdown fences.

    <ctrl>
    intent=smalltalk
    tools=note.create,reminder.create
    rag_query=
    dopamine=0; serotonin=0; norepinephrine=0; acetylcholine=0
    </ctrl>

    <advice>
    Friendly short note for Senior about tone, structure, key points to cover.
    </advice>

    Rules:
    - <ctrl> block must list key=value pairs, one per line, lowercase keys.
    - intent: classify the user request (smalltalk/task/ask/other...)
    - tools: comma-separated tool names from catalog (no descriptions). Leave empty if none.
    - rag_query: short search query or leave empty.
    - Provide dopamine, serotonin, norepinephrine, acetylcholine levels as integers -2..+2 ("+1", "0", "-1").
    - If a level is unchanged use 0. Separate multiple neuro pairs with semicolons in a single line.
    - <advice> is free-form text (1-3 sentences) with tips for Senior (tone, content, follow-up question ideas).
    - Do NOT return JSON, Markdown, or any other text outside the two blocks.
    - Never write the final assistant reply for the user.
    """
).strip()


_NEURO_KEYS = ("dopamine", "serotonin", "norepinephrine", "acetylcholine")


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
        "Respond now with <ctrl> and <advice> blocks only."
    )


def _run_model(prompt: str, temperature: float, max_new_tokens: int, repeat_penalty: float) -> str:
    return llm_generate(
        "junior",
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repeat_penalty=repeat_penalty,
    )


def _sanitize_int(value: str) -> int | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    match = re.search(r"[-+]?\d+", cleaned)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _parse_neuro_pairs(text: str) -> Dict[str, int]:
    levels: Dict[str, int] = {}
    if not text:
        return levels
    parts = [segment.strip() for segment in text.split(";") if segment.strip()]
    for part in parts:
        if "=" in part:
            key, raw = part.split("=", 1)
        else:
            continue
        key = key.strip().lower()
        if key not in _NEURO_KEYS:
            continue
        value = _sanitize_int(raw)
        if value is None:
            continue
        levels[key] = value
    return levels


def parse_neuro(ctrl: Dict[str, str]) -> Dict[str, Dict[str, int]]:
    levels: Dict[str, int] = {}
    for key in _NEURO_KEYS:
        if key in ctrl:
            parsed = _parse_neuro_pairs(f"{key}={ctrl[key]}")
            levels.update(parsed)
    for alias in ("neuro", "neuro_levels", "neuro_update", "levels"):
        if alias in ctrl:
            levels.update(_parse_neuro_pairs(ctrl[alias]))
    cleaned = {k: levels.get(k, 0) for k in _NEURO_KEYS if k in levels}
    if not cleaned:
        return {}
    return {"levels": cleaned}


def parse_junior(text: str) -> Dict[str, Any]:
    ctrl: Dict[str, str] = {}
    advice = ""
    if text:
        m1 = re.search(r"<ctrl>(.+?)</ctrl>", text, re.S | re.I)
        if m1:
            for line in m1.group(1).splitlines():
                line = line.strip()
                if not line or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                ctrl[k.strip().lower()] = v.strip()
        m2 = re.search(r"<advice>(.+?)</advice>", text, re.S | re.I)
        if m2:
            advice = m2.group(1).strip()

    intent = ctrl.get("intent", "other") or "other"
    tools_raw = ctrl.get("tools", "")
    tools = [s.strip() for s in tools_raw.split(",") if s.strip()]
    rag_query = ctrl.get("rag_query") or None
    neuro = parse_neuro(ctrl)
    return {
        "intent": intent,
        "tools_request": tools,
        "rag_query": rag_query,
        "neuro_update": neuro or {},
        "advice_text": advice,
    }


def generate(payload: Dict[str, Any], **kwargs) -> str:
    prompt = _build_prompt(payload)
    requested_tokens = kwargs.get("max_new_tokens")
    if isinstance(requested_tokens, int):
        max_new_tokens = max(160, requested_tokens)
    else:
        max_new_tokens = 160

    base_temperature = kwargs.get("temperature", 0.2)
    repeat_penalty = kwargs.get("repeat_penalty", 1.1)

    raw = _run_model(prompt, float(base_temperature), max_new_tokens, repeat_penalty).strip()
    if "<ctrl" not in raw or "<advice" not in raw:
        retry = _run_model(prompt, 0.1, max_new_tokens, repeat_penalty).strip()
        if retry:
            raw = retry
    return raw
