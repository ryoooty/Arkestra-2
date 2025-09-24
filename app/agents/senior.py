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
import json
from app.core.llm import generate as llm_generate

_SYS = """You are SENIOR for Arkestra (Mistral-7B).
You must produce a compact JSON with keys: text, tool_calls (optional array of {name,args}), memory (optional), plan (optional).
Use tool instructions exactly as given. 
User-facing text goes into "text". Do NOT include code fences. 
Keep helpful, follow style_directive, and constraints from preset and env_brief."""


def _build_prompt(payload: Dict[str, Any]) -> str:
    history = payload.get("history", [])
    user_text = payload.get("user_text", "")
    rag_hits = payload.get("rag_hits", [])
    junior_json = payload.get("junior_json", {})
    preset = payload.get("preset", {})
    style_directive = payload.get("style_directive", "")
    env_brief = payload.get("env_brief", {})
    tool_instructions = payload.get("tool_instructions", {})

    return (
        f"{_SYS}\n"
        f"ENV: {env_brief}\n"
        f"PRESET: {preset}\n"
        f"STYLE: {style_directive}\n"
        f"TOOLS_INSTRUCTIONS: {tool_instructions}\n"
        f"HISTORY: {history}\n"
        f"RAG_HITS: {rag_hits}\n"
        f"JUNIOR_JSON: {junior_json}\n"
        f"USER: {user_text}\n"
        f"Reply JSON with keys [text, tool_calls?, memory?, plan?]."
    )


def generate_structured(payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = _build_prompt(payload)
    raw = llm_generate("senior", prompt, max_new_tokens=512)
    # Extract JSON
    try:
        data = json.loads(raw)
    except Exception:
        start = raw.find("{"); end = raw.rfind("}")
        if start >= 0 and end >= 0 and end > start:
            data = json.loads(raw[start:end+1])
        else:
            raise RuntimeError(f"Senior returned non-JSON: {raw}")
    # light sanity
    if "text" not in data:
        raise RuntimeError("Senior reply lacks 'text'")
    # Normalize tool_calls
    if "tool_calls" in data and not isinstance(data["tool_calls"], list):
        data["tool_calls"] = []
    return data


def refine_reply(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback deterministic refinement step.

    Until a dedicated refine LLM prompt is introduced we simply attach tool
    outputs to the existing draft if necessary so that orchestrator logic
    remains uniform.
    """

    draft = payload.get("draft_reply") or {}
    if not isinstance(draft, dict):
        draft = {"text": str(draft)}
    tool_results = payload.get("tool_results") or []
    if not tool_results:
        return draft

    extra_chunks = []
    for item in tool_results:
        if not isinstance(item, dict):
            continue
        output = item.get("result") or item.get("output") or item.get("text")
        if isinstance(output, (dict, list)):
            output = json.dumps(output, ensure_ascii=False)
        if not output:
            continue
        extra_chunks.append(str(output))

    if not extra_chunks:
        return draft

    refined = dict(draft)
    base_text = refined.get("text", "")
    suffix = "\n\n" if base_text else ""
    refined["text"] = f"{base_text}{suffix}{'\n'.join(extra_chunks)}"
    return refined
