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


def generate_structured(payload: Dict[str, Any]) -> Dict[str, Any]:
    raise NotImplementedError
