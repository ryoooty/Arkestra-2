"""
env_state.py — активная память окружения (осознание где/с кем).
- env_sessions(channel, chat_id, participants[], last_seen)
- env_facts(env_id, key, value, importance)
- build_env_brief(env_id) -> короткая шапка для промпта обеих моделей
"""

from typing import Dict


def ensure_env_session(channel: str, chat_id: str) -> int:
    raise NotImplementedError


def build_env_brief(env_id: int) -> Dict:
    raise NotImplementedError
