"""
Active environment awareness helpers.
"""

import json
from typing import Dict
from app.memory.db import upsert_env_session, get_env_facts


def ensure_env_session(channel: str, chat_id: str, participants: list[str]) -> int:
    participants_json = json.dumps(participants, ensure_ascii=False)
    return upsert_env_session(channel, chat_id, participants_json)


def build_env_brief(env_id: int, channel: str, chat_id: str) -> Dict:
    facts = get_env_facts(env_id)
    top_facts = facts[:5]
    return {
        "channel": channel,
        "chat_id": chat_id,
        "participants_facts": top_facts
    }
