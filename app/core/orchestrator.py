from typing import Dict, Any, List

from app.memory.db import insert_message, get_last_messages
from app.memory.db import get_tool_instructions
from app.core.env_state import ensure_env_session
from app.core import neuro
from app.core.router import search as rag_search
from app.core.tools_runner import run_all
from app.core.guard import soft_censor
from app.agents.junior import generate as jr_generate
from app.agents.senior import generate_structured as sr_generate


def handle_user(
    user_id: str,
    text: str,
    channel: str = "cli",
    chat_id: str = "local",
    participants: List[str] | None = None,
) -> Dict[str, Any]:
    participants = participants or [user_id]
    insert_message(user_id, "user", text)

    # 1) env
    ensure_env_session(channel, chat_id, participants)
    env_brief = {
        "channel": channel,
        "chat_id": chat_id,
        "participants": participants,
    }

    # fetch history tail
    hist_tail = get_last_messages(user_id, n=6)

    # 2) junior
    jr_payload = {
        "history_tail": hist_tail,
        "user_text": text,
        "neuro_snapshot": neuro.snapshot(),
        "env_brief": env_brief,
        "tools_catalog": [  # краткий каталог
            {"name": "note.create", "desc": "сохранить заметку"},
            {"name": "reminder.create", "desc": "создать напоминание"},
            {"name": "tg.message.send", "desc": "отправить сообщение в TG"},
            {"name": "messages.search_by_date", "desc": "найти сообщения по дате"},
            {"name": "alias.add", "desc": "добавить алиас"},
            {"name": "alias.set_primary", "desc": "сделать алиас основным"},
        ],
    }
    jr = jr_generate(jr_payload)  # ожидается JSON v2

    # 3) neuro preset
    if jr and "neuro_update" in jr and "levels" in jr["neuro_update"]:
        neuro.set_levels(jr["neuro_update"]["levels"])
    preset = neuro.bias_to_style()

    # 4) RAG
    rag_query = jr.get("rag_query", "") if jr else ""
    intent = jr.get("intent", "other") if jr else "other"
    rag_hits = rag_search(rag_query, intent)

    # 5) senior
    sr_payload = {
        "history": hist_tail,
        "user_text": text,
        "rag_hits": rag_hits,
        "junior_json": jr,
        "preset": preset,
        "style_directive": jr.get("style_directive", "") if jr else "",
        "env_brief": env_brief,
    }
    tool_instr = get_tool_instructions(jr.get("tools_hint", [])) if jr else {}
    sr_payload["tool_instructions"] = tool_instr
    reply = sr_generate(sr_payload)

    # 6) tools
    tool_calls = reply.get("tool_calls", []) if reply else []
    tool_results = run_all(tool_calls) if tool_calls else []

    # 7) guard
    final_text, hits = soft_censor(reply.get("text", "")) if reply else ("", {})
    assistant_msg_id = insert_message(user_id, "assistant", final_text)
    bandit_kind = None
    # fast bandit update if junior had suggestions
    try:
        if jr and jr.get("suggestions"):
            chosen = jr.get("chosen_suggestion") or {}
            bandit_kind = chosen.get("kind") or (tool_calls and "good" or "mischief") or "good"
    except Exception:
        bandit_kind = None

    return {
        "text": final_text,
        "assistant_msg_id": assistant_msg_id,
        "bandit_kind": bandit_kind,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "rag_hits": rag_hits,
        "preset": preset,
        "guard_hits": hits,
        "junior": jr,
    }
