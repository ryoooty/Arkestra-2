from typing import Dict, Any, List
from pathlib import Path

from app.memory.db import insert_message, get_last_messages, set_message_meta
from app.memory.db import get_tool_instructions
from app.core.env_state import ensure_env_session, build_env_brief
from app.core import neuro
from app.core.router import search as rag_search
from app.core.tools_runner import run_all
from app.core.guard import soft_censor
from app.agents.junior import generate as jr_generate, parse_junior
from app.agents.senior import (
    generate_structured as sr_generate,
    refine_with_results as sr_refine,
)
from app.core.budget import trim as pack_budget
from app.core.tokens import count_struct, count_tokens
from app.core.logs import log, span

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback
    yaml = None
    from app.util import simple_yaml

    def _load_yaml(path: Path) -> Dict[str, Any]:
        return simple_yaml.loads(path.read_text(encoding="utf-8"))
else:

    def _load_yaml(path: Path) -> Dict[str, Any]:
        return yaml.safe_load(path.read_text(encoding="utf-8"))


_LLM_CFG = _load_yaml(Path("config/llm.yaml"))
_SENIOR_MAX_NEW_TOKENS = _LLM_CFG.get("senior", {}).get("max_new_tokens", 512)
if not isinstance(_SENIOR_MAX_NEW_TOKENS, int) or _SENIOR_MAX_NEW_TOKENS <= 0:
    _SENIOR_MAX_NEW_TOKENS = 512

_HISTORY_WINDOW = 16


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
    with span("env"):
        env_id = ensure_env_session(channel, chat_id, participants)
        env_brief = build_env_brief(env_id, channel, chat_id)
        env_brief.setdefault("participants", participants)
        env_brief.setdefault("language", "ru")

    # fetch history tail
    hist_tail = get_last_messages(user_id, n=_HISTORY_WINDOW)

    # 2) junior
    jr_raw = ""
    jr_struct: Dict[str, Any] = {}
    with span("junior"):
        jr_payload = {
            "history_tail": hist_tail,
            "user_text": text,
            "neuro_snapshot": neuro.snapshot(),
            "env_brief": env_brief,
            "tools_catalog": [
                {"name": "note.create", "desc": "сохранить заметку"},
                {"name": "reminder.create", "desc": "создать напоминание"},
                {"name": "tg.message.send", "desc": "отправить сообщение в TG"},
                {"name": "messages.search_by_date", "desc": "найти сообщения по дате"},
                {"name": "alias.add", "desc": "добавить алиас"},
                {"name": "alias.set_primary", "desc": "сделать алиас основным"},
            ],
        }
        try:
            jr_raw = jr_generate(jr_payload)
            jr_struct = parse_junior(jr_raw)
        except Exception:
            log.exception("junior failed; falling back to defaults")
            jr_raw = ""
            jr_struct = {}

    jr_intent = str(jr_struct.get("intent", "other") if isinstance(jr_struct, dict) else "other")
    jr_tools = (
        [t for t in jr_struct.get("tools_request", []) if isinstance(t, str)]
        if isinstance(jr_struct, dict)
        else []
    )
    jr_rag_query = jr_struct.get("rag_query") if isinstance(jr_struct, dict) else None
    if isinstance(jr_rag_query, str):
        jr_rag_query = jr_rag_query.strip() or None
    else:
        jr_rag_query = None
    jr_neuro = jr_struct.get("neuro_update") if isinstance(jr_struct, dict) else {}
    if not isinstance(jr_neuro, dict):
        jr_neuro = {}
    jr_levels_raw = jr_neuro.get("levels") if isinstance(jr_neuro.get("levels"), dict) else {}
    jr_levels: Dict[str, int] = {}
    for key, value in (jr_levels_raw or {}).items():
        try:
            jr_levels[key] = int(value)
        except (TypeError, ValueError):
            continue
    jr_neuro = {"levels": jr_levels} if jr_levels else {}
    jr_advice = (
        str(jr_struct.get("advice_text", ""))
        if isinstance(jr_struct, dict)
        else ""
    )

    jr_ctrl = {
        "intent": jr_intent or "other",
        "tools_request": jr_tools,
        "rag_query": jr_rag_query,
        "neuro_update": jr_neuro,
    }
    jr = dict(jr_ctrl)
    jr["advice_text"] = jr_advice
    jr["raw"] = jr_raw

    bandit_intent = str(jr.get("intent") or "task")
    tools_req = jr_tools
    catalog_names = {t["name"] for t in jr_payload["tools_catalog"]}
    missing = [name for name in tools_req if name not in catalog_names]
    if missing:
        # превратить в мягкую просьбу пользователю через senior первичным текстом
        # senior сам вставит предложение "Хочешь, добавим инструмент X? Я подскажу спецификацию."
        pass

    # 3) neuro preset
    with span("neuro"):
        if jr and "neuro_update" in jr and "levels" in jr["neuro_update"]:
            neuro.set_levels(jr["neuro_update"]["levels"])
        raw_preset = neuro.bias_to_style()
        preset = dict(raw_preset) if isinstance(raw_preset, dict) else {}
        style_hint = {
            k: preset.get(k)
            for k in (
                "humor_bias",
                "structure_bias",
                "ask_clarify_bias",
                "brevity_bias",
                "warmth_bias",
                "positivity_bias",
                "alertness_bias",
            )
        }

        # Normalise generation overrides so that downstream callers always
        # receive sane values even if neuro returns junk or partial presets.
        preset_temperature = preset.get("temperature")
        if not isinstance(preset_temperature, (int, float)):
            preset_temperature = _LLM_CFG.get("senior", {}).get("temperature", 0.7)
        preset["temperature"] = float(preset_temperature)

        preset_max_tokens = preset.get("max_tokens")
        if not isinstance(preset_max_tokens, int) or preset_max_tokens <= 0:
            preset_max_tokens = _SENIOR_MAX_NEW_TOKENS
        preset["max_tokens"] = int(preset_max_tokens)

    # 4) RAG
    with span("rag"):
        rag_query = jr_ctrl.get("rag_query") or ""
        intent = jr_ctrl.get("intent") or "other"
        rag_hits = rag_search(rag_query, intent)

    # 5) senior
    with span("senior"):
        packed = pack_budget(
            history=hist_tail,
            rag_hits=rag_hits,
            junior_meta=jr_ctrl,
            max_tokens=preset_max_tokens,
        )
        hist_tokens = count_struct(packed["history"])
        rag_tokens = sum(
            count_tokens(hit.get("text", "")) for hit in packed["rag_hits"]
        )
        jr_tokens = count_tokens(str(packed["junior_meta"])) if packed["junior_meta"] else 0
        total_tokens = hist_tokens + rag_tokens + jr_tokens
        persona = neuro.persona_brief()

        sr_payload = {
            "history": packed["history"],
            "history_tail": hist_tail,
            "user_text": text,
            "last_user_text": text,
            "rag_hits": packed["rag_hits"],
            "jr_ctrl": packed["junior_meta"],
            "jr_advice": jr_advice,
            "preset": preset,
            "style_directive": "",
            "style_hint": style_hint,
            "env_brief": env_brief,
            "persona": persona,
        }
        log.info(
            "budget: hist=%s rag=%s jr=%s tokens=(hist=%s, rag=%s, jr=%s, total=%s) max_tokens=%s",
            len(packed["history"]),
            len(packed["rag_hits"]),
            bool(packed["junior_meta"]),
            hist_tokens,
            rag_tokens,
            jr_tokens,
            total_tokens,
            preset_max_tokens,
        )

        def _normalise_tool_hints(value: Any) -> list[str]:
            if isinstance(value, str):
                return [value]
            if isinstance(value, list):
                return [str(v) for v in value if isinstance(v, str)]
            return []

        tool_instr = get_tool_instructions(
            _normalise_tool_hints(jr_ctrl.get("tools_request"))
        )
        sr_payload["tool_instructions"] = tool_instr
        reply = sr_generate(sr_payload)

    # 6) tools
    with span("tools"):
        tool_calls = reply.get("tool_calls", []) if reply else []
        if tool_calls:
            try:
                tool_results = run_all(tool_calls)
            except Exception:
                log.exception("tools failed")
                tool_results = []
        else:
            tool_results = []

    with span("refine"):
        if tool_results:
            refine_payload = dict(sr_payload)
            refine_payload["tool_results"] = tool_results
            try:
                refined = sr_refine(refine_payload)
            except Exception:
                refined = None
            if refined and refined.get("text"):
                if reply is None:
                    reply = refined
                else:
                    reply["text"] = refined["text"]

    with span("guard"):
        base_text = reply.get("text", "") if reply else ""
        final_text, hits = soft_censor(base_text)
    assistant_msg_id = insert_message(user_id, "assistant", final_text)
    bandit_kind = "good" if tool_calls else "mischief"

    try:
        set_message_meta(assistant_msg_id, "last_intent", str(bandit_intent))
        if bandit_kind:
            set_message_meta(assistant_msg_id, "last_kind", str(bandit_kind))
    except Exception:
        log.exception("failed to store bandit metadata")

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
