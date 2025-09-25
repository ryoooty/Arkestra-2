from typing import Dict, Any, List
from pathlib import Path

from app.memory.db import insert_message, get_last_messages, set_message_meta
from app.memory.db import get_tool_instructions
from app.core.env_state import ensure_env_session, build_env_brief
from app.core import neuro
from app.core.router import search as rag_search
from app.core.tools_runner import run_all
from app.core.guard import soft_censor
from app.agents.junior import generate as jr_generate
from app.agents.senior import (
    generate_structured as sr_generate,
    refine_with_results as sr_refine,
)
from app.core.budget import trim as pack_budget
from app.core.tokens import count_struct, count_tokens
from app.core.logs import log, span
from app.core.bandit import pick as bandit_pick

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
    with span("junior"):
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
        try:
            jr = jr_generate(jr_payload)  # ожидается JSON v2
        except Exception:
            log.exception("junior failed; falling back to defaults")
            jr = {
                "intent": "ask",
                "tools_hint": [],
                "style_directive": "дружелюбно и коротко",
                "neuro_update": {"levels": neuro.snapshot()},
                "suggestions": [
                    {
                        "kind": "good",
                        "text": "Привет! Я рядом и с удовольствием помогу. Расскажи, что сейчас занимает твои мысли, и чем я могу поддержать тебя?",
                    },
                    {
                        "kind": "mischief",
                        "text": "Эй, давай сделаем сегодня что-то необычное и весёлое! О чём мечтаешь прямо сейчас, может, устроим маленькое приключение?",
                    },
                ],
            }

    bandit_intent = "task"
    bandit_chosen_kind: str | None = None
    raw_tools_req = jr.get("tools_request") if jr else None
    tools_req = [r for r in raw_tools_req if isinstance(r, dict)] if isinstance(raw_tools_req, list) else []
    if jr:
        bandit_intent = jr.get("intent") or bandit_intent
        suggestions = jr.get("suggestions")
        if isinstance(suggestions, list) and suggestions:
            chosen = jr.get("chosen_suggestion") if isinstance(jr.get("chosen_suggestion"), dict) else None
            if not chosen:
                try:
                    chosen = bandit_pick(bandit_intent, suggestions)
                except Exception:
                    chosen = None
                if chosen:
                    jr["chosen_suggestion"] = chosen
            if isinstance(chosen, dict):
                bandit_chosen_kind = chosen.get("kind")
            if bandit_chosen_kind is None:
                first = suggestions[0] if isinstance(suggestions[0], dict) else {}
                bandit_chosen_kind = first.get("kind", "good")
    catalog_names = {t["name"] for t in jr_payload["tools_catalog"]}
    missing = [
        r
        for r in tools_req
        if r.get("name") and r["name"] not in catalog_names
    ]
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
        rag_query = jr.get("rag_query", "") if jr else ""
        intent = jr.get("intent", "other") if jr else "other"
        rag_hits = rag_search(rag_query, intent)

    # 5) senior
    with span("senior"):
        packed = pack_budget(
            history=hist_tail,
            rag_hits=rag_hits,
            junior_meta=jr,
            max_tokens=preset_max_tokens,
        )
        hist_tokens = count_struct(packed["history"])
        rag_tokens = sum(
            count_tokens(hit.get("text", "")) for hit in packed["rag_hits"]
        )
        jr_tokens = count_tokens(str(packed["junior_meta"])) if packed["junior_meta"] else 0
        total_tokens = hist_tokens + rag_tokens + jr_tokens
        persona = neuro.persona_brief()
        suggestions = jr.get("suggestions") if isinstance(jr, dict) else None
        if isinstance(suggestions, list):
            limited_suggestions = []
            for candidate in suggestions:
                if not isinstance(candidate, dict):
                    continue
                text_value = str(candidate.get("text", "")).strip()
                if not text_value:
                    text_value = (
                        "Привет! Я здесь, чтобы поддержать тебя и помочь разобраться."
                        " Расскажи, что сейчас важно, и чем мне заняться в первую очередь?"
                    )
                elif len(text_value.split()) < 12:
                    text_value = (
                        text_value
                        + " Я рядом и готова поддержать. Что сейчас кажется самым важным?"
                    )
                limited_suggestions.append(
                    {
                        "kind": candidate.get("kind", "good"),
                        "text": text_value,
                    }
                )
                if len(limited_suggestions) >= 3:
                    break
        else:
            limited_suggestions = []

        sr_payload = {
            "history": packed["history"],
            "history_tail": hist_tail,
            "user_text": text,
            "last_user_text": text,
            "rag_hits": packed["rag_hits"],
            "junior_json": packed["junior_meta"],
            "jr_suggestions": limited_suggestions,
            "preset": preset,
            "style_directive": jr.get("style_directive", "") if jr else "",
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

        tool_instr = (
            get_tool_instructions(_normalise_tool_hints(jr.get("tools_hint")))
            if jr
            else {}
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
    bandit_kind = bandit_chosen_kind
    if not bandit_kind:
        bandit_kind = "good" if tool_calls else "mischief"
    # fast bandit update if junior had suggestions
    try:
        if jr and jr.get("suggestions"):
            chosen = jr.get("chosen_suggestion") or {}
            bandit_kind = chosen.get("kind") or (tool_calls and "good" or "mischief") or "good"
    except Exception:
        bandit_kind = None

    if not bandit_kind:
        bandit_kind = bandit_chosen_kind or ("good" if tool_calls else "mischief")
    if not bandit_kind:
        bandit_kind = "good"

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
