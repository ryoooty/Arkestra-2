from app.core.orchestrator import handle_user
from app.core.reminders import init_scheduler
from app.core.bandit import update as bandit_update
from app.memory.db import (
    add_feedback,
    get_message_meta,
    last_assistant_msg_id,
    mark_approved,
)
from scripts import migrate as migrate_script
from scripts.consolidate_sleep import run_sleep_batch


HELP_TEXT = (
    "Commands: /help /quit /sleep /up /down /fb <text> /ok"
)


def _print_help() -> None:
    print(HELP_TEXT)


def main():
    migrate_script.migrate()
    init_scheduler()

    user_id = "local-user"
    print("Arkestra CLI. Type '/help' for available commands.")

    while True:
        try:
            text = input(">> ")
        except EOFError:
            break

        t = text.strip()
        if t in {"/quit", "quit", "exit"}:
            break

        if t in {"/help", "help"}:
            _print_help()
            continue

        if t == "/up":
            mid = last_assistant_msg_id(user_id)
            if mid:
                add_feedback(mid, "up", None)
                print("👍 recorded")
                meta = get_message_meta(mid)
                intent = meta.get("last_intent", "task")
                kind = meta.get("last_kind", "good")
                bandit_update(intent, kind, +1)
            continue
        if t == "/down":
            mid = last_assistant_msg_id(user_id)
            if mid:
                add_feedback(mid, "down", None)
                print("👎 recorded")
                meta = get_message_meta(mid)
                intent = meta.get("last_intent", "task")
                kind = meta.get("last_kind", "good")
                bandit_update(intent, kind, -1)
            continue
        if t.startswith("/fb "):
            mid = last_assistant_msg_id(user_id)
            if mid:
                add_feedback(mid, "text", t[4:])
                print("✍️ recorded")
            continue
        if t == "/ok":
            mid = last_assistant_msg_id(user_id)
            if mid:
                mark_approved(mid, 1)
                print("✅ approved")
            continue
        if t == "/sleep":
            run_sleep_batch()
            print("💤 sleep batch done.")
            continue

        result = handle_user(user_id, t, channel="cli", chat_id="local") or {}
        print("Arkestra:", result.get("text", ""))
        if result.get("tool_calls"):
            print("Tools:", result["tool_calls"])
        if result.get("rag_hits"):
            print("RAG:", [h.get("text", "") for h in result["rag_hits"]])


if __name__ == "__main__":
    main()
