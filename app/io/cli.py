from app.core import bandit
from app.core.orchestrator import handle_user
from app.core.reminders import init_scheduler
from app.memory.db import add_feedback, last_assistant_msg_id, mark_approved
from scripts.consolidate_sleep import run_sleep_batch


def main():
    user_id = "local-user"
    init_scheduler()
    print("Arkestra CLI. Type 'quit' to exit. Commands: /up /down /fb <text> /ok")

    last_intent: str | None = None
    last_kind: str | None = None

    while True:
        try:
            text = input(">> ")
        except EOFError:
            break

        t = text.strip()
        if t in {"quit", "exit"}:
            break

        if t == "/up":
            mid = last_assistant_msg_id(user_id)
            if mid:
                add_feedback(mid, "up", None)
                print("👍 recorded")
                if last_intent and last_kind:
                    bandit.update(last_intent, last_kind, +1)
            continue
        if t == "/down":
            mid = last_assistant_msg_id(user_id)
            if mid:
                add_feedback(mid, "down", None)
                print("👎 recorded")
                if last_intent and last_kind:
                    bandit.update(last_intent, last_kind, -1)
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

        jr = result.get("junior") or {}
        tool_calls = result.get("tool_calls") or []
        last_intent = jr.get("intent") if isinstance(jr, dict) else None
        bandit_kind = result.get("bandit_kind")
        if isinstance(bandit_kind, str) and bandit_kind:
            last_kind = bandit_kind
        else:
            last_kind = None
            if isinstance(jr, dict):
                chosen = jr.get("chosen_suggestion") or {}
                if isinstance(chosen, dict) and chosen.get("kind"):
                    last_kind = chosen.get("kind")
                elif tool_calls:
                    last_kind = "good"
                else:
                    last_kind = "mischief"
            if not last_kind:
                last_kind = "good"


if __name__ == "__main__":
    main()
