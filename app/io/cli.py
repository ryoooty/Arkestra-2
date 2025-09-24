"""
cli.py — консольный интерфейс:
- ввод текста
- показ ответа
- фидбек hotkeys: ↑(like) ↓(dislike) e(ввести текст)
"""
from __future__ import annotations

import sys
from typing import Iterable

from app.core.orchestrator import handle_user
from scripts.consolidate_sleep import run_sleep_batch

_HOTKEY_PROMPT = "Feedback [↑ like, ↓ dislike, e edit, Enter skip]: "
_ESCAPE_PREFIX = "\x1b"
_ARROW_UP = f"{_ESCAPE_PREFIX}[A"
_ARROW_DOWN = f"{_ESCAPE_PREFIX}[B"
_WIN_ARROW_UP = {"\x00H", "\xe0H"}
_WIN_ARROW_DOWN = {"\x00P", "\xe0P"}
_LIKE_KEYS = {_ARROW_UP, "u", "U"} | _WIN_ARROW_UP
_DISLIKE_KEYS = {_ARROW_DOWN, "d", "D"} | _WIN_ARROW_DOWN


def main() -> None:
    """Run a minimal REPL for local testing."""
    user_id = "local-user"
    print("Arkestra CLI. Type 'quit' to exit.")

    while True:
        try:
            text = input(">> ")
        except EOFError:
            print()
            break

        if text.strip().lower() in {"quit", "exit"}:
            break

        if text.strip() == "/sleep":
            run_sleep_batch()
            print("💤 sleep batch done.")
            continue

        result = handle_user(user_id, text, channel="cli", chat_id="local") or {}
        _display_result(result)
        _handle_feedback()


def _display_result(result: dict[str, object]) -> None:
    text = result.get("text", "")
    if text:
        print("Arkestra:", text)
    else:
        print("Arkestra: (no text)")

    tool_calls = result.get("tool_calls")
    if tool_calls:
        print("Tools:", tool_calls)

    rag_hits = result.get("rag_hits")
    if rag_hits:
        snippets = []
        for hit in _ensure_iterable(rag_hits):
            snippet = getattr(hit, "get", None)
            if callable(snippet):
                snippets.append(hit.get("text", ""))
            else:
                snippets.append(str(hit))
        if snippets:
            print("RAG:", snippets)


def _handle_feedback() -> None:
    key = _read_hotkey()
    if key == "like":
        print("[feedback] like")
    elif key == "dislike":
        print("[feedback] dislike")
    elif key == "edit":
        print("[feedback] edit → continue typing")


def _read_hotkey() -> str | None:
    print(_HOTKEY_PROMPT, end="", flush=True)
    key = _read_keypress()
    print()
    if key in _LIKE_KEYS:
        return "like"
    if key in _DISLIKE_KEYS:
        return "dislike"
    if key in {"e", "E"}:
        return "edit"
    return None


def _read_keypress() -> str:
    reader = _windows_keypress if sys.platform == "win32" else _posix_keypress
    try:
        return reader()
    except Exception:  # pragma: no cover - fallback for rare terminals
        return sys.stdin.readline().rstrip("\n")


def _posix_keypress() -> str:
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        first = sys.stdin.read(1)
        if first != _ESCAPE_PREFIX:
            return first
        second = sys.stdin.read(1)
        if second != "[":
            return first + second
        third = sys.stdin.read(1)
        return first + second + third
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _windows_keypress() -> str:
    import msvcrt

    ch = msvcrt.getwch()
    if ch in {"\x00", "\xe0"}:
        second = msvcrt.getwch()
        return ch + second
    return ch


def _ensure_iterable(value: object) -> Iterable:
    if isinstance(value, (list, tuple, set)):
        return value  # type: ignore[return-value]
    return [value]


if __name__ == "__main__":
    main()
