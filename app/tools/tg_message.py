"""TG tool for sending personal messages.

Args:
    to: "self" (default), username, or chat_id.
    text: message body (required).
"""

from pathlib import Path
from typing import Dict

import yaml

try:  # optional dependency
    from telegram import Bot
except Exception:  # pragma: no cover - runtime safeguard
    Bot = None  # type: ignore[assignment]

_cfg = None
_tools_cfg = None


def _load_tools_cfg() -> Dict:
    global _tools_cfg
    if _tools_cfg is None:
        text = Path("config/tools.yaml").read_text(encoding="utf-8")
        _tools_cfg = yaml.safe_load(text) or {}
    return _tools_cfg


def _telegram_enabled() -> bool:
    tools_cfg = _load_tools_cfg()
    features = tools_cfg.get("features") or {}
    return bool(features.get("telegram_enabled"))


def _load_cfg() -> Dict:
    global _cfg
    if _cfg is None:
        _cfg = yaml.safe_load(Path("config/tg.yaml").read_text(encoding="utf-8"))
    return _cfg


def main(args: Dict) -> Dict:
    if not _telegram_enabled():
        return {"ok": False, "error": "telegram_disabled"}

    cfg = _load_cfg()
    token = cfg.get("bot_token")
    if not token or Bot is None:
        return {"ok": False, "error": "telegram bot not configured"}

    to = args.get("to", "self")
    text = (args.get("text") or "").strip()
    if not text:
        return {"ok": False, "error": "text required"}

    chat_id = cfg.get("self_chat_id") if to == "self" else to
    try:
        bot = Bot(token=token)
        bot.send_message(chat_id=chat_id, text=text)
        return {"ok": True, "to": to}
    except Exception as exc:  # pragma: no cover - runtime safeguard
        return {"ok": False, "error": str(exc)}
