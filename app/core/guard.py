"""Модуль мягкой цензуры ответов."""

import re
from typing import Tuple, Dict

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"\+?\d[\d\-\s]{7,}\d")

_BAD = [
    r"\bх[уy][йi]\b",
    r"\bп[иi]зд",
    r"\bбл[яi]т",
    r"\bfuck\b",
    r"\bshit\b",
]


def _soften_word(m: re.Match) -> str:
    w = m.group(0)
    if len(w) <= 2:
        return "***"
    return w[0] + "***" + w[-1]


def soft_censor(text: str) -> Tuple[str, Dict]:
    hits = {"profanity": 0, "pii": 0}
    for pat in _BAD:
        regex = re.compile(pat, re.IGNORECASE | re.UNICODE)

        def repl(m):
            hits["profanity"] += 1
            return _soften_word(m)

        text = regex.sub(repl, text)
    if _EMAIL.search(text):
        hits["pii"] += 1
        text = _EMAIL.sub("[email скрыт]", text)
    if _PHONE.search(text):
        hits["pii"] += 1
        text = _PHONE.sub("[номер скрыт]", text)
    return text, hits
