"""
guard.py — мягкая цензура (последний шаг перед выводом ответа).
- мат → внутренняя маска '***' (оставляя первую/последнюю букву где безопасно)
- e-mail/телефоны → [скрыто]
"""

from typing import Tuple


def soft_censor(text: str) -> Tuple[str, dict]:
    raise NotImplementedError
