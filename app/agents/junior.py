"""
junior.py — DeepSeek (≤3B). Роль: диспетчер и регулятор.
ВХОД: history_tail, user_text, neuro_snapshot, env_brief, tools_catalog (название+краткое назначение).
ВЫХОД (JSON v2):
- intent
- tools_hint[] (имена инструментов из каталога)
- tools_request[] (если не хватает инструмента)
- rag_query?
- style_directive (короткая подсказка, как окрасить ответ)
- neuro_update.levels (целевые уровни нейро)
!!! Junior НЕ пишет финальный ответ пользователю и НЕ знает схем аргументов инструментов.
"""

from typing import Dict, Any


def generate(payload: Dict[str, Any]) -> Dict[str, Any]:
    raise NotImplementedError
