"""
orchestrator.py — главный конвейер Arkestra.

Поток:
1) Собираем env-brief (осознание окружения).
2) Вызываем junior (DeepSeek ≤3B) → JSON v2: intent, tools_hint, tools_request?, rag_query, style_directive, neuro_update.levels.
3) Применяем neuro.set_levels(...) → preset = neuro.bias_to_style().
4) RAG search → rerank (e5-small) → trim to budget.
5) Вызываем senior (Mistral-7B) c: history, rag_hits, junior_json, preset, style_directive, tool INSTRUCTIONS.
6) Исполняем tool_calls; (опц.) второй короткий прогон senior с tool_results.
7) Guard (мат → с***, PII → [скрыто]) → отдать ответ.
8) Async: memory store, bandit.update, neuro-log, messages save.
"""

from typing import Dict, Any, List


def handle_user(user_id: str, text: str) -> Dict[str, Any]:
    # TODO: 1) read env 2) call junior 3) set neuro 4) router.search+rerank 5) call senior 6) tools 7) guard 8) async logs
    raise NotImplementedError("Main orchestration pipeline stub")
