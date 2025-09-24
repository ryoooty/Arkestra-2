"""
orchestrator.py — главный поток обработки:
- принимает user_text
- вызывает junior → получает intent/tools_hint/suggestions
- дергает RAG через router
- применяет bandit и neuro.bias_to_style
- вызывает senior → получает ответ + tool_calls
- запускает tools_runner и guard
- пишет результаты в БД и логирует
"""

def handle_user(user_id, text):
    raise NotImplementedError("Main orchestration pipeline stub")
