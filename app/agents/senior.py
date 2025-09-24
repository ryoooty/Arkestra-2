"""
senior.py — старшая модель (GPU):
- принимает user_text + history + rag_hits + junior_json + preset
- выдаёт финальный ответ text
- формирует tool_calls
- формирует memory[] (факты, заметки)
- планирует plan[]
"""
