"""
router.py — маршрутизация RAG:
- решает какие индексы использовать (short/temp/long/jokes)
- фильтры по дате/TTL из config/router.yaml
- возвращает top-k документов для контекста
"""
