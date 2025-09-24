"""
tools_runner.py — исполнение инструментов:
- получает tool_calls от senior
- загружает entrypoint из tools/
- вызывает и возвращает результаты
- опционально делает второй проход senior с tool_results
"""
