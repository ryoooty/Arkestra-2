"""
tools_runner.py — исполняет tool_calls senior'а.
- Загружает entrypoint из БД/конфига (tools table)
- importlib и вызов .main(args) → dict result
- (опц.) второй короткий прогон senior с tool_results
"""

from typing import List, Dict


def run_all(tool_calls: List[Dict]) -> List[Dict]:
    raise NotImplementedError
