from typing import Any, Dict, List

import tiktoken


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Offline or missing tokeniser assets: fall back to a simple heuristic.
            return max(1, len(text.split()))
    return len(enc.encode(text))


def count_struct(history: List[Dict[str, Any]] | None) -> int:
    if not history:
        return 0
    return sum(count_tokens(f"{m.get('role','')}:{m.get('text','')}") for m in history)
