from typing import List, Dict, Any

import tiktoken


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            # Offline fallback approximating token count by characters.
            return len(text)


def count_struct(history: List[Dict[str, Any]] | None) -> int:
    if not history:
        return 0
    return sum(count_tokens(f"{m.get('role','')}:{m.get('text','')}") for m in history)
