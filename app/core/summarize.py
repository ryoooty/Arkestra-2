from typing import List


def summarize_day(chunks: List[str], max_tokens: int = 400) -> str:
    # naive summary: first N sentences; replace with LLM summarizer if needed
    text = " ".join(chunks)
    return text[:max_tokens*4]  # approx char budget
