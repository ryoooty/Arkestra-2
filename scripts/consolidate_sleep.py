"""
consolidate_sleep.py — авто/ручной «сон»:
- обрабатывает ВСЁ неразобранное с прошлого сна (по watermark из sleep_batches)
- группирует по дням; строит day_summaries и curated_facts (temp)
- TTL temp → агрегирует в long; инкрементально обновляет RAG индексы
- neuro.sleep_reset(); запускает SFT (senior) и микро-LoRA (junior)
Идемпотентность: consolidated_batch_id на обработанных, sleep_batches.status.
"""

def run_sleep_batch():
    raise NotImplementedError
