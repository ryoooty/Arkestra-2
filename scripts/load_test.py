import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.core.orchestrator import handle_user
from app.memory.db import migrate


def _one(i: int) -> float:
    user_id = 1
    texts = [
        "сохрани в заметки: небо красное",
        "сделай напоминание через 2 минуты с текстом 'попить воды'",
        "поищи, о чём мы говорили сегодня",
        "привет, как дела?",
    ]
    t = random.choice(texts)
    t0 = time.perf_counter()
    _ = handle_user(user_id, t, channel="cli", chat_id="local")
    return time.perf_counter() - t0


def main(concurrency: int = 8, total: int = 50) -> None:
    migrate()
    latencies = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_one, i) for i in range(total)]
        for future in as_completed(futures):
            latencies.append(future.result())

    if not latencies:
        print("No results")
        return

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95_index = max(int(len(latencies) * 0.95) - 1, 0)
    p95 = latencies[p95_index]
    print(
        "completed={}  p50={:.1f}ms  p95={:.1f}ms  max={:.1f}ms".format(
            len(latencies), p50 * 1000, p95 * 1000, latencies[-1] * 1000
        )
    )


if __name__ == "__main__":
    main()
