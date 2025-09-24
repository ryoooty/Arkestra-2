from typing import List, Dict
import random

from app.memory.db import upsert_bandit, get_conn

EPS = 0.1


def _ctr(intent: str, kind: str) -> float:
    with get_conn() as c:
        cur = c.execute(
            "SELECT wins,plays FROM bandit_stats WHERE intent=? AND kind=?",
            (intent, kind),
        )
        row = cur.fetchone()
        if not row:
            return 0.5  # Bayes starter (1/2)
        w, p = float(row["wins"]), float(row["plays"])
        return w / (p if p > 0 else 1.0)


def pick(intent: str, suggestions: List[Dict]) -> Dict:
    if not suggestions:
        return {}
    if random.random() < EPS:
        return random.choice(suggestions)
    best = None
    best_score = -1.0
    for s in suggestions:
        kind = s.get("kind", "good")
        conf = float(s.get("confidence", 0.5))
        score = _ctr(intent, kind) * conf
        if score > best_score:
            best, best_score = s, score
    return best or suggestions[0]


def update(intent: str, kind: str, reward: int) -> None:
    upsert_bandit(
        intent,
        kind,
        wins_delta=1.0 if reward > 0 else 0.0,
        plays_delta=1.0,
    )
