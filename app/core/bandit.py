"""
bandit.py — глобальный ε-greedy (без персоналок).
- arms = (intent, suggestion.kind)
- ε=0.1; Bayes-стартер: wins=1, plays=2
- reward: up=+1, down=−1
- decay дневной: 0.995
- хранение: bandit_stats(intent, kind, wins, plays, updated_at)
"""

from typing import Tuple, List, Dict


def pick(intent: str, suggestions: List[Dict]) -> Dict:
    # выбрать suggestion по ε-greedy c учётом CTR для (intent, kind)
    raise NotImplementedError


def update(intent: str, kind: str, reward: int) -> None:
    raise NotImplementedError
