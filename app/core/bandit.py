"""
bandit.py — глобальный ε-greedy выбор направления:
- arms = (intent, suggestion.kind)
- ε-greedy (ε=0.1)
- Bayes-стартер (wins=1, plays=2)
- reward: 👍=+1, 👎=−1
- decay дневной: 0.995
- хранение в таблице bandit_stats
"""
