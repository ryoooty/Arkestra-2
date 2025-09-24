from app.core.bandit import pick, update


def test_bandit_pick_and_update():
    sug = [{"kind": "good", "confidence": 0.9}, {"kind": "mischief", "confidence": 0.2}]
    chosen = pick("task", sug)
    assert chosen["kind"] in ("good", "mischief")
    update("task", chosen["kind"], +1)
