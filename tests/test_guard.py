from app.core.guard import soft_censor


def test_soft_censor_masks_profanity_and_pii():
    t = "Это бляд и email test@example.com и номер +7 999 123 45 67"
    out, hits = soft_censor(t)
    assert "***" in out
    assert "[email скрыт]" in out
    assert "[номер скрыт]" in out
    assert hits["profanity"] >= 1
    assert hits["pii"] >= 2
