from app.core import neuro


def test_neuro_set_levels_and_bias():
    snap = neuro.snapshot()
    neuro.set_levels({**snap, "dopamine": 11})
    preset = neuro.bias_to_style()
    assert "temperature" in preset
    assert preset["max_tokens"] >= 128
