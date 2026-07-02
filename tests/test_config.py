"""v1.1 config: lambda_C replaces lambda_E; epsilon removed; base_form switch."""
import pytest

from npc_policy.config import LevelParams, ScorerConfig


def test_level_params_v11_fields():
    prm = LevelParams()
    assert prm.lambda_C == 1.0
    assert not hasattr(prm, "lambda_E")


def test_epsilon_removed():
    assert not hasattr(ScorerConfig(), "epsilon")


def test_base_form_default_and_validation():
    assert ScorerConfig().base_form == "ideal_point"
    assert ScorerConfig(base_form="bilinear").base_form == "bilinear"
    with pytest.raises(ValueError):
        ScorerConfig(base_form="quadratic")
