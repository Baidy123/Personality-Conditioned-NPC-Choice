"""v1.1 split: intensity features (ideal-point) vs relational features (linear)."""
from npc_policy.schema import (
    ACTION_TAGS,
    INTENSITY_TAGS,
    LOCATION_TAGS,
    RELATIONAL_TAGS,
    n_intensity,
)


def test_intensity_tags():
    assert INTENSITY_TAGS["location"] == LOCATION_TAGS          # all 8
    assert INTENSITY_TAGS["action"] == ACTION_TAGS[:7]          # shared 7


def test_relational_tags():
    assert RELATIONAL_TAGS == ("cooperation", "helping", "conflict", "control")


def test_native_action_layout():
    # guarantee used by the scorer: intensity block first, relational block last
    assert ACTION_TAGS == INTENSITY_TAGS["action"] + RELATIONAL_TAGS


def test_n_intensity():
    assert n_intensity("location") == 8
    assert n_intensity("action") == 7
