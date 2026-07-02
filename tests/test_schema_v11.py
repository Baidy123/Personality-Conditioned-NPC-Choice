"""v1.1 split: intensity features (ideal-point) vs relational features (linear)."""
from npc_policy.schema import (
    ACTION_TAGS,
    INTENSITY_TAGS,
    LOCATION_TAGS,
    RELATIONAL_TAGS,
    n_intensity,
)


def test_intensity_tags():
    assert INTENSITY_TAGS["location"] == LOCATION_TAGS          # all 9 (v1.2)
    assert INTENSITY_TAGS["action"] == ACTION_TAGS[:7]          # shared 7


def test_relational_tags():
    assert RELATIONAL_TAGS == ("cooperation", "helping", "conflict", "control")


def test_native_action_layout():
    # guarantee used by the scorer: intensity block first, relational block last
    assert ACTION_TAGS == INTENSITY_TAGS["action"] + RELATIONAL_TAGS


def test_n_intensity():
    assert n_intensity("location") == 9    # v1.2: + location-level conflict
    assert n_intensity("action") == 7


def test_location_conflict_reuses_model_slot():
    # the 9th location feature must land in the existing 12-dim conflict slot
    from npc_policy.representation import Option
    from npc_policy.schema import model_index
    loc = Option.location("pit", conflict=0.8)
    assert loc.to_padded12()[model_index("conflict")] == 0.8
