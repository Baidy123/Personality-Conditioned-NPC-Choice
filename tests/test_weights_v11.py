"""v1.1 tables: shapes, ranges, and W_rel seeding from the fallback W^A."""
import numpy as np

from npc_policy.weights import (
    default_b_action,
    default_b_location,
    default_C_action,
    default_C_location,
    default_w_action,
    default_w_location,
    default_W_action,
    default_W_rel_action,
)


def test_shapes():
    assert default_b_location().shape == (9,)      # v1.2: + conflict column
    assert default_C_location().shape == (5, 9)
    assert default_w_location().shape == (9,)
    assert default_b_action().shape == (7,)
    assert default_C_action().shape == (5, 7)
    assert default_w_action().shape == (7,)
    assert default_W_rel_action().shape == (5, 4)


def test_ranges():
    for b in (default_b_location(), default_b_action()):
        assert np.all(b >= 0.0) and np.all(b <= 1.0)
    for w in (default_w_location(), default_w_action()):
        assert np.all(w > 0.0)


def test_W_rel_seeded_from_fallback_W_action():
    # spec §3: W_rel starts as the last four columns of the v1 W^A
    assert np.array_equal(default_W_rel_action(), default_W_action()[:, 7:])
