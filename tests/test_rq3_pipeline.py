"""RQ3 sequence pipeline: generation core + CLI
(plan: docs/plans/2026-07-17-rq3-sequence-pipeline.md)."""

import json

import numpy as np
import pytest

from experiments.rq3.common import (
    SequenceSpec, format_preview, generate_sequence, validate_sequence,
)
from npc_policy.representation import Personality
from npc_policy.scorer import HandAuthoredScorer
from npc_policy.world import load_world

TINY_WORLD = {
    "locations": [
        {"id": "tavern",
         "features": {"social": 0.9, "stimulation": 0.7},
         "actions": [{"id": "chat", "features": {"social": 0.9}},
                     {"id": "drink", "features": {"stimulation": 0.6}}]},
        {"id": "library",
         "features": {"cognitive": 0.9, "privacy": 0.8},
         "actions": [{"id": "read", "features": {"cognitive": 0.9}},
                     {"id": "study", "features": {"cognitive": 0.8, "structure": 0.6}}]},
    ]
}


@pytest.fixture
def world(tmp_path):
    path = tmp_path / "world.json"
    path.write_text(json.dumps(TINY_WORLD), encoding="utf-8")
    return load_world(path)


def make_spec(seed=42, n_cycles=5):
    return SequenceSpec(
        sequence_id="S01", policy_name="scorer", checkpoint="",
        personality_name="high_E",
        personality=Personality.from_traits(extraversion=1.0),
        world_path="world.json", n_cycles=n_cycles, seed=seed,
    )


def test_sequence_structure_and_moved_flags(world):
    seq = generate_sequence(make_spec(), HandAuthoredScorer(), world)
    assert seq["meta"]["sequence_id"] == "S01"
    assert seq["meta"]["ocean"] == {"O": 0.0, "C": 0.0, "E": 1.0, "A": 0.0, "N": 0.0}
    steps = seq["steps"]
    assert [s["cycle"] for s in steps] == [1, 2, 3, 4, 5]
    assert steps[0]["moved"] is True                     # first step always walks
    for prev, cur in zip(steps, steps[1:]):
        assert cur["moved"] == (cur["location"] != prev["location"])
    for s in steps:
        assert abs(sum(s["location_probs"].values()) - 1.0) < 1e-6
        assert abs(sum(s["action_probs"].values()) - 1.0) < 1e-6
        assert s["action"] in s["action_probs"]


def test_same_seed_reproduces_exactly(world):
    a = generate_sequence(make_spec(seed=7), HandAuthoredScorer(), world)
    b = generate_sequence(make_spec(seed=7), HandAuthoredScorer(), world)
    assert a["steps"] == b["steps"]


def test_different_seed_may_differ_and_validates(world):
    seq = generate_sequence(make_spec(seed=8, n_cycles=8), HandAuthoredScorer(), world)
    validate_sequence(seq, world)                        # should not raise


def test_validate_rejects_unknown_ids_and_bad_probs(world):
    seq = generate_sequence(make_spec(), HandAuthoredScorer(), world)
    bad = json.loads(json.dumps(seq))                    # deep copy
    bad["steps"][2]["location"] = "arena"
    with pytest.raises(ValueError, match="arena"):
        validate_sequence(bad, world)

    bad2 = json.loads(json.dumps(seq))
    bad2["steps"][0]["action_probs"] = {"chat": 0.4, "drink": 0.4}
    with pytest.raises(ValueError, match="sum"):
        validate_sequence(bad2, world)


def test_preview_is_one_line_per_sequence(world):
    seq = generate_sequence(make_spec(), HandAuthoredScorer(), world)
    line = format_preview(seq)
    assert line.startswith("S01")
    assert "scorer" in line and "high_E" in line
    assert line.count("/") >= 5                          # loc/act per cycle
