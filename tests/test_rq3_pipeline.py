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


def make_spec(seed=42, n_cycles=5, **kw):
    return SequenceSpec(
        sequence_id="S01", policy_name="scorer", checkpoint="",
        personality_name="high_E",
        personality=Personality.from_traits(extraversion=1.0),
        world_path="world.json", n_cycles=n_cycles, seed=seed, **kw,
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


def test_validate_rejects_inconsistent_moved_and_alien_prob_keys(world):
    seq = generate_sequence(make_spec(), HandAuthoredScorer(), world)

    bad = json.loads(json.dumps(seq))
    bad["steps"][1]["moved"] = not bad["steps"][1]["moved"]
    with pytest.raises(ValueError, match="moved"):
        validate_sequence(bad, world)

    bad2 = json.loads(json.dumps(seq))
    bad2["steps"][0]["location_probs"] = {"atlantis": 1.0}
    with pytest.raises(ValueError, match="atlantis"):
        validate_sequence(bad2, world)

    bad3 = json.loads(json.dumps(seq))
    first = bad3["steps"][0]["action_probs"]
    k1, k2 = list(first)
    bad3["steps"][0]["action_probs"] = {k1: -0.5, k2: 1.5}
    with pytest.raises(ValueError, match="negative"):
        validate_sequence(bad3, world)


def test_selection_temperature_in_meta_and_reproducible(world):
    a = generate_sequence(make_spec(seed=9, selection_temperature=0.1),
                          HandAuthoredScorer(), world)
    b = generate_sequence(make_spec(seed=9, selection_temperature=0.1),
                          HandAuthoredScorer(), world)
    assert a["meta"]["selection_temperature"] == 0.1
    assert a["steps"] == b["steps"]
    validate_sequence(a, world)


def test_preview_is_one_line_per_sequence(world):
    seq = generate_sequence(make_spec(), HandAuthoredScorer(), world)
    line = format_preview(seq)
    assert line.startswith("S01")
    assert "scorer" in line and "high_E" in line
    assert line.count("/") >= 5                          # loc/act per cycle


# ----------------------------------------------------------------------- CLI --

import csv

import torch

from experiments.rq3.gen_sequences import run_config
from npc_policy.policies import build_architecture


def write_config(tmp_path, world_path, policies):
    cfg = {
        "out_dir": str(tmp_path / "out"),
        "worlds": [str(world_path)],
        "personalities": [
            {"name": "high_E", "ocean": {"extraversion": 1.0}},
            {"name": "low_E", "ocean": {"extraversion": -1.0}},
        ],
        "policies": policies,
        "n_cycles": 4,
        "seeds": [42, 43],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path, cfg


def test_run_config_end_to_end(tmp_path, world):
    world_path = tmp_path / "world.json"          # written by the fixture
    torch.manual_seed(0)
    model = build_architecture("simple")
    ckpt = tmp_path / "simple.pt"
    torch.save({"model": "simple", "state_dict": model.state_dict()}, ckpt)

    cfg_path, cfg = write_config(
        tmp_path, world_path,
        [{"name": "scorer"}, {"name": "simple_2b", "checkpoint": str(ckpt)}],
    )
    n = run_config(cfg_path, preview=False)
    # 1 world x 2 personalities x 2 policies x 2 seeds
    assert n == 8

    out = tmp_path / "out"
    files = sorted(f.name for f in out.glob("S*.json"))
    assert files == [f"S{i:03d}.json" for i in range(1, 9)]

    with open(out / "manifest.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 8
    assert rows[0]["sequence_id"] == "S001"
    assert rows[0]["selection_temperature"] == "1.0"   # default, recorded
    assert {r["policy"] for r in rows} == {"scorer", "simple_2b"}

    # every written file passes the export gate again on re-read
    for name in files:
        seq = json.loads((out / name).read_text(encoding="utf-8"))
        validate_sequence(seq, world)


def test_run_config_reproducible_steps(tmp_path, world):
    world_path = tmp_path / "world.json"
    cfg_path, _ = write_config(tmp_path, world_path, [{"name": "scorer"}])
    run_config(cfg_path, preview=False)
    first = {f.name: json.loads(f.read_text(encoding="utf-8"))["steps"]
             for f in (tmp_path / "out").glob("S*.json")}
    run_config(cfg_path, preview=False)               # overwrite in place
    second = {f.name: json.loads(f.read_text(encoding="utf-8"))["steps"]
              for f in (tmp_path / "out").glob("S*.json")}
    assert first == second


def test_run_config_removes_stale_outputs(tmp_path, world):
    world_path = tmp_path / "world.json"
    cfg_path, cfg = write_config(tmp_path, world_path, [{"name": "scorer"}])
    assert run_config(cfg_path, preview=False) == 4   # 2 personalities x 2 seeds
    cfg["seeds"] = [42]                               # shrink the cross
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    assert run_config(cfg_path, preview=False) == 2
    files = sorted(f.name for f in (tmp_path / "out").glob("S*.json"))
    assert files == ["S001.json", "S002.json"]        # S003/S004 cleaned up


def test_personality_name_lookup_from_roster_file(tmp_path, world):
    world_path = tmp_path / "world.json"
    roster = {"personalities": [
        {"name": "Karlach", "ocean": {"extraversion": 0.9, "agreeableness": 0.6}},
    ]}
    roster_path = tmp_path / "personalities.json"
    roster_path.write_text(json.dumps(roster), encoding="utf-8")

    cfg_path, cfg = write_config(tmp_path, world_path, [{"name": "scorer"}])
    cfg["personalities"] = [{"name": "Karlach"}]      # name only -> roster lookup
    cfg["personalities_file"] = str(roster_path)
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    assert run_config(cfg_path, preview=False) == 2   # 1 personality x 2 seeds

    seq = json.loads((tmp_path / "out" / "S001.json").read_text(encoding="utf-8"))
    assert seq["meta"]["personality_name"] == "Karlach"
    assert seq["meta"]["ocean"] == {"O": 0.0, "C": 0.0, "E": 0.9, "A": 0.6, "N": 0.0}

    cfg["personalities"] = [{"name": "Nobody"}]
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="Nobody"):
        run_config(cfg_path, preview=False)


def test_unity_dir_sync_mirrors_and_cleans(tmp_path, world):
    world_path = tmp_path / "world.json"
    cfg_path, cfg = write_config(tmp_path, world_path, [{"name": "scorer"}])
    unity_dir = tmp_path / "unity_seq"
    cfg["unity_dir"] = str(unity_dir)
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    assert run_config(cfg_path, preview=False) == 4
    assert sorted(f.name for f in unity_dir.glob("S*.json")) == \
        [f"S{i:03d}.json" for i in range(1, 5)]
    # mirrored bytes identical to the archive copy
    assert (unity_dir / "S001.json").read_bytes() == \
        (tmp_path / "out" / "S001.json").read_bytes()

    cfg["seeds"] = [42]                               # shrink -> stale cleanup
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    assert run_config(cfg_path, preview=False) == 2
    assert sorted(f.name for f in unity_dir.glob("S*.json")) == \
        ["S001.json", "S002.json"]
