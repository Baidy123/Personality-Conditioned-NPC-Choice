"""Live demo session: config parsing, world mutation, stepping, forcing
(plan: docs/plans/2026-07-18-live-demo-mode.md)."""

import json

import numpy as np
import pytest

from experiments.rq3.live_session import LiveSession
from npc_policy.controller import DecisionController
from npc_policy.representation import Personality
from npc_policy.scorer import HandAuthoredScorer
from npc_policy.world import load_world

LIVE_WORLD = {
    "locations": [
        {"id": "tavern",
         "features": {"social": 0.9, "stimulation": 0.7},
         "actions": [{"id": "chat", "features": {"social": 0.9}},
                     {"id": "drink", "features": {"stimulation": 0.6}}],
         "events": [
             {"name": "celebration", "active": False,
              "buff": {"social": 0.1}, "force_npc": None},
             {"name": "brawl_night", "active": False,
              "buff": {}, "force_npc": "tavern/drink"},
         ]},
        {"id": "library",
         "features": {"cognitive": 0.9, "privacy": 0.8},
         "actions": [{"id": "read", "features": {"cognitive": 0.9}},
                     {"id": "study", "features": {"cognitive": 0.8, "structure": 0.6}}]},
        {"id": "cave",
         "features": {"exploration": 0.9, "risk": 0.7},
         "unlocked": False,
         "actions": [{"id": "spelunk", "features": {"exploration": 0.9, "risk": 0.6}}],
         "events": [
             {"name": "expedition", "active": False,
              "buff": {}, "force_npc": "cave/spelunk"},
         ]},
    ]
}

PERSONALITIES = {
    "personalities": [
        {"name": "Alice", "ocean": {"extraversion": 1.0}},
        {"name": "Bob", "ocean": {"extraversion": -1.0, "openness": 0.5}},
    ]
}


@pytest.fixture
def config_path(tmp_path):
    world_path = tmp_path / "world.json"
    world_path.write_text(json.dumps(LIVE_WORLD), encoding="utf-8")
    pers_path = tmp_path / "personalities.json"
    pers_path.write_text(json.dumps(PERSONALITIES), encoding="utf-8")
    cfg = {
        "world": str(world_path),
        "personalities": str(pers_path),
        "port": 0,
        "mode": "sample",
        "seed": 42,
        "npcs": [
            {"name": "Alice", "policy": "scorer"},
            {"name": "Bob", "policy": "scorer"},
        ],
    }
    path = tmp_path / "live_config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def write_config(tmp_path, config_path, **overrides):
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg.update(overrides)
    path = tmp_path / "live_config_override.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


# -------------------------------------------------------------- construction --

def test_roster_and_initial_state(config_path):
    session = LiveSession.from_config(config_path)
    state = session.state()
    assert state["step_count"] == 0
    npcs = state["npcs"]
    assert [n["slot"] for n in npcs] == [0, 1]
    assert [n["name"] for n in npcs] == ["Alice", "Bob"]
    assert npcs[0]["policy"] == "scorer"
    # ocean order is [O, C, E, A, N]
    assert npcs[0]["ocean"] == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert npcs[1]["ocean"] == [0.5, 0.0, -1.0, 0.0, 0.0]
    assert all(n["location"] == "" for n in npcs)

    locs = {l["id"]: l for l in state["locations"]}
    assert set(locs) == {"tavern", "library", "cave"}
    assert locs["cave"]["unlocked"] is False
    events = {e["name"]: e for e in locs["tavern"]["events"]}
    assert events["celebration"]["active"] is False
    assert events["celebration"]["force_npc"] == ""
    assert events["brawl_night"]["force_npc"] == "tavern/drink"


def test_inline_ocean_overrides_lookup(tmp_path, config_path):
    path = write_config(tmp_path, config_path, npcs=[
        {"name": "Custom", "policy": "scorer",
         "ocean": {"neuroticism": 0.9}},
    ])
    session = LiveSession.from_config(path)
    assert session.state()["npcs"][0]["ocean"] == [0.0, 0.0, 0.0, 0.0, 0.9]


def test_unknown_personality_raises(tmp_path, config_path):
    path = write_config(tmp_path, config_path, npcs=[
        {"name": "Nobody", "policy": "scorer"},
    ])
    with pytest.raises(ValueError, match="Nobody"):
        LiveSession.from_config(path)


def test_learned_policy_requires_checkpoint(tmp_path, config_path):
    path = write_config(tmp_path, config_path, npcs=[
        {"name": "Alice", "policy": "nonlinear_2b"},
    ])
    with pytest.raises(ValueError, match="checkpoint"):
        LiveSession.from_config(path)


# ------------------------------------------------------------------ stepping --

def test_step_shape_and_moved_flags(config_path):
    session = LiveSession.from_config(config_path)
    world = load_world(json.loads(config_path.read_text(encoding="utf-8"))["world"])

    first = session.step()
    assert first["step_count"] == 1
    assert [s["slot"] for s in first["steps"]] == [0, 1]
    for s in first["steps"]:
        assert s["moved"] is True
        assert s["overridden"] is False
        # cave is locked: candidates are exactly tavern + library
        assert {p["id"] for p in s["location_probs"]} == {"tavern", "library"}
        action_ids = {a.id for a in world.actions_at(s["location"])}
        assert s["action"] in action_ids
        assert {p["id"] for p in s["action_probs"]} == action_ids
        assert abs(sum(p["p"] for p in s["location_probs"]) - 1.0) < 1e-9
        assert abs(sum(p["p"] for p in s["action_probs"]) - 1.0) < 1e-9

    second = session.step()
    assert second["step_count"] == 2
    for prev, cur in zip(first["steps"], second["steps"]):
        assert cur["moved"] == (cur["location"] != prev["location"])
    # state reflects current locations
    state = session.state()
    assert [n["location"] for n in state["npcs"]] == \
        [s["location"] for s in second["steps"]]


def test_deterministic_replay_and_reset(config_path):
    a = LiveSession.from_config(config_path)
    b = LiveSession.from_config(config_path)
    trail_a = [a.step() for _ in range(4)]
    trail_b = [b.step() for _ in range(4)]
    assert trail_a == trail_b

    a.reset()
    assert a.state()["step_count"] == 0
    assert all(n["location"] == "" for n in a.state()["npcs"])
    assert [a.step() for _ in range(4)] == trail_a


# -------------------------------------------------------------------- events --

def test_event_buff_applies_and_reverts(config_path):
    session = LiveSession.from_config(config_path)
    social = lambda: float(  # noqa: E731
        session.world.effective_location("tavern").features[0])
    base = social()
    session.set_event("tavern", "celebration", True)
    assert social() == pytest.approx(base + 0.1)
    assert session.state()["locations"][0]["events"][0]["active"] is True
    session.set_event("tavern", "celebration", False)
    assert social() == pytest.approx(base)


def test_unlock_expands_candidates_and_relocks(config_path):
    session = LiveSession.from_config(config_path)
    session.set_unlocked("cave", True)
    step = session.step()
    for s in step["steps"]:
        assert {p["id"] for p in s["location_probs"]} == {"tavern", "library", "cave"}
    session.set_unlocked("cave", False)
    step = session.step()
    for s in step["steps"]:
        assert {p["id"] for p in s["location_probs"]} == {"tavern", "library"}


def test_unknown_event_or_location_raises(config_path):
    session = LiveSession.from_config(config_path)
    with pytest.raises(ValueError, match="atlantis"):
        session.set_unlocked("atlantis", True)
    with pytest.raises(ValueError, match="eclipse"):
        session.set_event("tavern", "eclipse", True)
    with pytest.raises(ValueError, match="atlantis"):
        session.set_event("atlantis", "celebration", True)


# ------------------------------------------------------------------- forcing --

def test_force_event_overrides_every_npc(config_path):
    session = LiveSession.from_config(config_path)
    session.step()
    session.set_event("tavern", "brawl_night", True)
    forced = session.step()
    for s in forced["steps"]:
        assert (s["location"], s["action"]) == ("tavern", "drink")
        assert s["overridden"] is True
        assert s["location_probs"] == [] and s["action_probs"] == []
    assert all(n["location"] == "tavern" for n in session.state()["npcs"])

    session.set_event("tavern", "brawl_night", False)
    released = session.step()
    for s in released["steps"]:
        assert s["overridden"] is False
        assert s["moved"] == (s["location"] != "tavern")


def test_force_target_locked_raises(config_path):
    session = LiveSession.from_config(config_path)
    session.set_event("cave", "expedition", True)   # cave itself is locked
    with pytest.raises(ValueError, match="cave"):
        session.step()


# ------------------------------------------------------ runtime npc editing --

def test_set_personality_replaces_ocean_and_keeps_buffers(config_path):
    session = LiveSession.from_config(config_path)
    first = session.step()
    where = first["steps"][0]["location"]

    session.set_personality(0, {"extraversion": -1.0, "neuroticism": 0.8})
    state = session.state()
    assert state["npcs"][0]["ocean"] == [0.0, 0.0, -1.0, 0.0, 0.8]
    # buffers kept by default: the controller still remembers the location
    assert state["npcs"][0]["location"] == where

    session.set_personality(0, {"extraversion": 0.5}, reset_buffers=True)
    assert session.state()["npcs"][0]["location"] == ""
    nxt = session.step()
    assert nxt["steps"][0]["moved"] is True   # fresh history: first step walks


def test_set_personality_validation(config_path):
    session = LiveSession.from_config(config_path)
    with pytest.raises(ValueError, match="slot"):
        session.set_personality(9, {"extraversion": 0.5})
    with pytest.raises(ValueError, match="charisma"):
        session.set_personality(0, {"charisma": 0.5})
    with pytest.raises(ValueError):
        session.set_personality(0, {"extraversion": 2.0})   # out of [-1, 1]


def test_add_npc_appends_and_reset_restores_roster(config_path):
    session = LiveSession.from_config(config_path)
    session.add_npc({"name": "Zed", "ocean": {"openness": 1.0}})
    state = session.state()
    assert [n["name"] for n in state["npcs"]] == ["Alice", "Bob", "Zed"]
    assert state["npcs"][2]["slot"] == 2
    assert state["npcs"][2]["ocean"] == [1.0, 0.0, 0.0, 0.0, 0.0]

    step = session.step()
    assert [s["name"] for s in step["steps"]] == ["Alice", "Bob", "Zed"]

    session.reset()
    assert [n["name"] for n in session.state()["npcs"]] == ["Alice", "Bob"]


def test_add_npc_by_personality_lookup_and_errors(config_path):
    session = LiveSession.from_config(config_path)
    session.add_npc({"name": "Bob"})          # exists in personalities.json
    assert session.state()["npcs"][2]["ocean"] == [0.5, 0.0, -1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="Ghost"):
        session.add_npc({"name": "Ghost"})    # unknown, no inline ocean


def test_policy_catalog_and_global_switch(tmp_path, config_path):
    import torch
    from npc_policy.policies import build_architecture
    torch.manual_seed(0)
    ckpt = tmp_path / "simple.pt"
    torch.save({"model": "simple",
                "state_dict": build_architecture("simple").state_dict()}, ckpt)

    path = write_config(tmp_path, config_path, policies={
        "scorer": {},
        "student": {"checkpoint": str(ckpt)},
    })
    session = LiveSession.from_config(path)
    state = session.state()
    assert state["policies"] == ["scorer", "student"]
    assert state["active_policy"] == "scorer"

    session.step()
    where = session.state()["npcs"][0]["location"]

    session.set_policy("student")
    state = session.state()
    assert state["active_policy"] == "student"
    assert all(n["policy"] == "student" for n in state["npcs"])
    # checkpoint principle: history kept, switch shows at the next step
    assert state["npcs"][0]["location"] == where
    step = session.step()                      # student runs without error
    assert all(s["overridden"] is False for s in step["steps"])

    session.set_policy("scorer")
    assert session.state()["active_policy"] == "scorer"
    with pytest.raises(ValueError, match="ghost"):
        session.set_policy("ghost")

    session.set_policy("student")
    session.reset()                            # reset -> config assignments
    assert session.state()["active_policy"] == "scorer"


def test_mixed_roster_reports_mixed_active_policy(tmp_path, config_path):
    import torch
    from npc_policy.policies import build_architecture
    torch.manual_seed(0)
    ckpt = tmp_path / "simple.pt"
    torch.save({"model": "simple",
                "state_dict": build_architecture("simple").state_dict()}, ckpt)
    path = write_config(tmp_path, config_path, npcs=[
        {"name": "Alice", "policy": "scorer"},
        {"name": "Bob", "policy": "student", "checkpoint": str(ckpt)},
    ])
    session = LiveSession.from_config(path)
    state = session.state()
    assert state["active_policy"] == "mixed"
    assert "student" in state["policies"]      # ad-hoc entry joins the catalog
    session.set_policy("student")
    assert session.state()["active_policy"] == "student"


def test_step_is_atomic_when_one_npc_fails(config_path):
    session = LiveSession.from_config(config_path)
    session.step()
    before = {
        npc.slot: [o.id for o in npc.controller.H_L.recent_to_old()]
        for npc in session.npcs
    }
    count_before = session.step_count

    # second NPC's decision blows up mid-cycle
    def boom(*args, **kwargs):
        raise RuntimeError("policy exploded")
    session.npcs[1].controller.choose_location = boom

    with pytest.raises(RuntimeError, match="exploded"):
        session.step()

    # first NPC's already-committed buffers were rolled back
    after = {
        npc.slot: [o.id for o in npc.controller.H_L.recent_to_old()]
        for npc in session.npcs
    }
    assert after == before
    assert session.step_count == count_before


# ------------------------------------------- controller commit_forced (unit) --

def _options(world, loc_id, act_id):
    loc = world.effective_location(loc_id)
    act = next(a for a in world.actions_at(loc_id) if a.id == act_id)
    return loc, act


def test_commit_forced_matches_autonomous_buffers(tmp_path):
    world_path = tmp_path / "w.json"
    world_path.write_text(json.dumps(LIVE_WORLD), encoding="utf-8")
    world = load_world(world_path)
    loc, act = _options(world, "tavern", "drink")

    forced = DecisionController(HandAuthoredScorer())
    forced.commit_forced(loc, act)

    # autonomous controller pushed the same pair by hand
    reference = DecisionController(HandAuthoredScorer())
    reference.H_L.push(loc)
    reference.H_A.push(act)

    assert forced.current_location_id == "tavern"
    assert [o.id for o in forced.H_L.recent_to_old()] == \
        [o.id for o in reference.H_L.recent_to_old()]
    assert [o.id for o in forced.H_A.recent_to_old()] == \
        [o.id for o in reference.H_A.recent_to_old()]


def test_commit_forced_clears_action_buffer_on_location_change(tmp_path):
    world_path = tmp_path / "w.json"
    world_path.write_text(json.dumps(LIVE_WORLD), encoding="utf-8")
    world = load_world(world_path)
    tavern, drink = _options(world, "tavern", "drink")
    library, read = _options(world, "library", "read")

    ctrl = DecisionController(HandAuthoredScorer())
    ctrl.commit_forced(tavern, drink)
    ctrl.commit_forced(tavern, drink)
    assert len(ctrl.H_A) == 2               # same location: buffer accumulates
    ctrl.commit_forced(library, read)
    assert [o.id for o in ctrl.H_A.recent_to_old()] == ["read"]   # cleared on change
    assert ctrl.current_location_id == "library"


def test_after_force_next_autonomous_action_sees_history(tmp_path):
    world_path = tmp_path / "w.json"
    world_path.write_text(json.dumps(LIVE_WORLD), encoding="utf-8")
    world = load_world(world_path)
    tavern, drink = _options(world, "tavern", "drink")

    rng = np.random.default_rng(0)
    ctrl = DecisionController(HandAuthoredScorer(), mode="argmax", rng=rng)
    ctrl.commit_forced(tavern, drink)
    p = Personality.from_traits(extraversion=1.0)

    # the next action choice at the same location must run against a
    # NON-empty H_A (drink is in it); compare with a hand-built trace
    scorer = HandAuthoredScorer()
    expected = scorer.trace(p, world.actions_at("tavern"),
                            buffer=ctrl.H_A, level="action").P_rule
    decision = ctrl.choose_action(p, world.actions_at("tavern"))
    np.testing.assert_allclose(decision.distribution, expected, atol=1e-12)
