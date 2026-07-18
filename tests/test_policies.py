"""Tests for npc_policy.policies (unified policy layer) and the generalised
DecisionController (plan: docs/plans/2026-07-17-rq3-sequence-pipeline.md)."""

import numpy as np
import pytest
import torch

from npc_policy.config import DEFAULT_CONFIG
from npc_policy.learned import predict_distribution
from npc_policy.policies import LearnedPolicyAdapter, build_architecture
from npc_policy.relations import compute_relations
from npc_policy.representation import Option, Personality, RecentBuffer

LOCS = [
    Option.location("tavern", social=0.9, stimulation=0.7),
    Option.location("library", cognitive=0.9, privacy=0.8),
    Option.location("forge", physical=0.8, structure=0.7),
]
ACTS = [
    Option.action("chat", social=0.9),
    Option.action("drink", stimulation=0.6),
]
P = Personality.from_traits(extraversion=0.8, openness=0.4)


def save_checkpoint(tmp_path, name, seed=0):
    torch.manual_seed(seed)
    model = build_architecture(name)
    path = tmp_path / f"{name}.pt"
    torch.save({"model": name, "state_dict": model.state_dict()}, path)
    return path, model


def test_build_architecture_unknown_name():
    with pytest.raises(ValueError, match="unknown architecture"):
        build_architecture("transformer")


def test_adapter_all_architectures_valid_distribution(tmp_path):
    for name in ("simple", "nonlinear", "agnostic_simple", "agnostic_nonlinear"):
        path, _ = save_checkpoint(tmp_path, name)
        adapter = LearnedPolicyAdapter(path)
        dist = adapter.distribution(P, LOCS, level="location")
        assert dist.shape == (3,)
        assert abs(dist.sum() - 1.0) < 1e-9
        assert (dist >= 0).all()


def test_adapter_matches_predict_distribution_with_buffer(tmp_path):
    path, model = save_checkpoint(tmp_path, "nonlinear")
    adapter = LearnedPolicyAdapter(path)
    buf = RecentBuffer(maxlen=3)
    buf.push(LOCS[0])
    buf.push(LOCS[1])
    got = adapter.distribution(P, LOCS, buffer=buf, level="location")
    rel = compute_relations(LOCS, buf, DEFAULT_CONFIG)
    want = predict_distribution(model, P, LOCS, "location", relations=rel)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_adapter_empty_buffer_equals_no_relations(tmp_path):
    path, model = save_checkpoint(tmp_path, "nonlinear")
    adapter = LearnedPolicyAdapter(path)
    got = adapter.distribution(P, LOCS, buffer=RecentBuffer(maxlen=3), level="location")
    want = predict_distribution(model, P, LOCS, "location", relations=None)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_adapter_action_requires_selected_location(tmp_path):
    path, _ = save_checkpoint(tmp_path, "nonlinear")
    adapter = LearnedPolicyAdapter(path)
    with pytest.raises(ValueError):
        adapter.distribution(P, ACTS, level="action")   # no selected_location
    dist = adapter.distribution(P, ACTS, level="action", selected_location=LOCS[0])
    assert abs(dist.sum() - 1.0) < 1e-9


# ---------------------------------------------------------------- controller --

from npc_policy.controller import DecisionController
from npc_policy.scorer import HandAuthoredScorer


def replay_scorer_by_hand(scorer, personality, worlds_cycles, seed):
    """Reference behaviour: drive scorer + buffers manually, mirroring the
    controller's documented rules, for regression comparison."""
    rng = np.random.default_rng(seed)
    H_L = RecentBuffer(maxlen=DEFAULT_CONFIG.K_L)
    H_A = RecentBuffer(maxlen=DEFAULT_CONFIG.K_A)
    last_loc = None
    picks = []
    for locs, acts_of in worlds_cycles:
        t = scorer.trace(personality, locs, buffer=H_L, level="location")
        i = int(rng.choice(len(locs), p=t.P_rule))
        loc = locs[i]
        if loc.id != last_loc:
            H_A.clear()
        H_L.push(loc)
        last_loc = loc.id
        acts = acts_of(loc.id)
        t2 = scorer.trace(personality, acts, buffer=H_A, level="action")
        j = int(rng.choice(len(acts), p=t2.P_rule))
        H_A.push(acts[j])
        picks.append((loc.id, acts[j].id))
    return picks


def test_controller_scorer_regression():
    """Scorer through the new policy-shaped controller reproduces the manual
    reference exactly (same rng consumption order, same buffer rules)."""
    scorer = HandAuthoredScorer()
    acts_of = lambda loc_id: ACTS
    cycles = [(LOCS, acts_of)] * 6
    want = replay_scorer_by_hand(scorer, P, cycles, seed=7)

    ctrl = DecisionController(scorer, mode="sample", rng=np.random.default_rng(7))
    got = []
    for locs, acts_fn in cycles:
        d_loc = ctrl.choose_location(P, locs)
        d_act = ctrl.choose_action(P, acts_fn(d_loc.option.id))
        assert d_loc.trace is not None                       # scorer keeps traces
        np.testing.assert_array_equal(d_loc.distribution, d_loc.trace.P_rule)
        got.append((d_loc.option.id, d_act.option.id))
    assert got == want


def test_controller_drives_learned_adapter(tmp_path):
    path, model = save_checkpoint(tmp_path, "nonlinear")
    ctrl = DecisionController(LearnedPolicyAdapter(path), mode="sample",
                              rng=np.random.default_rng(0))
    d_loc = ctrl.choose_location(P, LOCS)
    assert d_loc.trace is None
    assert abs(d_loc.distribution.sum() - 1.0) < 1e-9
    d_act = ctrl.choose_action(P, ACTS)        # would raise without ctx pass-through
    assert d_act.trace is None

    # first action after arriving: H_A empty -> equals context-only prediction
    want = predict_distribution(model, P, ACTS, "action",
                                relations=None, selected_location=d_loc.option)
    np.testing.assert_allclose(d_act.distribution, want, rtol=0, atol=1e-12)


def test_controller_action_context_follows_location_change(tmp_path):
    path, model = save_checkpoint(tmp_path, "nonlinear")
    ctrl = DecisionController(LearnedPolicyAdapter(path), mode="argmax")
    ctrl.choose_location(P, [LOCS[0]])          # forced tavern
    ctrl.choose_action(P, ACTS)
    ctrl.choose_location(P, [LOCS[1]])          # forced move -> library
    d = ctrl.choose_action(P, ACTS)
    # buffer was cleared by the move AND context switched to the new location
    want = predict_distribution(model, P, ACTS, "action",
                                relations=None, selected_location=LOCS[1])
    np.testing.assert_allclose(d.distribution, want, rtol=0, atol=1e-12)
