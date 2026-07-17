"""Tests for npc_policy.policies (unified policy layer) and the generalised
DecisionController (plan: docs/plans/2026-07-17-rq3-sequence-pipeline.md)."""

import numpy as np
import pytest
import torch

from npc_policy.config import DEFAULT_CONFIG
from npc_policy.learned import NonlinearPolicy, predict_distribution
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
