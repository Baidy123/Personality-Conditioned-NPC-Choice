"""Acceptance tests for the Study 2A pipeline (design spec §6:
docs/specs/2026-07-10-rq2-2a-pipeline-design.md)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from npc_policy import (
    DEFAULT_CONFIG,
    ControlledCase,
    HandAuthoredScorer,
    Option,
    Personality,
    ScorerConfig,
)
from experiments.rq2 import common


def _loc(id_, **tags):
    return Option.location(id_, **tags)


def _act(id_, **tags):
    return Option.action(id_, **tags)


def _mk_case(decision_type="location", personality=None, n_cand=3,
             selected=None, recent_locs=(), recent_acts=(), target=None):
    """Minimal hand-built ControlledCase for unit tests."""
    p = np.zeros(5) if personality is None else np.asarray(personality, float)
    if decision_type == "location":
        cands = [_loc(f"L{i}", social=0.1 * i) for i in range(n_cand)]
    else:
        cands = [_act(f"A{i}", social=0.1 * i) for i in range(n_cand)]
    t = np.full(n_cand, 1.0 / n_cand) if target is None else np.asarray(target, float)
    return ControlledCase(
        personality=p, decision_type=decision_type, candidates=cands,
        selected_location=selected, recent_locations=list(recent_locs),
        recent_actions_same_location=list(recent_acts),
        candidate_history_features=None, target_distribution=t,
    )


class TestCommon:
    def test_config_hash_stable_and_sensitive(self):
        h1 = common.config_hash(DEFAULT_CONFIG)
        h2 = common.config_hash(ScorerConfig())
        h3 = common.config_hash(ScorerConfig(K_L=2))
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 64

    def test_pool_roundtrip(self, tmp_path):
        case = _mk_case()
        common.write_pool(tmp_path / "p.jsonl", [(case, {"id": "syn-000000", "source": "synthetic", "world": "full"})])
        [(back, tags)] = common.read_pool(tmp_path / "p.jsonl")
        assert tags == {"id": "syn-000000", "source": "synthetic", "world": "full"}
        assert back.decision_type == "location"
        np.testing.assert_allclose(back.target_distribution, case.target_distribution)

    def test_pool_roundtrip_maximal_action_case(self, tmp_path):
        from npc_policy.relations import Relations

        home = _loc("home", social=0.4, privacy=0.9)
        far = _loc("far", stimulation=0.6)
        acts = [_act("rest", social=0.2), _act("read", cognitive=0.8)]
        rel = Relations(np.array([0.5, 0.0]), np.array([0.7, 0.2]), np.array([0.3, 0.8]))
        case = _mk_case(
            "action", personality=[0.5, -0.3, 0.1, 0.9, -1.0], n_cand=2,
            selected="home", recent_locs=[far, home], recent_acts=acts,
            target=[0.25, 0.75],
        )
        case.candidate_history_features = rel
        common.write_pool(tmp_path / "p.jsonl", [(case, {"id": "roll-000001", "source": "rollout", "world": "full"})])
        [(back, tags)] = common.read_pool(tmp_path / "p.jsonl")
        assert tags == {"id": "roll-000001", "source": "rollout", "world": "full"}
        np.testing.assert_allclose(back.personality, case.personality)
        assert back.decision_type == "action"
        assert back.selected_location == "home"
        assert [o.id for o in back.candidates] == [o.id for o in case.candidates]
        assert [o.level for o in back.candidates] == [o.level for o in case.candidates]
        for b, o in zip(back.candidates, case.candidates):
            np.testing.assert_allclose(b.features, o.features)
        assert [o.id for o in back.recent_locations] == ["far", "home"]
        assert [o.level for o in back.recent_locations] == ["location", "location"]
        for b, o in zip(back.recent_locations, case.recent_locations):
            np.testing.assert_allclose(b.features, o.features)
        assert [o.id for o in back.recent_actions_same_location] == ["rest", "read"]
        assert [o.level for o in back.recent_actions_same_location] == ["action", "action"]
        for b, o in zip(back.recent_actions_same_location, case.recent_actions_same_location):
            np.testing.assert_allclose(b.features, o.features)
        assert back.candidate_history_features is not None
        np.testing.assert_allclose(back.candidate_history_features.rep, rel.rep)
        np.testing.assert_allclose(back.candidate_history_features.sim, rel.sim)
        np.testing.assert_allclose(back.candidate_history_features.nov, rel.nov)
        np.testing.assert_allclose(back.target_distribution, case.target_distribution)

    def test_read_pool_rejects_missing_gen_tags(self, tmp_path):
        d = _mk_case().to_dict()   # no "gen" key: a tagless record is corruption
        path = tmp_path / "p.jsonl"
        path.write_text(json.dumps(d) + "\n", encoding="utf-8")
        with pytest.raises(KeyError):
            common.read_pool(path)

    def test_case_to_inputs_action_uses_newest_recent_location(self):
        home = _loc("home", social=0.4, privacy=0.9)
        case = _mk_case("action", selected="home", recent_locs=[_loc("far"), home])
        d = common.case_to_inputs(case)
        np.testing.assert_allclose(d["ctx"], home.to_padded12())
        np.testing.assert_allclose(d["target"], case.target_distribution)
        assert d["d"] == 1

    def test_case_to_inputs_rejects_broken_action_invariant(self):
        case = _mk_case("action", selected="home", recent_locs=[_loc("elsewhere")])
        with pytest.raises(ValueError):
            common.case_to_inputs(case)

    def test_case_to_inputs_rejects_stale_selected_location_on_location_case(self):
        case = _mk_case("location", selected="home", recent_locs=[_loc("home")])
        with pytest.raises(ValueError):
            common.case_to_inputs(case)

    def test_case_to_inputs_copies_arrays(self):
        case = _mk_case()
        d = common.case_to_inputs(case)
        assert not np.shares_memory(d["p"], case.personality)
        assert not np.shares_memory(d["target"], case.target_distribution)

    def test_ablation_zeroes_relations(self):
        from npc_policy.relations import Relations
        rel = Relations(np.array([0.5, 0.0]), np.array([0.7, 0.2]), np.array([0.3, 0.8]))
        case = _mk_case(n_cand=2)
        case.candidate_history_features = rel
        full = common.case_to_inputs(case, "full")
        no_ctx = common.case_to_inputs(case, "no_context")
        assert full["rel"].any()
        assert not no_ctx["rel"].any()
        # location_only keeps location-case relations, zeroes action-case ones
        loc_only = common.case_to_inputs(case, "location_only")
        assert loc_only["rel"].any()
        home = _loc("home")
        act = _mk_case("action", n_cand=2, selected="home", recent_locs=[home])
        act.candidate_history_features = rel
        assert not common.case_to_inputs(act, "location_only")["rel"].any()

    def test_metrics_pinned(self):
        t = np.array([1.0, 0.0])
        q = np.array([0.5, 0.5])
        assert common.kl_np(t, q) == pytest.approx(np.log(2.0))
        assert common.kl_np(t, t) == pytest.approx(0.0)
        assert common.jsd_np(t, t) == pytest.approx(0.0)
        # nonzero JSD pins: disjoint supports hit the log(2) maximum, and the
        # mixed case must match the definitional 0.5*KL(t‖m) + 0.5*KL(q‖m)
        assert common.jsd_np(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(np.log(2.0))
        m = np.array([0.75, 0.25])
        assert common.jsd_np(t, q) == pytest.approx(
            0.5 * common.kl_np(t, m) + 0.5 * common.kl_np(q, m)
        )
        assert common.top1_agree(np.array([0.9, 0.1]), np.array([0.6, 0.4]))
        assert not common.top1_agree(np.array([0.9, 0.1]), np.array([0.4, 0.6]))

    def test_run_matrix_counts(self):
        runs = common.run_matrix()
        assert len(runs) == 130
        assert len({r.run_id for r in runs}) == 130
        s0_main = [r for r in runs if r.split == "S0" and r.ablation == "full" and r.n_train is None]
        assert len(s0_main) == 20
        assert all(r.model in ("simple", "nonlinear") for r in runs if r.split != "S0")
        smoke = common.run_matrix(smoke=True)
        assert 0 < len(smoke) <= 6

    def test_build_model_names(self):
        import torch
        for name in common.S0_MODELS:
            m = common.build_model(name, seed=0)
            assert isinstance(m, torch.nn.Module)
        with pytest.raises(ValueError):
            common.build_model("mlp", 0)
