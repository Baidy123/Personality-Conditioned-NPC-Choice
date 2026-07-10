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


class TestGeneration:
    @pytest.fixture(scope="class")
    @classmethod
    def gen(cls):
        from experiments.rq2 import gen_controlled as g
        from npc_policy import load_world
        g_worlds = g.ensure_worlds()          # rq1 variants + arena_locked path dict
        world = load_world(g_worlds["full"])
        scorer = HandAuthoredScorer()
        return g, world, scorer

    def test_synthetic_location_labels_match_teacher(self, gen):
        g, world, scorer = gen
        s = g.SyntheticSampler(world, scorer, np.random.default_rng(1))
        for _ in range(20):
            c = s.location_case()
            assert 2 <= len(c.candidates) <= 8
            ids = [o.id for o in c.candidates]
            assert len(set(ids)) == len(ids)
            expect = scorer.distribution(
                Personality(c.personality), c.candidates,
                relations=c.candidate_history_features, level="location")
            np.testing.assert_allclose(c.target_distribution, expect, atol=1e-12)

    def test_synthetic_action_case_invariants(self, gen):
        g, world, scorer = gen
        s = g.SyntheticSampler(world, scorer, np.random.default_rng(2))
        for _ in range(20):
            c = s.action_case()
            assert c.decision_type == "action"
            assert c.recent_locations, "action case must carry its location context"
            assert c.recent_locations[-1].id == c.selected_location
            expect = scorer.distribution(
                Personality(c.personality), c.candidates,
                relations=c.candidate_history_features, level="action")
            np.testing.assert_allclose(c.target_distribution, expect, atol=1e-12)
            # feeds the model layer without tripping the action-context guard
            common.case_to_inputs(c)

    def test_rollout_labels_and_buffers(self, gen):
        g, world, scorer = gen
        recs = g.rollout_records("full", world, scorer, n_traj=1,
                                 rng=np.random.default_rng(3), rounds=6)
        assert len(recs) == 12                       # 6 location + 6 action cases
        for case, tags in recs:
            assert tags["source"] == "rollout" and tags["world"] == "full"
            expect = scorer.distribution(
                Personality(case.personality), case.candidates,
                relations=case.candidate_history_features,
                level=case.decision_type)
            np.testing.assert_allclose(case.target_distribution, expect, atol=1e-12)
            if case.decision_type == "action":
                assert case.recent_locations[-1].id == case.selected_location
        # very first location case: empty buffer -> zero relations
        first = recs[0][0]
        assert first.decision_type == "location"
        assert not first.recent_locations
        assert not first.candidate_history_features.rep.any()


def _held_out_checks(g) -> dict:
    """Independent restatements of each split's held-out condition.

    Used positively on test sets and NEGATED on train/val manifests — the
    negated direction is what catches an inverted/no-op TRAIN_FILTERS entry
    (2026-07-10 quality review, I1); validating manifests with TRAIN_FILTERS
    itself would be circular.
    """
    return {
        "G1": lambda c: c.personality[0] > 0.5 and c.personality[1] < -0.5,
        "G2": lambda c: c.decision_type == "location" and any(
            o.tag("risk") > 0.6 and o.tag("privacy") > 0.6 for o in c.candidates),
        "G3": lambda c: c.decision_type == "location" and len(c.candidates) in (2, 8),
        "G5": lambda c: g.g5_saturated_history(c),
        "G6": lambda c: g.g6_touches_arena(c),
    }


TINY_SIZES = dict(n_syn_loc=140, n_syn_act=140,
                  n_traj={"full": 1, "celebration": 1, "war_camp": 1,
                          "market_locked": 1, "arena_locked": 2},
                  train=60, val=15, n_test=12, rounds=10)


class TestSplits:
    @pytest.fixture(scope="class")
    @classmethod
    def tiny_dataset(cls, tmp_path_factory):
        """End-to-end tiny generation into a temp dir (same code path as the CLI)."""
        from experiments.rq2 import gen_controlled as g
        out = tmp_path_factory.mktemp("rq2data")
        g.generate(dict(TINY_SIZES), out, seed=123)
        return g, out

    def test_outputs_exist(self, tiny_dataset):
        g, out = tiny_dataset
        assert (out / "pool.jsonl").exists()
        assert (out / "splits.json").exists()
        assert (out / "meta.json").exists()
        for split in common.ALL_SPLITS:
            assert (out / f"test_{split}.jsonl").exists()
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        assert meta["config_hash"] == common.config_hash()

    def test_manifests_disjoint_and_sized(self, tiny_dataset):
        g, out = tiny_dataset
        manifest = json.loads((out / "splits.json").read_text(encoding="utf-8"))
        s0_test = set(manifest["s0_test_ids"])
        for split, part in manifest["splits"].items():
            train, val = part["train"], part["val"]
            assert len(train) == 60 and len(val) == 15
            assert not set(train) & set(val)
            assert not set(train) & s0_test and not set(val) & s0_test

    def test_train_filters_hold(self, tiny_dataset):
        g, out = tiny_dataset
        records = common.read_pool(out / "pool.jsonl")
        by_id = {t["id"]: (c, t) for c, t in records}
        manifest = json.loads((out / "splits.json").read_text(encoding="utf-8"))
        checks = _held_out_checks(g)
        for split in common.ALL_SPLITS:
            for part in ("train", "val"):
                for cid in manifest["splits"][split][part]:
                    case, tags = by_id[cid]
                    assert g.TRAIN_FILTERS[split](case, tags), f"{split}/{part}: {cid}"
                    # independent negated held-out condition — catches an
                    # inverted TRAIN_FILTERS entry that the line above cannot
                    if split in checks:
                        assert not checks[split](case), f"{split}/{part}: {cid}"
                    if split == "G4":
                        assert tags["world"] == "full", f"G4/{part}: {cid}"
        # arena_locked cases may appear in G6 only
        for split in common.ALL_SPLITS:
            if split == "G6":
                continue
            for part in ("train", "val"):
                for cid in manifest["splits"][split][part]:
                    assert by_id[cid][1]["world"] != "arena_locked"

    def test_targeted_test_sets_satisfy_conditions(self, tiny_dataset):
        g, out = tiny_dataset
        checks = _held_out_checks(g)
        for split, ok in checks.items():
            recs = common.read_pool(out / f"test_{split}.jsonl")
            assert len(recs) == 12
            assert all(ok(c) for c, _ in recs), split
            assert all(t["id"].startswith(f"tgt-{split}-") for _, t in recs), split
        g4 = common.read_pool(out / "test_G4.jsonl")
        assert all(t["world"] in ("celebration", "war_camp", "market_locked")
                   for _, t in g4)
        # G5's held-out condition must exist in the model inputs: the repeated
        # option is among the candidates (rep hits its 1.0 ceiling)
        for c, _ in common.read_pool(out / "test_G5.jsonl"):
            buf = (c.recent_locations if c.decision_type == "location"
                   else c.recent_actions_same_location)
            assert buf[0].id in [o.id for o in c.candidates]
            assert np.max(c.candidate_history_features.rep) == pytest.approx(1.0)

    def test_test_labels_match_teacher(self, tiny_dataset):
        g, out = tiny_dataset
        scorer = HandAuthoredScorer()
        for split in ("G1", "G2", "G3", "G4", "G5", "G6"):
            for c, _ in common.read_pool(out / f"test_{split}.jsonl"):
                expect = scorer.distribution(
                    Personality(c.personality), c.candidates,
                    relations=c.candidate_history_features, level=c.decision_type)
                np.testing.assert_allclose(c.target_distribution, expect, atol=1e-12)

    def test_relations_recomputable_from_buffers(self, tiny_dataset):
        """Stored relations must equal recomputation from the stored buffers —
        pins wrong-K / wrong-decay bugs that label self-consistency cannot see
        (2026-07-10 quality review, I2)."""
        from npc_policy import DEFAULT_CONFIG, RecentBuffer
        from npc_policy.relations import compute_relations
        g, out = tiny_dataset
        recs = common.read_pool(out / "pool.jsonl")[:40]
        for split in ("G1", "G4", "G5"):
            recs += common.read_pool(out / f"test_{split}.jsonl")[:10]
        for c, _ in recs:
            if c.decision_type == "location":
                history, maxlen = c.recent_locations, DEFAULT_CONFIG.K_L
            else:
                history, maxlen = c.recent_actions_same_location, DEFAULT_CONFIG.K_A
            stored = c.candidate_history_features
            if not history:
                assert stored is None or not stored.rep.any()
                continue
            buf = RecentBuffer(maxlen=maxlen)
            for o in history:
                buf.push(o)
            expect = compute_relations(c.candidates, buf, DEFAULT_CONFIG)
            np.testing.assert_allclose(stored.rep, expect.rep, atol=1e-12)
            np.testing.assert_allclose(stored.sim, expect.sim, atol=1e-12)
            np.testing.assert_allclose(stored.nov, expect.nov, atol=1e-12)

    def test_generation_is_deterministic(self, tmp_path):
        """Same seed → byte-identical dataset files (2026-07-10 review, I4)."""
        from experiments.rq2 import gen_controlled as g
        sizes = dict(TINY_SIZES, n_syn_loc=60, n_syn_act=60, train=25, val=5, n_test=6)
        a, b = tmp_path / "a", tmp_path / "b"
        g.generate(dict(sizes), a, seed=7)
        g.generate(dict(sizes), b, seed=7)
        for name in ["pool.jsonl"] + [f"test_{s}.jsonl" for s in common.ALL_SPLITS]:
            assert (a / name).read_bytes() == (b / name).read_bytes(), name
        assert (a / "splits.json").read_bytes() == (b / "splits.json").read_bytes()

    def test_predicates_on_crafted_cases(self):
        from experiments.rq2 import gen_controlled as g
        inside = _mk_case(personality=[0.6, -0.7, 0, 0, 0])
        outside = _mk_case(personality=[0.6, 0.0, 0, 0, 0])
        assert g.g1_region(inside) and not g.g1_region(outside)
        risky = _mk_case()
        risky.candidates[0] = _loc("den", risk=0.8, privacy=0.9)
        assert g.g2_combo(risky) and not g.g2_combo(_mk_case())
        # saturated history: single-family buffer of ANY length (rep = 1.0
        # ceiling under length-normalised recency weights)
        assert g.g5_saturated_history(
            _mk_case(recent_locs=[_loc("a"), _loc("a"), _loc("a")]))
        assert g.g5_saturated_history(_mk_case(recent_locs=[_loc("a")]))
        assert g.g5_saturated_history(
            _mk_case(recent_locs=[_loc("a"), _loc("a#p1")])), "same family counts"
        assert not g.g5_saturated_history(
            _mk_case(recent_locs=[_loc("a"), _loc("a"), _loc("b")]))
        assert not g.g5_saturated_history(_mk_case()), "empty buffer stays in train"
        arena = _mk_case()
        arena.candidates[0] = _loc("arena", risk=0.8)
        assert g.g6_touches_arena(arena)
        pert = _mk_case()
        pert.candidates[0] = _loc("arena#p9", risk=0.8)
        assert g.g6_touches_arena(pert), "perturbed arena variants count as arena"
        assert not g.g6_touches_arena(_mk_case())
