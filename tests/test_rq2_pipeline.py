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


class TestTraining:
    @pytest.fixture(scope="class")
    @classmethod
    def tiny_cases(cls):
        """Representable teacher (bilinear, N temperature off) — design §6 item 3."""
        from experiments.rq2 import gen_controlled as g
        from npc_policy import load_world
        from npc_policy.config import LevelParams
        worlds = g.ensure_worlds()
        world = load_world(worlds["full"])
        cfg = ScorerConfig(
            base_form="bilinear",
            location=LevelParams(tau_0=0.9, lambda_N=0.0),
            action=LevelParams(lambda_N=0.0),
        )
        scorer = HandAuthoredScorer(config=cfg)
        s = g.SyntheticSampler(world, scorer, np.random.default_rng(7))
        cases = ([s.location_case() for _ in range(150)]
                 + [s.action_case() for _ in range(150)])
        return cases[:240], cases[240:]        # train, val

    def test_simple_converges_on_representable_teacher(self, tiny_cases):
        import torch
        from experiments.rq2 import train as tr
        train_cases, val_cases = tiny_cases
        result, state = tr.train_one(
            common.RunSpec("S0", "simple", 0), train_cases, val_cases,
            device=torch.device("cpu"),
            lr=0.05, batch_size=60, max_epochs=500, patience=500,
        )
        # representable target (bilinear teacher, N temperature off) + overdetermined
        # (240 cases >> 228 params) -> near-zero KL certifies representability, not
        # interpolation (dev_log.md 2026-07-09 lesson). Adam is slower than the
        # LBFGS certificate in test_learned.py, hence the looser 1e-3 bar.
        assert result["best_val_kl"] < 1e-3
        assert result["dtype"] == "float64"
        assert all(v.dtype == torch.float64 for v in state.values())

    def test_early_stopping_triggers(self, tiny_cases):
        import torch
        from experiments.rq2 import train as tr
        train_cases, val_cases = tiny_cases
        result, _ = tr.train_one(
            common.RunSpec("S0", "simple", 0), train_cases[:40], val_cases[:20],
            device=torch.device("cpu"),
            lr=0.0, batch_size=40, max_epochs=100, patience=5,
        )
        assert result["epochs_run"] == 6       # epoch 0 sets best; 5 stale epochs; stop

    def test_run_all_resumes(self, tiny_cases, tmp_path):
        import torch
        from experiments.rq2 import train as tr
        train_cases, val_cases = tiny_cases
        specs = [common.RunSpec("S0", "simple", 0), common.RunSpec("S0", "simple", 1)]
        calls = []

        def fake_train(spec, tc, vc, device, **kw):
            calls.append(spec.run_id)
            return ({"run_id": spec.run_id, "best_val_kl": 0.5, "dtype": "float64"},
                    common.build_model(spec.model, spec.seed).state_dict())

        loader = lambda spec: (train_cases[:10], val_cases[:10])
        tr.run_all(specs, loader, tmp_path, torch.device("cpu"), train_fn=fake_train)
        assert sorted(calls) == [s.run_id for s in specs]
        assert (tmp_path / "runs" / "S0__simple__s0.json").exists()
        assert (tmp_path / "models" / "S0__simple__s0.pt").exists()
        calls.clear()
        tr.run_all(specs, loader, tmp_path, torch.device("cpu"), train_fn=fake_train)
        assert calls == []                     # everything skipped on re-run

    def test_agnostic_ignores_personality_after_training(self, tiny_cases):
        import torch
        from experiments.rq2 import train as tr
        from npc_policy.learned import predict_distribution
        train_cases, val_cases = tiny_cases
        _, state = tr.train_one(
            common.RunSpec("S0", "agnostic_simple", 0), train_cases[:60], val_cases[:20],
            device=torch.device("cpu"), lr=0.05, batch_size=60, max_epochs=20, patience=20,
        )
        model = common.build_model("agnostic_simple", 0)
        model.load_state_dict(state)
        case = train_cases[0]
        d1 = predict_distribution(model, Personality(np.array([1.0, -1, 1, -1, 1])),
                                  case.candidates, "location",
                                  relations=case.candidate_history_features)
        d2 = predict_distribution(model, Personality(np.zeros(5)),
                                  case.candidates, "location",
                                  relations=case.candidate_history_features)
        np.testing.assert_allclose(d1, d2, atol=1e-12)


class TestRun2A:
    def test_eval_cases_uniform_pin(self):
        from experiments.rq2 import run_2a
        from npc_policy.learned import UniformBaseline
        case = _mk_case(n_cand=2, target=[1.0, 0.0])
        rows = run_2a.eval_cases(UniformBaseline(), [case])
        assert len(rows) == 1
        assert rows[0]["decision_type"] == "location"
        assert rows[0]["kl"] == pytest.approx(np.log(2.0))
        assert rows[0]["top1"] == 1               # argmax tie resolves to index 0

    def test_eval_cases_applies_ablation(self):
        from experiments.rq2 import run_2a
        from npc_policy.learned import SimplePolicy
        import torch
        from npc_policy.relations import Relations
        torch.manual_seed(0)
        model = SimplePolicy()
        with torch.no_grad():
            model.w += torch.randn_like(model.w)
        case = _mk_case(n_cand=3)
        case.candidate_history_features = Relations(
            np.array([0.9, 0.0, 0.0]), np.array([0.8, 0.1, 0.1]), np.array([0.2, 0.9, 0.9]))
        r_full = run_2a.eval_cases(model, [case], ablation="full")[0]
        r_none = run_2a.eval_cases(model, [case], ablation="no_context")[0]
        assert r_full["kl"] != pytest.approx(r_none["kl"])

    def test_aggregate_mean_std(self):
        from experiments.rq2 import run_2a
        per_run = [
            {"split": "S0", "model": "simple", "ablation": "full", "n_train": None,
             "seed": s, "eval_split": "S0", "decision_type": "all",
             "kl": v, "jsd": v / 2, "top1": 1.0, "n_cases": 10}
            for s, v in enumerate([0.1, 0.2, 0.3])
        ]
        table = run_2a.aggregate(per_run)
        [row] = table
        assert row["kl_mean"] == pytest.approx(0.2)
        assert row["kl_std"] == pytest.approx(np.std([0.1, 0.2, 0.3], ddof=1))
        assert row["n_seeds"] == 3

    def test_eval_splits_for(self):
        from experiments.rq2 import run_2a
        avail = list(common.ALL_SPLITS)
        s0_main = {"split": "S0", "ablation": "full", "n_train": None}
        assert run_2a.eval_splits_for(s0_main, avail) == avail
        s0_abl = {"split": "S0", "ablation": "no_context", "n_train": None}
        assert run_2a.eval_splits_for(s0_abl, avail) == ["S0"]
        s0_size = {"split": "S0", "ablation": "full", "n_train": 1000}
        assert run_2a.eval_splits_for(s0_size, avail) == ["S0"]
        g_run = {"split": "G3", "ablation": "full", "n_train": None}
        assert run_2a.eval_splits_for(g_run, avail) == ["G3", "S0"]

    def test_load_student_roundtrip(self, tmp_path):
        import torch
        from experiments.rq2 import run_2a
        from npc_policy.learned import predict_distribution
        model = common.build_model("nonlinear", seed=3)
        (tmp_path / "models").mkdir()
        torch.save({"model": "nonlinear", "state_dict": model.state_dict()},
                   tmp_path / "models" / "X.pt")
        back = run_2a.load_student(tmp_path, "X")
        case = _mk_case(n_cand=4)
        p = Personality(np.array([0.3, -0.2, 0.5, 0.0, -0.9]))
        np.testing.assert_allclose(
            predict_distribution(model, p, case.candidates, "location"),
            predict_distribution(back, p, case.candidates, "location"), atol=1e-12)


class TestEDiag:
    @pytest.fixture()
    def setup(self):
        import torch
        from experiments.rq2 import run_e_diag as ed
        from experiments.rq2 import gen_controlled as g
        from npc_policy import load_world
        from npc_policy.learned import SimplePolicy
        torch.manual_seed(11)
        model = SimplePolicy()
        with torch.no_grad():
            model.w += 0.3 * torch.randn_like(model.w)
        world = load_world(g.ensure_worlds()["full"])
        return ed, ed.StudentTraceAdapter(model), world, model

    def test_adapter_matches_predict_distribution(self, setup):
        from npc_policy import RecentBuffer
        from npc_policy.learned import predict_distribution
        from npc_policy.relations import compute_relations
        ed, adapter, world, model = setup
        p = Personality(np.array([0.5, -0.5, 0.2, 0.0, -0.3]))
        locs = world.resolve()
        # empty buffer
        np.testing.assert_allclose(
            adapter.trace(p, locs, buffer=None, level="location").P_rule,
            predict_distribution(model, p, locs, "location"), atol=1e-12)
        # non-empty buffer -> relations computed exactly as the controller would
        buf = RecentBuffer(maxlen=3)
        buf.push(locs[0]); buf.push(locs[2])
        rel = compute_relations(locs, buf, DEFAULT_CONFIG)
        np.testing.assert_allclose(
            adapter.trace(p, locs, buffer=buf, level="location").P_rule,
            predict_distribution(model, p, locs, "location", relations=rel), atol=1e-12)
        # action level uses current_location as the context
        adapter.current_location = locs[1]
        acts = world.actions_at(locs[1].id)
        np.testing.assert_allclose(
            adapter.trace(p, acts, buffer=None, level="action").P_rule,
            predict_distribution(model, p, acts, "action",
                                 selected_location=locs[1]), atol=1e-12)

    def test_adapter_action_without_location_raises(self, setup):
        ed, adapter, world, model = setup
        adapter.current_location = None
        with pytest.raises(ValueError):
            adapter.trace(Personality(np.zeros(5)),
                          world.actions_at("tavern"), level="action")

    def test_student_trajectory_runs_and_tracks_location(self, setup):
        ed, adapter, world, model = setup
        p = Personality(np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
        for memory in ("full", "location_only", "none"):
            visits, acts = ed.student_trajectory(adapter, world, p, seed=0,
                                                 rounds=4, memory=memory)
            assert len(visits) == len(acts) == 4
            assert adapter.current_location.id == visits[-1]

    def test_train_one_saves_best_not_last_state(self, monkeypatch):
        """A val curve that worsens after its minimum must return the epoch-1
        snapshot, not the final weights (final review test gap)."""
        import torch
        from experiments.rq2 import train as tr
        vals = [1.0, 0.5, 0.9, 0.9, 0.9, 0.9]
        snaps = []

        def fake_val(model, batches):
            snaps.append({k: v.detach().to("cpu", torch.float64).clone()
                          for k, v in model.state_dict().items()})
            return vals[len(snaps) - 1]

        monkeypatch.setattr(tr, "_mean_val_kl", fake_val)
        cases = [_mk_case(n_cand=3) for _ in range(8)]
        result, state = tr.train_one(
            common.RunSpec("S0", "simple", 0), cases, cases[:2],
            device=torch.device("cpu"),
            lr=0.1, batch_size=8, max_epochs=10, patience=3,
        )
        assert result["best_epoch"] == 1
        assert result["best_val_kl"] == pytest.approx(0.5)
        assert result["epochs_run"] == 5      # epochs 2-4 stale -> stop
        for k, v in state.items():            # snapshot from epoch 1, not last
            assert torch.equal(v, snaps[1][k]), k

    def test_make_loader_nested_subsets(self, tmp_path):
        from experiments.rq2 import train as tr
        cases = [(_mk_case(), {"id": f"syn-{i:07d}", "source": "synthetic",
                               "world": "full"}) for i in range(8)]
        common.write_pool(tmp_path / "pool.jsonl", cases)
        ids = [t["id"] for _, t in cases]
        (tmp_path / "splits.json").write_text(json.dumps(
            {"s0_test_ids": [], "splits": {"S0": {"train": ids[:6], "val": ids[6:]}}}),
            encoding="utf-8")
        loader = tr.make_loader(tmp_path)
        full_train, val = loader(common.RunSpec("S0", "simple", 0))
        sub_train, _ = loader(common.RunSpec("S0", "simple", 0, n_train=3))
        assert len(full_train) == 6 and len(val) == 2 and len(sub_train) == 3
        # nested prefix: the n_train subset is exactly the head of the full list
        for a, b in zip(sub_train, full_train[:3]):
            assert a is b
