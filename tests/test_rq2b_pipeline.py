"""Study 2B pipeline tests. Plan: docs/plans/2026-07-11-rq2-2b-pipeline.md."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.rq2.common import RunSpec, case_input_dict, read_pool, write_pool
from npc_policy import DEFAULT_CONFIG, IndependentCase, Option, RecentBuffer, compute_relations
from npc_policy.representation import Personality

CODE = Path(__file__).resolve().parents[1]


def _buf(options, maxlen):
    b = RecentBuffer(maxlen=maxlen)
    for o in options:
        b.push(o)
    return b


def _mini_independent_case() -> IndependentCase:
    cands = [Option.location("tavern", social=0.9), Option.location("library", cognitive=0.9)]
    return IndependentCase(
        personality=np.array([0.1, 0.2, 0.3, -0.1, 0.0]),
        decision_type="location",
        candidates=cands,
        recent_locations=[Option.location("market", social=0.5)],
        candidate_history_features=compute_relations(
            cands, _buf([Option.location("market", social=0.5)], 3)),
        target_choice=1,
        source="test", review_status="accepted",
    )


class TestPoolRoundTrip:
    def test_independent_case_round_trip(self, tmp_path):
        case = _mini_independent_case()
        path = tmp_path / "pool.jsonl"
        write_pool(path, [(case, {"id": "c0", "group": "train"})])
        [(back, tags)] = read_pool(path, case_cls=IndependentCase)
        assert tags == {"id": "c0", "group": "train"}
        assert back.target_choice == 1
        assert back.source == "test"
        assert [o.id for o in back.candidates] == ["tavern", "library"]

    def test_default_case_cls_unchanged(self, tmp_path):
        # 2A call sites pass no case_cls and still get ControlledCase
        from npc_policy import ControlledCase
        c = ControlledCase(
            personality=np.zeros(5), decision_type="location",
            candidates=[Option.location("tavern", social=0.9)],
            target_distribution=np.array([1.0]),
        )
        path = tmp_path / "pool.jsonl"
        write_pool(path, [(c, {"id": "x"})])
        [(back, _)] = read_pool(path)
        assert isinstance(back, ControlledCase)


class TestRunSpecTag:
    def test_tag_in_run_id(self):
        assert RunSpec("IND", "nonlinear", 3, tag="wd0.001").run_id == \
            "IND__nonlinear__wd0.001__s3"

    def test_empty_tag_keeps_2a_ids(self):
        assert RunSpec("S0", "simple", 0).run_id == "S0__simple__s0"
        assert RunSpec("G1", "simple", 0, ablation="no_context").run_id == \
            "G1__simple__abl_no_context__s0"


class TestTrainOneInjection:
    def test_to_inputs_and_weight_decay_params(self):
        """train_one accepts to_inputs and weight_decay and trains on one-hot targets."""
        from experiments.rq2.train import train_one

        def onehot_inputs(case, ablation="full"):
            d = case_input_dict(case, ablation)
            t = np.zeros(len(case.candidates))
            t[case.target_choice] = 1.0
            d["target"] = t
            return d

        cases = [_mini_independent_case() for _ in range(8)]
        result, state = train_one(
            RunSpec("IND", "simple", 0), cases, cases[:4],
            torch.device("cpu"), to_inputs=onehot_inputs, weight_decay=0.0,
            max_epochs=2, batch_size=4,
        )
        assert result["epochs_run"] <= 2
        assert np.isfinite(result["best_val_kl"])   # one-hot KL == NLL of the choice
        assert any(k == "w" for k in state)


class TestCaseInputDict:
    def test_matches_case_to_inputs_minus_target(self):
        from experiments.rq2.common import case_to_inputs
        from npc_policy import ControlledCase
        cands = [Option.location("tavern", social=0.9),
                 Option.location("library", cognitive=0.9)]
        c = ControlledCase(
            personality=np.array([0.5, 0.0, 0.0, 0.0, 0.0]),
            decision_type="location", candidates=cands,
            target_distribution=np.array([0.7, 0.3]),
        )
        d_full = case_to_inputs(c)
        d_input = case_input_dict(c)
        assert "target" not in d_input
        for k in ("p", "d", "ctx", "cand", "rel"):
            np.testing.assert_array_equal(d_full[k], d_input[k])


def _raw_case(**over) -> dict:
    base = {
        "personality": {"O": 0.2, "C": -0.1, "E": 0.5, "A": 0.0, "N": -0.3},
        "decision_type": "location",
        "recent_locations": ["market", "tavern"],
        "candidates": ["tavern", "library", "forest"],
        "choice": "library",
        "reason": "quiet after the bustle",
    }
    base.update(over)
    return base


def _raw_action_case(**over) -> dict:
    base = {
        "personality": {"O": -0.5, "C": 0.8, "E": -0.2, "A": 0.4, "N": 0.1},
        "decision_type": "action",
        "selected_location": "library",
        "recent_locations": ["chapel", "library"],
        "recent_actions_same_location": ["read"],
        "candidates": ["read", "research", "discuss"],
        "choice": "read",
    }
    base.update(over)
    return base


class TestValidate:
    def _validate(self, case):
        from experiments.rq2.independent import load_base_world, validate_case
        return validate_case(case, load_base_world())

    def test_valid_location_case(self):
        assert self._validate(_raw_case()) is None

    def test_valid_action_case(self):
        assert self._validate(_raw_action_case()) is None

    @pytest.mark.parametrize("case,reason", [
        (_raw_case(candidates=["tavern", "netbar"]), "unknown_location"),
        (_raw_case(recent_locations=["enemy_camp"]), "unknown_location"),
        (_raw_case(personality={"O": 1.5, "C": 0, "E": 0, "A": 0, "N": 0}), "trait_range"),
        (_raw_case(personality={"O": 0.1}), "traits_missing"),
        (_raw_case(recent_locations=["market"] * 4), "history_too_long"),
        (_raw_case(candidates=["tavern"]), "candidate_count"),
        (_raw_case(candidates=["tavern", "tavern", "library"]), "duplicate_candidates"),
        (_raw_case(choice="arena"), "choice_not_in_candidates"),
        (_raw_case(decision_type="teleport"), "bad_decision_type"),
        (_raw_case(selected_location="tavern"), "location_case_has_selected"),
        (_raw_action_case(candidates=["read", "research"]), "not_full_action_set"),
        (_raw_action_case(recent_locations=["chapel"]), "selected_location_mismatch"),
        (_raw_action_case(recent_actions_same_location=["pray"]), "action_not_native"),
        (_raw_action_case(selected_location=None), "missing_selected_location"),
    ])
    def test_rejections(self, case, reason):
        assert self._validate(case) == reason

    def test_empty_recent_locations_ok_for_action(self):
        # enrich later auto-fills [selected_location]; validation accepts it
        assert self._validate(_raw_action_case(recent_locations=[])) is None


def _write_raw_dir(tmp_path, n_general=40, n_pers=6, n_arena=6) -> Path:
    """Synthesise a small raw/ directory covering all three batch types."""
    import random
    rng = random.Random(0)
    locs = ["tavern", "library", "chapel", "market", "forest"]

    def general(i):
        pool = rng.sample(locs, 3)
        return _raw_case(
            personality={"O": rng.uniform(-1, 0.5), "C": rng.uniform(-0.5, 1),
                         "E": rng.uniform(-1, 1), "A": 0.0, "N": 0.0},
            recent_locations=rng.sample(locs, rng.randint(0, 3)),
            candidates=pool, choice=pool[i % 3])

    def pers(i):
        pool = rng.sample(locs, 3)
        return _raw_case(
            personality={"O": 0.8, "C": -0.8, "E": 0.1, "A": 0.0, "N": 0.0},
            candidates=pool, choice=pool[i % 3])

    def arena(i):
        pool = ["arena"] + rng.sample(locs, 2)
        return _raw_case(candidates=pool, choice="arena")

    raw = tmp_path / "raw"
    raw.mkdir()
    for name, gen, n in [("general.json", general, n_general),
                         ("pers.json", pers, n_pers),
                         ("arena.json", arena, n_arena)]:
        payload = [{"_meta": {"source": "test-llm"}}] + [gen(i) for i in range(n)]
        (raw / name).write_text(json.dumps(payload), encoding="utf-8")
    return raw


class TestImport:
    def test_end_to_end_import(self, tmp_path):
        from experiments.rq2.import_independent import run_import
        raw = _write_raw_dir(tmp_path)
        meta = run_import(raw_dir=raw, out_dir=tmp_path / "out")
        splits = json.loads((tmp_path / "out" / "splits.json").read_text(encoding="utf-8"))["splits"]
        # proportional scaling: general pool of 40 → 550:100:75 ratios
        n = sum(len(splits[k]) for k in ("train", "val", "test_iid"))
        assert n == 40
        assert len(splits["test_iid"]) == round(40 * 75 / 725)
        assert len(splits["test_pers"]) == 6 and len(splits["test_arena"]) == 6
        assert (tmp_path / "out" / "report.txt").exists()
        assert meta["accepted"] == 52

    def test_isolation_no_structured_content_in_train_val(self, tmp_path):
        from experiments.rq2.import_independent import run_import
        from experiments.rq2.independent import in_pers_region, touches_arena
        raw = _write_raw_dir(tmp_path)
        run_import(raw_dir=raw, out_dir=tmp_path / "out")
        pool = read_pool(tmp_path / "out" / "cases.jsonl", case_cls=IndependentCase)
        splits = json.loads((tmp_path / "out" / "splits.json").read_text(encoding="utf-8"))["splits"]
        trainval = set(splits["train"]) | set(splits["val"])
        for case, tags in pool:
            if tags["id"] in trainval:
                assert not in_pers_region(case) and not touches_arena(case)

    def test_rejects_recorded_and_duplicates_dropped(self, tmp_path):
        from experiments.rq2.import_independent import run_import
        raw = tmp_path / "raw"
        raw.mkdir()
        bad = _raw_case(choice="arena")                  # invalid: not a candidate
        dup = _raw_case()
        payload = [{"_meta": {"source": "m"}}, dup, dup, bad,
                   {**_raw_case(candidates=["tavern", "library"], choice="tavern"),
                    "review_status": "rejected"}]
        (raw / "b.json").write_text(json.dumps(payload), encoding="utf-8")
        meta = run_import(raw_dir=raw, out_dir=tmp_path / "out")
        rejected = [json.loads(l) for l in
                    (tmp_path / "out" / "rejected.jsonl").read_text(encoding="utf-8").splitlines()]
        reasons = sorted(r["reason"] for r in rejected)
        assert reasons == ["choice_not_in_candidates", "duplicate", "user_rejected"]
        assert meta["accepted"] == 1

    def test_deterministic_splits(self, tmp_path):
        from experiments.rq2.import_independent import run_import
        raw = _write_raw_dir(tmp_path)
        run_import(raw_dir=raw, out_dir=tmp_path / "o1")
        run_import(raw_dir=raw, out_dir=tmp_path / "o2")
        s1 = (tmp_path / "o1" / "splits.json").read_text(encoding="utf-8")
        s2 = (tmp_path / "o2" / "splits.json").read_text(encoding="utf-8")
        assert s1 == s2


class TestEnrich:
    def test_location_features_match_world(self):
        from experiments.rq2.independent import enrich_case, load_base_world
        w = load_base_world()
        case = enrich_case(_raw_case(), w, source="m1")
        np.testing.assert_array_equal(
            case.candidates[0].features, w.effective_location("tavern").features)
        assert case.target_choice == 1          # "library" at index 1
        assert case.source == "m1" and case.review_status == "accepted"

    def test_relations_match_reference(self):
        from experiments.rq2.independent import enrich_case, load_base_world
        w = load_base_world()
        case = enrich_case(_raw_case(), w, source="m1")
        history = [w.effective_location("market"), w.effective_location("tavern")]
        ref = compute_relations(case.candidates, _buf(history, DEFAULT_CONFIG.K_L))
        np.testing.assert_allclose(case.candidate_history_features.rep, ref.rep)
        np.testing.assert_allclose(case.candidate_history_features.sim, ref.sim)

    def test_empty_history_gives_none_relations(self):
        from experiments.rq2.independent import enrich_case, load_base_world
        case = enrich_case(_raw_case(recent_locations=[]), load_base_world(), "m")
        assert case.candidate_history_features is None

    def test_action_case_autofills_recent_location(self):
        from experiments.rq2.independent import enrich_case, load_base_world
        case = enrich_case(_raw_action_case(recent_locations=[]),
                           load_base_world(), "m")
        assert [o.id for o in case.recent_locations] == ["library"]

    def test_enriched_case_feeds_model_inputs(self):
        from experiments.rq2.independent import (
            enrich_case, independent_case_to_inputs, load_base_world)
        case = enrich_case(_raw_action_case(), load_base_world(), "m")
        d = independent_case_to_inputs(case)
        assert d["target"].tolist() == [1.0, 0.0, 0.0]      # choice "read" = index 0
        assert d["cand"].shape == (3, 12) and d["d"] == 1


class TestTrain2B:
    def test_run_matrix_shape(self):
        from experiments.rq2.train_2b import run_matrix_2b
        specs = run_matrix_2b()
        assert len(specs) == 40                     # 2×5 simple + 2×5×3 nonlinear
        assert len({s.run_id for s in specs}) == 40
        assert all(s.split == "IND" for s in specs)

    def test_wd_of_spec(self):
        from experiments.rq2.train_2b import run_matrix_2b, wd_of
        for s in run_matrix_2b():
            if "nonlinear" in s.model:
                assert wd_of(s) in (1e-4, 1e-3, 1e-2) and s.tag.startswith("wd")
            else:
                assert wd_of(s) == 0.0 and s.tag == ""

    def test_end_to_end_smoke(self, tmp_path):
        """import → train 2 runs, 2 epochs → resumable artefacts on disk."""
        from experiments.rq2.import_independent import run_import
        from experiments.rq2.train_2b import train_main
        raw = _write_raw_dir(tmp_path, n_general=30, n_pers=4, n_arena=4)
        run_import(raw_dir=raw, out_dir=tmp_path / "data")
        train_main(["--data", str(tmp_path / "data"),
                    "--results", str(tmp_path / "res"),
                    "--only", "IND__simple", "--max-epochs", "2"])
        runs = list((tmp_path / "res" / "runs").glob("*.json"))
        assert len(runs) == 5                       # simple × 5 seeds
        meta = json.loads(runs[0].read_text(encoding="utf-8"))
        assert np.isfinite(meta["best_val_kl"])


class TestRun2B:
    def test_scorer_and_uniform_rows(self, tmp_path):
        """Full mini-pipeline: import → train simple → evaluate all systems."""
        import csv
        from experiments.rq2.import_independent import run_import
        from experiments.rq2.run_2b import eval_main
        from experiments.rq2.train_2b import train_main
        raw = _write_raw_dir(tmp_path, n_general=30, n_pers=4, n_arena=4)
        run_import(raw_dir=raw, out_dir=tmp_path / "data")
        train_main(["--data", str(tmp_path / "data"),
                    "--results", str(tmp_path / "res"),
                    "--only", "IND__simple", "--max-epochs", "2"])
        eval_main(["--data", str(tmp_path / "data"),
                   "--results", str(tmp_path / "res")])
        with open(tmp_path / "res" / "main_table.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        systems = {r["system"] for r in rows}
        assert {"uniform", "scorer", "simple"} <= systems
        groups = {r["group"] for r in rows}
        assert {"all", "test_iid", "test_pers", "test_arena"} <= groups
        for r in rows:
            assert 0.0 <= float(r["top1_mean"]) <= 1.0
            assert float(r["nll_mean"]) >= 0.0
        assert (tmp_path / "res" / "group_bars.png").exists()

    def test_scorer_predictions_scored_like_models(self):
        """Scorer NLL/top-1 computed from its distribution on the labelled choice."""
        from experiments.rq2.run_2b import scorer_probs
        from experiments.rq2.independent import enrich_case, load_base_world
        case = enrich_case(_raw_case(), load_base_world(), "m")
        q = scorer_probs(case)
        assert q.shape == (3,) and abs(q.sum() - 1.0) < 1e-9


class TestParseRawFile:
    def test_meta_header_and_user_rejection(self, tmp_path):
        from experiments.rq2.independent import parse_raw_file
        payload = [
            {"_meta": {"source": "gpt-5.5"}},
            _raw_case(),
            {**_raw_case(choice="tavern"), "review_status": "rejected"},
        ]
        f = tmp_path / "batch_01.json"
        f.write_text(json.dumps(payload), encoding="utf-8")
        source, cases, user_rejected = parse_raw_file(f)
        assert source == "gpt-5.5"
        assert len(cases) == 1 and len(user_rejected) == 1

    def test_missing_meta_defaults_unknown(self, tmp_path):
        from experiments.rq2.independent import parse_raw_file
        f = tmp_path / "b.json"
        f.write_text(json.dumps([_raw_case()]), encoding="utf-8")
        source, cases, _ = parse_raw_file(f)
        assert source == "unknown" and len(cases) == 1
