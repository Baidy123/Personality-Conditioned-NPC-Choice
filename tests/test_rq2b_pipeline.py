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
