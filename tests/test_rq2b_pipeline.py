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
