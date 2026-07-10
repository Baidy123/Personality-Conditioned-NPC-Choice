"""Controlled-dataset generation for Study 2A (design §2).

Outputs (``data/rq2_controlled/``; smoke mode uses ``…_smoke/``):

  pool.jsonl              master pool with ``gen`` tags (id / source / world)
  test_<split>.jsonl      held-out sets (S0's is a random pool holdout)
  splits.json             per-split train/val case-id manifests + S0 test ids
  worlds/arena_locked.json  G6 top-up world variant
  meta.json               sizes, seed, frozen-teacher config hash

Run from ``code/``:  python -m experiments.rq2.gen_controlled [--smoke]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.rq1.gen_cases import build_worlds
from npc_policy import (
    DEFAULT_CONFIG,
    ControlledCase,
    DecisionController,
    HandAuthoredScorer,
    Option,
    Personality,
    RecentBuffer,
    load_world,
)
from npc_policy.relations import compute_relations
from npc_policy.schema import tag_index

from .common import (
    ALL_SPLITS,
    GEN_SEED,
    RQ1_WORLDS,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
    config_hash,
    dirs,
    write_pool,
)

ROLLOUT_VARIANTS = ("full", "celebration", "war_camp", "market_locked")
ROUNDS_PER_TRAJ = 50            # → 100 cases per trajectory (50 location + 50 action)

# Pool sizes. Master pool ≈ 170k so every filtered split retains ≥ 105k
# (TRAIN_SIZE + VAL_SIZE); the arena_locked block only feeds G6 (design §2).
FULL_SIZES = dict(
    n_syn_loc=49_000, n_syn_act=49_000,
    n_traj={"full": 210, "celebration": 70, "war_camp": 70, "market_locked": 70,
            "arena_locked": 300},
    train=TRAIN_SIZE, val=VAL_SIZE, n_test=TEST_SIZE, rounds=ROUNDS_PER_TRAJ,
)
SMOKE_SIZES = dict(
    n_syn_loc=700, n_syn_act=700,
    n_traj={"full": 3, "celebration": 1, "war_camp": 1, "market_locked": 1,
            "arena_locked": 4},
    train=1_200, val=200, n_test=300, rounds=ROUNDS_PER_TRAJ,
)


def base_id(option_id: str) -> str:
    """Family id: perturbed variants are ``<base>#p<k>``."""
    return option_id.split("#", 1)[0]


def ensure_worlds() -> dict[str, Path]:
    """RQ1 world variants (idempotent regeneration) + the arena-locked variant.

    Returns ``{variant_name: json_path}`` for every rollout world.
    """
    build_worlds()
    paths = {v: RQ1_WORLDS / f"{v}.json" for v in ROLLOUT_VARIANTS}
    base = json.loads(paths["full"].read_text(encoding="utf-8"))
    for loc in base["locations"]:
        if loc["id"] == "arena":
            loc["unlocked"] = False
    out = RQ1_WORLDS / "arena_locked.json"
    out.write_text(json.dumps(base, indent=2), encoding="utf-8")
    paths["arena_locked"] = out
    return paths


# ------------------------------------------------------------------ synthetic --
class SyntheticSampler:
    """Coverage sampler over the base world plus Gaussian-perturbed variants.

    Perturbed options get fresh ids (``tavern#p7``) so exact-repetition (``rep``)
    stays distinct from semantic similarity. ``mutate`` hooks let the targeted
    G-split generators edit the candidate list *before* relations and the teacher
    label are computed, so labels always match the stored features.
    """

    def __init__(self, world, scorer: HandAuthoredScorer,
                 rng: np.random.Generator, sigma: float = 0.1):
        self.world = world
        self.scorer = scorer
        self.rng = rng
        self.sigma = sigma
        self.locations = world.resolve()
        self._n = 0

    # -- option pools -------------------------------------------------------
    def _perturb(self, option: Option) -> Option:
        self._n += 1
        feats = np.clip(
            option.features + self.rng.normal(0.0, self.sigma, option.features.shape),
            0.0, 1.0,
        )
        return Option(id=f"{base_id(option.id)}#p{self._n}", features=feats,
                      level=option.level)

    def _variant(self, option: Option) -> Option:
        return self._perturb(option) if self.rng.random() < 0.5 else option

    def _random_location(self) -> Option:
        return self.locations[int(self.rng.integers(len(self.locations)))]

    def _personality(self) -> Personality:
        return Personality(self.rng.uniform(-1.0, 1.0, 5))

    def location_candidates(self, m: int) -> list[Option]:
        cands: list[Option] = []
        used: set[str] = set()
        while len(cands) < m:
            o = self._variant(self._random_location())
            if o.id in used:                       # duplicate base id → force variant
                o = self._perturb(o)
            used.add(o.id)
            cands.append(o)
        return cands

    # -- cases ----------------------------------------------------------------
    def location_case(self, personality: Personality | None = None,
                      m: int | None = None,
                      history: list[Option] | None = None,
                      mutate=None) -> ControlledCase:
        p = self._personality() if personality is None else personality
        m = int(self.rng.integers(2, 9)) if m is None else m
        cands = self.location_candidates(m)
        if mutate is not None:
            cands = mutate(cands)
        if history is None:
            k = int(self.rng.integers(0, DEFAULT_CONFIG.K_L + 1))
            history = [self._variant(self._random_location()) for _ in range(k)]
        rel = self._relations(cands, history, DEFAULT_CONFIG.K_L)
        target = self.scorer.distribution(p, cands, relations=rel, level="location")
        return ControlledCase(
            personality=p.vector, decision_type="location", candidates=cands,
            recent_locations=list(history), candidate_history_features=rel,
            target_distribution=target,
        )

    def action_case(self, personality: Personality | None = None,
                    at: Option | None = None,
                    history: list[Option] | None = None) -> ControlledCase:
        p = self._personality() if personality is None else personality
        loc = self._random_location() if at is None else at
        native = self.world.actions_at(base_id(loc.id))
        cands = []
        used: set[str] = set()
        for a in native:
            o = self._variant(a)
            if o.id in used:
                o = self._perturb(o)
            used.add(o.id)
            cands.append(o)
        if history is None:                        # same-location persistence: history
            k = int(self.rng.integers(0, DEFAULT_CONFIG.K_A + 1))   # from native actions
            history = [native[int(self.rng.integers(len(native)))] for _ in range(k)]
        rel = self._relations(cands, history, DEFAULT_CONFIG.K_A)
        target = self.scorer.distribution(p, cands, relations=rel, level="action")
        j = int(self.rng.integers(0, DEFAULT_CONFIG.K_L))           # older entries
        older = [self._variant(self._random_location()) for _ in range(j)]
        return ControlledCase(
            personality=p.vector, decision_type="action", candidates=cands,
            selected_location=loc.id, recent_locations=older + [loc],
            recent_actions_same_location=list(history),
            candidate_history_features=rel, target_distribution=target,
        )

    def _relations(self, cands, history, maxlen):
        if not history:
            return None
        buf = RecentBuffer(maxlen=maxlen)
        for o in history:
            buf.push(o)
        return compute_relations(cands, buf, self.scorer.config)


# -------------------------------------------------------------------- rollout --
def rollout_records(world_name: str, world, scorer: HandAuthoredScorer,
                    n_traj: int, rng: np.random.Generator,
                    rounds: int = ROUNDS_PER_TRAJ) -> list[tuple[ControlledCase, dict]]:
    """Trajectory cases: every decision's inputs + ``trace.P_rule`` (design §2)."""
    records: list[tuple[ControlledCase, dict]] = []
    for _ in range(n_traj):
        p = Personality(rng.uniform(-1.0, 1.0, 5))
        ctrl = DecisionController(
            scorer, config=scorer.config, mode="sample",
            rng=np.random.default_rng(int(rng.integers(2**32))),
        )
        for _ in range(rounds):
            locs = world.resolve()
            h_l = list(reversed(ctrl.H_L.recent_to_old()))     # oldest → newest
            d_loc = ctrl.choose_location(p, locs)
            records.append((
                ControlledCase(
                    personality=p.vector, decision_type="location",
                    candidates=locs, recent_locations=h_l,
                    candidate_history_features=d_loc.trace.relations,
                    target_distribution=d_loc.trace.P_rule,
                ),
                {"source": "rollout", "world": world_name},
            ))
            acts = world.actions_at(d_loc.option.id)
            h_l2 = list(reversed(ctrl.H_L.recent_to_old()))    # now ends with the choice
            h_a = list(reversed(ctrl.H_A.recent_to_old()))
            d_act = ctrl.choose_action(p, acts)
            records.append((
                ControlledCase(
                    personality=p.vector, decision_type="action",
                    candidates=acts, selected_location=d_loc.option.id,
                    recent_locations=h_l2, recent_actions_same_location=h_a,
                    candidate_history_features=d_act.trace.relations,
                    target_distribution=d_act.trace.P_rule,
                ),
                {"source": "rollout", "world": world_name},
            ))
    return records
