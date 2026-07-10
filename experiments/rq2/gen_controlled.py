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

# Pool sizes. Master pool ≈ 190k so every filtered split retains ≥ 105k
# (TRAIN_SIZE + VAL_SIZE); the arena_locked block only feeds G6 (design §2).
# arena_locked was raised 300 → 500 trajectories after measuring G6 eligibility:
# rollout location cases in arena-unlocked worlds always list arena as a
# candidate (0% eligible), so the plan's 300 left G6 ≈ 12k short of 105k.
FULL_SIZES = dict(
    n_syn_loc=49_000, n_syn_act=49_000,
    n_traj={"full": 210, "celebration": 70, "war_camp": 70, "market_locked": 70,
            "arena_locked": 500},
    train=TRAIN_SIZE, val=VAL_SIZE, n_test=TEST_SIZE, rounds=ROUNDS_PER_TRAJ,
)
SMOKE_SIZES = dict(
    n_syn_loc=700, n_syn_act=700,
    n_traj={"full": 3, "celebration": 1, "war_camp": 1, "market_locked": 1,
            "arena_locked": 8},   # 4 → 8: same G6 shortfall at smoke scale
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


# ------------------------------------------------------------- split filters --
def _max_run(ids: list[str]) -> int:
    best = run = 0
    prev = None
    for x in ids:
        run = run + 1 if x == prev else 1
        prev = x
        best = max(best, run)
    return best


def g1_region(case: ControlledCase) -> bool:
    """Excluded personality region: O > 0.5 ∧ C < −0.5 (research spec §6)."""
    return case.personality[0] > 0.5 and case.personality[1] < -0.5


def g2_combo(case: ControlledCase) -> bool:
    """Location candidate with risk > 0.6 ∧ privacy > 0.6 present."""
    return case.decision_type == "location" and any(
        o.tag("risk") > 0.6 and o.tag("privacy") > 0.6 for o in case.candidates
    )


def g3_train_ok(case: ControlledCase) -> bool:
    """Train on location sets of 3–6 only; action cases are unaffected."""
    return case.decision_type != "location" or 3 <= len(case.candidates) <= 6


def g5_has_3run(case: ControlledCase) -> bool:
    """Three consecutive same-family entries in the relevant same-type buffer."""
    buf = (case.recent_locations if case.decision_type == "location"
           else case.recent_actions_same_location)
    return _max_run([base_id(o.id) for o in buf]) >= 3


def g6_touches_arena(case: ControlledCase) -> bool:
    ids = [base_id(o.id) for o in case.candidates]
    ids += [base_id(o.id) for o in case.recent_locations]
    return "arena" in ids or case.selected_location == "arena"


def _core(tags: dict) -> bool:
    """arena_locked top-up cases feed G6 only."""
    return tags.get("world") != "arena_locked"


TRAIN_FILTERS = {
    "S0": lambda c, t: _core(t),
    "G1": lambda c, t: _core(t) and not g1_region(c),
    "G2": lambda c, t: _core(t) and not g2_combo(c),
    "G3": lambda c, t: _core(t) and g3_train_ok(c),
    "G4": lambda c, t: t.get("world") == "full",
    "G5": lambda c, t: _core(t) and not g5_has_3run(c),
    "G6": lambda c, t: not g6_touches_arena(c),
}


# ------------------------------------------------------- targeted test sets --
def targeted_records(split: str, sampler: SyntheticSampler, worlds: dict,
                     scorer: HandAuthoredScorer, n: int,
                     rng: np.random.Generator, rounds: int) -> list[tuple[ControlledCase, dict]]:
    """Generate ``n`` cases satisfying ``split``'s held-out condition (design §2)."""
    tag = {"source": "targeted", "world": "full"}
    out: list[tuple[ControlledCase, dict]] = []

    if split == "S0":
        raise ValueError("S0's test set is a pool holdout, not targeted")

    if split == "G1":
        def region_p():
            v = rng.uniform(-1.0, 1.0, 5)
            v[0] = rng.uniform(0.5 + 1e-9, 1.0)
            v[1] = rng.uniform(-1.0, -0.5 - 1e-9)
            return Personality(v)
        while len(out) < n:
            case = (sampler.location_case(personality=region_p())
                    if rng.random() < 0.5 else sampler.action_case(personality=region_p()))
            out.append((case, dict(tag)))

    elif split == "G2":
        risk_i = tag_index("location", "risk")
        priv_i = tag_index("location", "privacy")

        def spike(cands: list[Option]) -> list[Option]:
            k = int(rng.integers(len(cands)))
            f = cands[k].features.copy()
            f[risk_i] = rng.uniform(0.65, 1.0)
            f[priv_i] = rng.uniform(0.65, 1.0)
            cands[k] = Option(id=cands[k].id, features=f, level="location")
            return cands
        while len(out) < n:
            out.append((sampler.location_case(mutate=spike), dict(tag)))

    elif split == "G3":
        while len(out) < n:
            m = 2 if rng.random() < 0.5 else 8
            out.append((sampler.location_case(m=m), dict(tag)))

    elif split == "G4":
        variants = ("celebration", "war_camp", "market_locked")
        i = 0
        while len(out) < n:
            name = variants[i % 3]
            i += 1
            world = load_world(worlds[name])
            recs = rollout_records(name, world, scorer, n_traj=1, rng=rng, rounds=rounds)
            for case, t in recs:
                t["source"] = "targeted"
                out.append((case, t))
        out = out[:n]

    elif split == "G5":
        while len(out) < n:
            if rng.random() < 0.5:
                x = sampler._variant(sampler._random_location())
                out.append((sampler.location_case(history=[x, x, x]), dict(tag)))
            else:
                loc = sampler._random_location()
                a = sampler.world.actions_at(base_id(loc.id))[0]
                out.append((sampler.action_case(at=loc, history=[a, a, a]), dict(tag)))

    elif split == "G6":
        arena = sampler.world.effective_location("arena")

        def force_arena(cands: list[Option]) -> list[Option]:
            if not any(base_id(o.id) == "arena" for o in cands):
                cands[int(rng.integers(len(cands)))] = arena
            return cands
        while len(out) < n:
            case = (sampler.location_case(mutate=force_arena)
                    if rng.random() < 0.5 else sampler.action_case(at=arena))
            out.append((case, dict(tag)))

    else:
        raise ValueError(f"unknown split {split!r}")
    return out


# ------------------------------------------------------------------ assembly --
def generate(sizes: dict, out_dir: Path, seed: int = GEN_SEED) -> dict:
    """Full generation pass; returns the meta dict (also written to meta.json)."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    worlds = ensure_worlds()
    scorer = HandAuthoredScorer()            # DEFAULT_CONFIG — the frozen teacher
    base_world = load_world(worlds["full"])

    # -- master pool ---------------------------------------------------------
    rng = np.random.default_rng([seed, 0])
    sampler = SyntheticSampler(base_world, scorer, rng)
    records: list[tuple[ControlledCase, dict]] = []
    for _ in range(sizes["n_syn_loc"]):
        records.append((sampler.location_case(), {"source": "synthetic", "world": "full"}))
    for _ in range(sizes["n_syn_act"]):
        records.append((sampler.action_case(), {"source": "synthetic", "world": "full"}))
    for w_idx, (name, n_traj) in enumerate(sorted(sizes["n_traj"].items())):
        world = load_world(worlds[name])
        records += rollout_records(name, world, scorer, n_traj,
                                   np.random.default_rng([seed, 1, w_idx]),
                                   rounds=sizes["rounds"])
    for i, (_, tags) in enumerate(records):
        tags["id"] = f"{tags['source'][:4]}-{i:07d}"

    # -- S0 holdout + split manifests -----------------------------------------
    core_ids = [t["id"] for c, t in records if _core(t)]
    hold_rng = np.random.default_rng([seed, 2])
    hold_rng.shuffle(core_ids)
    s0_test_ids = set(core_ids[: sizes["n_test"]])

    by_id = {t["id"]: (c, t) for c, t in records}
    splits: dict[str, dict] = {}
    for k, split in enumerate(ALL_SPLITS):
        elig = [t["id"] for c, t in records
                if t["id"] not in s0_test_ids and TRAIN_FILTERS[split](c, t)]
        need = sizes["train"] + sizes["val"]
        if len(elig) < need:
            raise RuntimeError(
                f"{split}: filtered pool has {len(elig)} cases, needs {need}; "
                "increase generation sizes"
            )
        srng = np.random.default_rng([seed, 3, k])
        srng.shuffle(elig)
        splits[split] = {"train": elig[: sizes["train"]],
                         "val": elig[sizes["train"]: need]}

    # -- write ----------------------------------------------------------------
    write_pool(out_dir / "pool.jsonl", records)
    write_pool(out_dir / "test_S0.jsonl",
               [by_id[i] for i in core_ids[: sizes["n_test"]]])
    trng = np.random.default_rng([seed, 4])
    test_counts = {"S0": sizes["n_test"]}
    for split in ALL_SPLITS[1:]:
        recs = targeted_records(split, sampler, worlds, scorer,
                                sizes["n_test"], trng, sizes["rounds"])
        write_pool(out_dir / f"test_{split}.jsonl", recs)
        test_counts[split] = len(recs)
    (out_dir / "splits.json").write_text(
        json.dumps({"s0_test_ids": sorted(s0_test_ids), "splits": splits}),
        encoding="utf-8",
    )
    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "config_hash": config_hash(),
        "pool_cases": len(records),
        "per_split": {s: {"train": len(splits[s]["train"]), "val": len(splits[s]["val"]),
                          "test": test_counts[s]} for s in ALL_SPLITS},
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Generate the Study 2A controlled dataset")
    ap.add_argument("--smoke", action="store_true", help="small isolated end-to-end pass")
    args = ap.parse_args(argv)
    out_dir, _ = dirs(args.smoke)
    meta = generate(SMOKE_SIZES if args.smoke else FULL_SIZES, out_dir)
    print(f"written: {out_dir}")
    print(f"  pool cases: {meta['pool_cases']}, teacher config: {meta['config_hash'][:12]}…")
    for s, n in meta["per_split"].items():
        print(f"  {s}: train {n['train']} / val {n['val']} / test {n['test']}")
    print(f"  elapsed: {meta['elapsed_s']}s")


if __name__ == "__main__":
    main()
