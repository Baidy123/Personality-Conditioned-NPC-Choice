"""End-to-end demo — everything comes from the world/data files, nothing hardcoded.

Run from the ``code`` folder:
    python -m examples.demo            # 1 round per NPC, random each run
    python -m examples.demo 5          # 5 rounds per NPC
    python -m examples.demo 5 42       # 5 rounds, fixed seed 42 (reproducible)

``rounds`` must be an integer >= 1 (a non-integer or value < 1 is rejected). The
optional ``seed`` makes the sampled trajectory reproducible; without it each run
differs.

World content (locations, action sets, unlocked flags, local events) and NPCs come
from ``code/data/*.json``. Candidate locations are whatever the world currently makes
available (``World.resolve()``), so editing the JSON changes the demo with no code
edit. Across rounds the controller keeps each NPC's recent-choice memory, so later
choices reflect repetition / novelty / similarity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from npc_policy import (
    DecisionController,
    HandAuthoredScorer,
    load_personalities,
    load_world,
)
from npc_policy.schema import LOCATION_TAGS

DATA = Path(__file__).resolve().parent.parent / "data"


def _fmt(options, dist) -> str:
    pairs = sorted(zip(options, dist), key=lambda t: -t[1])
    return "  ".join(f"{o.id}:{p:.2f}" for o, p in pairs)


def main(rounds: int = 1, seed: int | None = None) -> None:
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError(f"rounds must be an integer >= 1, got {rounds!r}")

    np.set_printoptions(precision=3, suppress=True)
    world = load_world(DATA / "world.json")
    npcs = load_personalities(DATA / "personalities.json")
    scorer = HandAuthoredScorer()

    # --- 0. What the world currently offers (locked dropped, buffs applied) -----
    candidates = world.resolve()
    print("All locations in file :", world.location_ids())
    print("Currently available   :", [o.id for o in candidates])

    print("Active local-event buffs:")
    any_active = False
    for loc_id, e in world.entries.items():
        active = [ev.name for ev in e.events if ev.active]
        if not (e.unlocked and active):
            continue
        any_active = True
        base, eff = e.base.features, e.effective().features
        diffs = ", ".join(
            f"{t} {base[i]:.2f}->{eff[i]:.2f}"
            for i, t in enumerate(LOCATION_TAGS)
            if base[i] != eff[i]
        )
        print(f"   {loc_id} [{', '.join(active)}]: {diffs}")
    if not any_active:
        print("   (none)")
    print(f"\nRunning {rounds} round(s) per NPC.\n")

    # --- 1. Each NPC: personality preference + a multi-round trajectory ---------
    for name, npc in npcs.items():
        # round-1 location preference (the personality signature, empty memory)
        loc_dist = scorer.distribution(npc, candidates, level="location")
        print(f"{name}")
        print(f"   location pref -> {_fmt(candidates, loc_dist)}")
        # trajectory: the controller carries H_L / H_A across rounds. Sampling is
        # sharpened (selection_temperature < 1) so the low-probability tail is
        # suppressed without being removed.
        ctrl = DecisionController(
            scorer, mode="sample", rng=np.random.default_rng(seed), selection_temperature=0.1
        )
        for t in range(rounds):
            loc = ctrl.choose_location(npc, candidates)
            act = ctrl.choose_action(npc, world.actions_at(loc.option.id))
            recent = [o.id for o in ctrl.H_L.recent_to_old()]
            print(f"   round {t + 1}: {loc.option.id:13} -> {act.option.id:12} | H_L={recent}")
        print()

    # --- 2. Self-check: the action buffer resets on a location change -----------
    a_id, b_id = candidates[0].id, candidates[1].id
    npc = next(iter(npcs.values()))
    ctrl = DecisionController(scorer, mode="argmax")
    ctrl.choose_location(npc, [world.effective_location(a_id)])
    ctrl.choose_action(npc, world.actions_at(a_id))
    assert len(ctrl.H_A) == 1
    ctrl.choose_location(npc, [world.effective_location(b_id)])     # forced move
    assert len(ctrl.H_A) == 0
    first = ctrl.choose_action(npc, world.actions_at(b_id))
    assert np.allclose(first.trace.P_rule, first.trace.P_base)
    print(f"self-check: move {a_id} -> {b_id} cleared H_A; first action used base dist [OK]")


def _parse_rounds(argv: list[str]) -> int:
    if len(argv) <= 1:
        return 1
    raw = argv[1]
    try:
        rounds = int(raw)
    except ValueError:
        raise SystemExit(f"error: rounds must be an integer, got {raw!r}")
    if rounds < 1:
        raise SystemExit(f"error: rounds must be >= 1, got {rounds}")
    return rounds


def _parse_seed(argv: list[str]) -> int | None:
    if len(argv) <= 2:
        return None                              # no seed -> different every run
    raw = argv[2]
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"error: seed must be an integer, got {raw!r}")


if __name__ == "__main__":
    main(_parse_rounds(sys.argv), _parse_seed(sys.argv))
