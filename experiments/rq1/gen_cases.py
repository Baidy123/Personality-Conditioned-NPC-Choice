"""Matched-case generator for RQ1 (project_flow.md §10 step 2).

Produces ``data/rq1_cases/``:

  worlds/*.json   — candidate-set variants derived from data/world.json
  cases.json      — personality profiles + decision contexts

A *matched case* fixes everything except personality: one world variant, one
memory state, one decision level. Every RQ1 analysis compares profiles inside
the same context, so behavioural differences are attributable to personality
alone.

Run from ``code/``:  python -m experiments.rq1.gen_cases
"""

from __future__ import annotations

import copy
import json
from datetime import date

import numpy as np

from npc_policy import load_personalities, load_world

from .common import CASES, DATA, TRAITS

GEN_SEED = 20260703       # random-profile seed; fixed for reproducibility
N_RANDOM = 300
SWEEP = [round(v, 2) for v in np.linspace(-1.0, 1.0, 9)]

# World variants: the base world plus the three kinds of change the game layer
# can make (unlock, lock, event buff). Applied to the raw JSON so the variant
# files are self-contained and reviewable.
WORLD_VARIANTS = {
    "full": [],
    "war_camp": [("unlock", "enemy_camp")],
    "celebration": [("activate_event", "tavern", "celebration")],
    "market_locked": [("lock", "market")],
}

# Location-level memory states (ids oldest -> newest), all on the "full" world;
# the other variants are probed with empty memory.
LOCATION_MEMORY = {
    "empty": [],
    "rep1_tavern": ["tavern"],
    "rep2_tavern": ["tavern", "tavern"],
    "rep3_arena": ["arena", "arena", "arena"],
    "mixed_social": ["market", "tavern", "arena"],
    "mixed_quiet": ["library", "chapel", "forest"],
}

# Action-level memory states per location: a0/a1 = the location's first two
# actions. Only meaningful across consecutive same-location cycles.
ACTION_MEMORY = ("empty", "rep1_a0", "rep2_a0", "mix_a0_a1")


def build_worlds() -> None:
    base = json.loads((DATA / "world.json").read_text(encoding="utf-8"))
    (CASES / "worlds").mkdir(parents=True, exist_ok=True)
    for name, mods in WORLD_VARIANTS.items():
        j = copy.deepcopy(base)
        for mod in mods:
            for loc in j["locations"]:
                if mod[0] == "unlock" and loc["id"] == mod[1]:
                    loc["unlocked"] = True
                elif mod[0] == "lock" and loc["id"] == mod[1]:
                    loc["unlocked"] = False
                elif mod[0] == "activate_event" and loc["id"] == mod[1]:
                    for ev in loc.get("events", []):
                        if ev["name"] == mod[2]:
                            ev["active"] = True
        (CASES / "worlds" / f"{name}.json").write_text(
            json.dumps(j, indent=2), encoding="utf-8"
        )


def build_profiles() -> dict:
    sweep = []
    for trait in TRAITS:
        for v in SWEEP:
            vec = [0.0] * 5
            vec[TRAITS.index(trait)] = v
            sweep.append({"id": f"{trait[:4]}@{v:+.2f}", "trait": trait,
                          "value": v, "vector": vec})
    rng = np.random.default_rng(GEN_SEED)
    rand = [{"id": f"rand{i:03d}", "vector": [round(x, 4) for x in row]}
            for i, row in enumerate(rng.uniform(-1.0, 1.0, size=(N_RANDOM, 5)))]
    named = [{"id": name, "vector": list(p.vector)}
             for name, p in load_personalities(DATA / "personalities.json").items()]
    return {"sweep": sweep, "random": rand, "named": named,
            "neutral": {"id": "neutral", "vector": [0.0] * 5}}


def build_contexts() -> tuple[list[dict], list[dict]]:
    loc_ctx = [{"id": f"loc:full/{mem}", "world": "full", "memory": ids}
               for mem, ids in LOCATION_MEMORY.items()]
    loc_ctx += [{"id": f"loc:{w}/empty", "world": w, "memory": []}
                for w in WORLD_VARIANTS if w != "full"]

    world = load_world(CASES / "worlds" / "full.json")
    act_ctx = []
    for loc in world.resolve():
        acts = [a.id for a in world.actions_at(loc.id)]
        a0, a1 = acts[0], acts[1]
        histories = {"empty": [], "rep1_a0": [a0], "rep2_a0": [a0, a0],
                     "mix_a0_a1": [a0, a1]}
        for mem in ACTION_MEMORY:
            act_ctx.append({"id": f"act:{loc.id}/{mem}", "world": "full",
                            "location": loc.id, "memory": histories[mem]})
    return loc_ctx, act_ctx


def main() -> None:
    build_worlds()
    profiles = build_profiles()
    loc_ctx, act_ctx = build_contexts()
    cases = {
        "meta": {
            "generated": str(date.today()),
            "seed": GEN_SEED,
            "n_profiles": {"sweep": len(profiles["sweep"]),
                           "random": len(profiles["random"]),
                           "named": len(profiles["named"])},
            "n_contexts": {"location": len(loc_ctx), "action": len(act_ctx)},
            "note": "memory lists are option ids oldest -> newest",
        },
        "profiles": profiles,
        "location_contexts": loc_ctx,
        "action_contexts": act_ctx,
    }
    out = CASES / "cases.json"
    out.write_text(json.dumps(cases, indent=1), encoding="utf-8")
    print(f"written: {out}")
    print(f"  profiles: {cases['meta']['n_profiles']}")
    print(f"  contexts: {cases['meta']['n_contexts']}")


if __name__ == "__main__":
    main()
