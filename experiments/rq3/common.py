"""RQ3 sequence-generation core: one (personality, policy, world, seed) spec ->
one replayable sequence dict (spec: docs/specs/2026-07-17-rq3-sequence-interface-design.md).

The sequence dict is exactly what is written to JSON for the Unity playback
player. ``location_probs`` / ``action_probs`` are research-archive fields the
player ignores. No time information is stored: playback pacing is a player
concern (continue button / auto-advance)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from npc_policy.controller import DecisionController
from npc_policy.representation import Personality
from npc_policy.world import World

OCEAN_KEYS = ("O", "C", "E", "A", "N")


@dataclass(frozen=True)
class SequenceSpec:
    """Generation conditions for one sequence (everything the manifest records)."""

    sequence_id: str
    policy_name: str
    checkpoint: str           # "" for the hand-authored scorer
    personality_name: str
    personality: Personality
    world_path: str
    n_cycles: int
    seed: int


def generate_sequence(spec: SequenceSpec, policy, world: World) -> dict:
    """Roll one NPC for ``n_cycles`` decision cycles and return the sequence dict.

    Same seed + same spec -> identical steps (the controller's rng is the only
    randomness; learned adapters run in eval mode)."""
    ctrl = DecisionController(policy, mode="sample",
                              rng=np.random.default_rng(spec.seed))
    steps: list[dict] = []
    prev: str | None = None
    for cycle in range(1, spec.n_cycles + 1):
        locs = world.resolve()
        d_loc = ctrl.choose_location(spec.personality, locs)
        acts = world.actions_at(d_loc.option.id)
        d_act = ctrl.choose_action(spec.personality, acts)
        steps.append({
            "cycle": cycle,
            "location": d_loc.option.id,
            "action": d_act.option.id,
            "moved": d_loc.option.id != prev,
            "location_probs": {o.id: float(p) for o, p in zip(locs, d_loc.distribution)},
            "action_probs": {o.id: float(p) for o, p in zip(acts, d_act.distribution)},
        })
        prev = d_loc.option.id
    return {
        "meta": {
            "sequence_id": spec.sequence_id,
            "policy": spec.policy_name,
            "checkpoint": spec.checkpoint,
            "personality_name": spec.personality_name,
            "ocean": dict(zip(OCEAN_KEYS, (float(v) for v in spec.personality.vector))),
            "world": spec.world_path,
            "seed": spec.seed,
            "n_cycles": spec.n_cycles,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "steps": steps,
    }


def validate_sequence(seq: dict, world: World) -> None:
    """Export gate: raise ValueError rather than let a bad file reach Unity."""
    sid = seq["meta"]["sequence_id"]
    steps = seq["steps"]
    if len(steps) != seq["meta"]["n_cycles"]:
        raise ValueError(f"{sid}: {len(steps)} steps != n_cycles {seq['meta']['n_cycles']}")
    for step in steps:
        loc = step["location"]
        if loc not in world.location_ids() or not world.entries[loc].unlocked:
            raise ValueError(f"{sid} cycle {step['cycle']}: unknown/locked location {loc!r}")
        action_ids = {a.id for a in world.actions_at(loc)}
        if step["action"] not in action_ids:
            raise ValueError(
                f"{sid} cycle {step['cycle']}: action {step['action']!r} not at {loc!r}")
        for key in ("location_probs", "action_probs"):
            total = sum(step[key].values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"{sid} cycle {step['cycle']}: {key} sum {total} != 1")


def format_preview(seq: dict) -> str:
    """One text line per sequence for pre-recording quality control."""
    m = seq["meta"]
    trail = " -> ".join(f"{s['location']}/{s['action']}" for s in seq["steps"])
    return (f"{m['sequence_id']} [{m['policy']} | {m['personality_name']} | "
            f"seed {m['seed']}]: {trail}")
