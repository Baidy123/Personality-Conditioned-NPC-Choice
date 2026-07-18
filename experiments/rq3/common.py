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
    # sampling sharpness (DecisionController): 1.0 = raw policy distribution;
    # 0.1 = the sharp, in-character setting the tuning-log personas were
    # validated at. A stimulus-design knob — recorded in meta and manifest.
    selection_temperature: float = 1.0


def generate_sequence(spec: SequenceSpec, policy, world: World) -> dict:
    """Roll one NPC for ``n_cycles`` decision cycles and return the sequence dict.

    Same seed + same spec -> identical steps (the controller's rng is the only
    randomness; learned adapters run in eval mode)."""
    ctrl = DecisionController(policy, mode="sample",
                              rng=np.random.default_rng(spec.seed),
                              selection_temperature=spec.selection_temperature)
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
            "location_probs": {o.id: float(p)
                               for o, p in zip(locs, d_loc.distribution, strict=True)},
            "action_probs": {o.id: float(p)
                             for o, p in zip(acts, d_act.distribution, strict=True)},
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
            "selection_temperature": spec.selection_temperature,
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
    all_loc_ids = set(world.location_ids())
    prev: str | None = None
    for step in steps:
        cyc = step["cycle"]
        loc = step["location"]
        if loc not in all_loc_ids or not world.entries[loc].unlocked:
            raise ValueError(f"{sid} cycle {cyc}: unknown/locked location {loc!r}")
        action_ids = {a.id for a in world.actions_at(loc)}
        if step["action"] not in action_ids:
            raise ValueError(
                f"{sid} cycle {cyc}: action {step['action']!r} not at {loc!r}")
        if bool(step["moved"]) != (loc != prev):
            raise ValueError(
                f"{sid} cycle {cyc}: moved={step['moved']} inconsistent with previous "
                f"location {prev!r}")
        for key, known in (("location_probs", all_loc_ids), ("action_probs", action_ids)):
            unknown = set(step[key]) - known
            if unknown:
                raise ValueError(f"{sid} cycle {cyc}: unknown ids in {key}: {sorted(unknown)}")
            vals = list(step[key].values())
            if any(v < 0 for v in vals):
                raise ValueError(f"{sid} cycle {cyc}: negative probability in {key}")
            total = sum(vals)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"{sid} cycle {cyc}: {key} sum {total} != 1")
        if loc not in step["location_probs"] or step["action"] not in step["action_probs"]:
            raise ValueError(f"{sid} cycle {cyc}: chosen option missing from its probs")
        prev = loc


def format_preview(seq: dict) -> str:
    """One text line per sequence for pre-recording quality control."""
    m = seq["meta"]
    trail = " -> ".join(f"{s['location']}/{s['action']}" for s in seq["steps"])
    return (f"{m['sequence_id']} [{m['policy']} | {m['personality_name']} | "
            f"seed {m['seed']}]: {trail}")
