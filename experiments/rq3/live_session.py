"""Live demo session — all logic behind the live HTTP server, no HTTP here.

One ``LiveSession`` owns the mutable world state and one
``DecisionController`` per configured NPC. ``state()`` / ``step()`` return
dicts shaped exactly like the ``/state`` and ``/step`` payloads in
``docs/specs/2026-07-17-rq3-sequence-interface-design.md`` (section 5, live
mode): arrays rather than maps so Unity's JsonUtility can parse them, and the
override flag is spelled ``overridden`` because ``override`` is a C# keyword.

World mutation never touches the frozen core dataclasses: entries are swapped
via ``dataclasses.replace`` inside the (mutable) ``World.entries`` dict, and
``reset()`` reloads the authored file. Consistent with the project's
checkpoint principle, an event/unlock change only alters what NPCs see at the
next ``step()`` call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from npc_policy.controller import DecisionController
from npc_policy.policies import LearnedPolicyAdapter
from npc_policy.representation import Personality
from npc_policy.scorer import HandAuthoredScorer
from npc_policy.world import World, load_personalities, load_world

OCEAN_TRAITS = ("openness", "conscientiousness", "extraversion",
                "agreeableness", "neuroticism")


@dataclass
class LiveNpc:
    """One configured NPC: identity + policy + its stateful controller."""

    slot: int
    name: str
    policy_label: str
    personality: Personality
    policy: object                    # scorer or LearnedPolicyAdapter (stateless)
    controller: DecisionController


class LiveSession:
    """Mutable world + per-NPC controllers driving the live demo."""

    def __init__(self, config: dict):
        self.config = config
        self.world: World = load_world(config["world"])
        self.step_count = 0
        self.npcs: list[LiveNpc] = self._build_npcs()

    @classmethod
    def from_config(cls, path: str | Path) -> "LiveSession":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- construction -----------------------------------------------------------
    def _build_npcs(self) -> list[LiveNpc]:
        lookup: dict[str, Personality] = {}
        if self.config.get("personalities"):
            lookup = load_personalities(self.config["personalities"])
        npcs = []
        for slot, entry in enumerate(self.config["npcs"]):
            name = entry["name"]
            if "ocean" in entry:
                personality = Personality.from_traits(**entry["ocean"])
            elif name in lookup:
                personality = lookup[name]
            else:
                raise ValueError(
                    f"npc {name!r}: not in personalities file and no inline ocean")
            label = entry.get("policy", "scorer")
            if label == "scorer":
                policy = HandAuthoredScorer()
            elif "checkpoint" in entry:
                policy = LearnedPolicyAdapter(entry["checkpoint"])
            else:
                raise ValueError(f"npc {name!r}: policy {label!r} needs a checkpoint")
            controller = DecisionController(
                policy,
                mode=self.config.get("mode", "sample"),
                rng=np.random.default_rng([int(self.config.get("seed", 0)), slot]),
                selection_temperature=float(
                    self.config.get("selection_temperature", 1.0)),
            )
            npcs.append(LiveNpc(slot=slot, name=name, policy_label=label,
                                personality=personality, policy=policy,
                                controller=controller))
        return npcs

    # -- world mutation ---------------------------------------------------------
    def _entry(self, location: str):
        try:
            return self.world.entries[location]
        except KeyError:
            raise ValueError(f"no location {location!r}") from None

    def set_event(self, location: str, event: str, active: bool) -> None:
        entry = self._entry(location)
        if all(ev.name != event for ev in entry.events):
            raise ValueError(f"no event {event!r} at {location!r}")
        self.world.entries[location] = replace(entry, events=[
            replace(ev, active=bool(active)) if ev.name == event else ev
            for ev in entry.events
        ])

    def set_unlocked(self, location: str, unlocked: bool) -> None:
        entry = self._entry(location)
        self.world.entries[location] = replace(entry, unlocked=bool(unlocked))

    # -- stepping ---------------------------------------------------------------
    def _active_force(self) -> tuple[str, str] | None:
        """First active force event in location-then-event order, parsed and
        validated against the *current* world state."""
        for loc_id, entry in self.world.entries.items():
            for ev in entry.events:
                if not (ev.active and ev.force_npc):
                    continue
                target_loc, _, target_act = ev.force_npc.partition("/")
                target = self._entry(target_loc)
                if not target.unlocked:
                    raise ValueError(
                        f"force event {ev.name!r} at {loc_id!r} targets locked "
                        f"location {target_loc!r}")
                if all(a.id != target_act for a in target.actions):
                    raise ValueError(
                        f"force event {ev.name!r}: no action {target_act!r} "
                        f"at {target_loc!r}")
                return target_loc, target_act
        return None

    @staticmethod
    def _probs(options, distribution) -> list[dict]:
        return [{"id": o.id, "p": float(p)}
                for o, p in zip(options, distribution, strict=True)]

    def step(self) -> dict:
        """One decision cycle for every NPC (the checkpoint)."""
        force = self._active_force()
        steps = []
        for npc in self.npcs:
            prev = npc.controller.current_location_id
            if force is not None:
                loc_id, act_id = force
                loc = self.world.effective_location(loc_id)
                act = next(a for a in self.world.actions_at(loc_id)
                           if a.id == act_id)
                npc.controller.commit_forced(loc, act)
                steps.append({
                    "slot": npc.slot, "name": npc.name,
                    "location": loc_id, "action": act_id,
                    "moved": loc_id != prev, "overridden": True,
                    "location_probs": [], "action_probs": [],
                })
                continue
            locations = self.world.resolve()
            d_loc = npc.controller.choose_location(npc.personality, locations)
            actions = self.world.actions_at(d_loc.option.id)
            d_act = npc.controller.choose_action(npc.personality, actions)
            steps.append({
                "slot": npc.slot, "name": npc.name,
                "location": d_loc.option.id, "action": d_act.option.id,
                "moved": d_loc.option.id != prev, "overridden": False,
                "location_probs": self._probs(locations, d_loc.distribution),
                "action_probs": self._probs(actions, d_act.distribution),
            })
        self.step_count += 1
        return {"step_count": self.step_count, "steps": steps}

    # -- reporting --------------------------------------------------------------
    def state(self) -> dict:
        return {
            "world": self.config["world"],
            "step_count": self.step_count,
            "locations": [
                {"id": loc_id, "unlocked": entry.unlocked,
                 "events": [
                     {"name": ev.name, "active": ev.active,
                      "force_npc": ev.force_npc or ""}
                     for ev in entry.events
                 ]}
                for loc_id, entry in self.world.entries.items()
            ],
            "npcs": [
                {"slot": npc.slot, "name": npc.name,
                 "policy": npc.policy_label,
                 "ocean": [float(v) for v in npc.personality.vector],
                 "location": npc.controller.current_location_id or ""}
                for npc in self.npcs
            ],
        }

    # -- lifecycle --------------------------------------------------------------
    def reset(self) -> None:
        """Authored world restored, controllers rebuilt, rngs re-seeded —
        a reset session replays a fresh session exactly."""
        self.world = load_world(self.config["world"])
        self.step_count = 0
        self.npcs = self._build_npcs()
