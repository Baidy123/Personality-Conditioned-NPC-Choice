"""World content loader and local-event resolver — the data/IO boundary between
authored content (or, later, a game engine) and the policy.

A location carries its base ``features``, its action set, an ``unlocked`` flag, and
a set of **local events** (see ``location_schema_figure.html``, Listing 1). Each
local event has ``active``, a ``buff`` (temporary feature deltas added while active),
and ``force_npc`` (a scripted override). ``unlocked``, the events, and ``force_npc``
are game-layer fields the model never sees.

When producing candidates, this layer:
  1. drops locations whose ``unlocked`` is false;
  2. for each remaining location, adds every *active* event's ``buff`` to the base
     features (clamped to [0, 1]) to get the **effective** features;
  3. hands the resulting effective ``Option`` (and its actions) to the policy.

Global events (the upper-layer switches that toggle ``unlocked`` / ``active``, e.g.
"war won") are deferred to game-engine integration; for now those flags are set
directly in the JSON. Buffs apply to **location** features only.

JSON shape (features/buff/ocean are name->value dicts; unspecified entries -> 0):

    { "locations": [
        { "id": "tavern", "features": {...}, "unlocked": true,
          "actions": [ { "id": "chat", "features": {...} }, ... ],
          "events":  [ { "name": "celebration", "active": true,
                         "buff": {"social": 0.1, ...}, "force_npc": null } ] },
        ... ] }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .representation import Option, Personality
from .schema import tag_index


@dataclass(frozen=True)
class LocalEvent:
    """A game-layer local event attached to one location. Model never sees it."""

    name: str
    active: bool
    buff: dict[str, float]            # location feature name -> delta
    force_npc: str | None = None


@dataclass(frozen=True)
class LocationEntry:
    """Authored state for one location (base features + game-layer fields)."""

    base: Option                      # base (unbuffed) location features
    actions: list[Option]
    unlocked: bool = True
    events: list[LocalEvent] = field(default_factory=list)

    def effective(self) -> Option:
        """Base features + every active event's buff, clamped to [0, 1]."""
        feats = self.base.features.copy()
        for ev in self.events:
            if ev.active:
                for name, delta in ev.buff.items():
                    feats[tag_index("location", name)] += delta
        return Option(id=self.base.id, features=np.clip(feats, 0.0, 1.0), level="location")


@dataclass(frozen=True)
class World:
    """Authored content: locations (with game-layer state) and their action sets."""

    entries: dict[str, LocationEntry]

    def location_ids(self) -> list[str]:
        """All location ids, including currently locked ones."""
        return list(self.entries)

    def resolve(self) -> list[Option]:
        """Current candidate locations: unlocked only, with effective features.

        This is what a policy receives as its location candidate set.
        """
        return [e.effective() for e in self.entries.values() if e.unlocked]

    def effective_location(self, location_id: str) -> Option:
        """Effective (buffed) features for one location; errors if it is locked."""
        e = self._entry(location_id)
        if not e.unlocked:
            raise KeyError(f"location {location_id!r} is locked")
        return e.effective()

    def actions_at(self, location_id: str) -> list[Option]:
        """Action set of a location (actions are not buffed; buffs are location-only)."""
        return list(self._entry(location_id).actions)

    def _entry(self, location_id: str) -> LocationEntry:
        try:
            return self.entries[location_id]
        except KeyError:
            raise KeyError(f"no location {location_id!r} in world") from None


def _parse_event(e: dict) -> LocalEvent:
    buff = e.get("buff", {})
    for name in buff:                                  # validate buff targets location tags
        tag_index("location", name)
    return LocalEvent(
        name=e["name"],
        active=bool(e.get("active", False)),
        buff={k: float(v) for k, v in buff.items()},
        force_npc=e.get("force_npc"),
    )


def load_world(path: str | Path) -> World:
    """Read ``world.json`` into a :class:`World`.

    Option construction validates feature dimensions, the ``[0, 1]`` range, and that
    a location uses only location tags / an action only action tags. Event buffs are
    validated to target location tags.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries: dict[str, LocationEntry] = {}
    for loc in data["locations"]:
        loc_id = loc["id"]
        if loc_id in entries:
            raise ValueError(f"duplicate location id {loc_id!r}")
        entries[loc_id] = LocationEntry(
            base=Option.location(loc_id, **loc.get("features", {})),
            actions=[
                Option.action(a["id"], **a.get("features", {}))
                for a in loc.get("actions", [])
            ],
            unlocked=bool(loc.get("unlocked", True)),
            events=[_parse_event(e) for e in loc.get("events", [])],
        )
    return World(entries=entries)


def load_personalities(path: str | Path) -> dict[str, Personality]:
    """Read ``personalities.json`` into ``{name: Personality}``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        p["name"]: Personality.from_traits(**p.get("ocean", {}))
        for p in data["personalities"]
    }
