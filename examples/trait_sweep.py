"""Trait-sweep sanity check — the phase-2 debugging entry point.

For each OCEAN trait, sweep its value over {-1, -0.5, 0, +0.5, +1} with all other
traits at 0, and print the scorer's choice distribution (empty memory) for the
current world's locations and for one location's action set. Eyeball check: does
each trait move the distribution in the intended direction?

Run from the ``code`` folder:
    python -m examples.trait_sweep
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from npc_policy import HandAuthoredScorer, load_world
from npc_policy.representation import Option, Personality
from npc_policy.schema import OCEAN

DATA = Path(__file__).resolve().parent.parent / "data"
SWEEP = (-1.0, -0.5, 0.0, 0.5, 1.0)


def sweep_level(
    scorer: HandAuthoredScorer, candidates: list[Option], level: str, header: str
) -> None:
    print(header)
    print(" " * 26 + "".join(f"{o.id:>14}" for o in candidates))
    for trait_pos, trait in enumerate(OCEAN):
        for v in SWEEP:
            vec = np.zeros(5)
            vec[trait_pos] = v
            dist = scorer.distribution(Personality(vec), candidates, level=level)
            row = "".join(f"{p:14.3f}" for p in dist)
            print(f"{trait:>20} {v:+4.1f} {row}")
        print()


def main() -> None:
    world = load_world(DATA / "world.json")
    scorer = HandAuthoredScorer()
    locations = world.resolve()
    sweep_level(scorer, locations, "location",
                "=== location choice (empty memory) ===")
    first = locations[0]
    sweep_level(scorer, world.actions_at(first.id), "action",
                f"=== action choice at {first.id!r} (empty memory) ===")


if __name__ == "__main__":
    main()
