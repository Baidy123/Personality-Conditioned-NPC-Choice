"""Batch-generate RQ3 behaviour sequences from a config file.

    python -m experiments.rq3.gen_sequences --config experiments/rq3/config_smoke.json --preview

Run from ``code/``. Config JSON:

    { "out_dir": "data/rq3_sequences",
      "worlds": ["data/world.json"],
      "personalities": [ {"name": "high_E", "ocean": {"extraversion": 1.0}}, ... ],
      "policies":      [ {"name": "scorer"},
                         {"name": "nonlinear_2b",
                          "checkpoint": "results/rq2b/models/<file>.pt"}, ... ],
      "n_cycles": 10,
      "seeds": [42] }

Sequences are crossed in world -> personality -> policy -> seed order and
numbered S01, S02, ... deterministically; ``manifest.csv`` in ``out_dir`` is the
stimulus ledger (one row per sequence). Files are permanent research artefacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from npc_policy.policies import LearnedPolicyAdapter
from npc_policy.representation import Personality
from npc_policy.scorer import HandAuthoredScorer
from npc_policy.world import load_world

from .common import SequenceSpec, format_preview, generate_sequence, validate_sequence

MANIFEST_FIELDS = ["sequence_id", "file", "policy", "checkpoint",
                   "personality", "world", "seed", "n_cycles", "generated_at"]


def build_policy(entry: dict):
    """'scorer' -> hand-authored scorer; anything else needs a checkpoint path."""
    if entry["name"] == "scorer":
        return HandAuthoredScorer()
    if "checkpoint" not in entry:
        raise ValueError(f"policy {entry['name']!r} needs a 'checkpoint' path")
    return LearnedPolicyAdapter(entry["checkpoint"])


def run_config(config_path: str | Path, preview: bool = False) -> int:
    """Generate every configured sequence; returns the number written."""
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    policies = [(e, build_policy(e)) for e in cfg["policies"]]

    rows: list[dict] = []
    i = 0
    for world_path in cfg["worlds"]:
        world = load_world(world_path)
        for pers in cfg["personalities"]:
            personality = Personality.from_traits(**pers.get("ocean", {}))
            for entry, policy in policies:
                for seed in cfg["seeds"]:
                    i += 1
                    spec = SequenceSpec(
                        sequence_id=f"S{i:02d}",
                        policy_name=entry["name"],
                        checkpoint=entry.get("checkpoint", ""),
                        personality_name=pers["name"],
                        personality=personality,
                        world_path=str(world_path),
                        n_cycles=int(cfg["n_cycles"]),
                        seed=int(seed),
                    )
                    seq = generate_sequence(spec, policy, world)
                    validate_sequence(seq, world)
                    fname = f"{spec.sequence_id}.json"
                    (out_dir / fname).write_text(
                        json.dumps(seq, indent=2), encoding="utf-8")
                    m = seq["meta"]
                    rows.append({
                        "sequence_id": m["sequence_id"], "file": fname,
                        "policy": m["policy"], "checkpoint": m["checkpoint"],
                        "personality": m["personality_name"], "world": m["world"],
                        "seed": m["seed"], "n_cycles": m["n_cycles"],
                        "generated_at": m["generated_at"],
                    })
                    if preview:
                        print(format_preview(seq))

    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {i} sequences + manifest.csv to {out_dir}")
    return i


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--preview", action="store_true",
                    help="print each sequence as one text line for QC")
    args = ap.parse_args()
    run_config(args.config, preview=args.preview)


if __name__ == "__main__":
    main()
