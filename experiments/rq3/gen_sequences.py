"""Batch-generate RQ3 behaviour sequences from a config file.

    python -m experiments.rq3.gen_sequences [--config configs/rq3_sequences.json] [--preview]

Run from ``code/`` (the default config is the user-facing entry point — see
``configs/README.md`` for the "change X -> edit Y" map). Config JSON:

    { "out_dir": "data/rq3_sequences",
      "unity_dir": "unity/dissertation/Assets/StreamingAssets/sequences",
      "worlds": ["data/world.json"],
      "personalities_file": "data/personalities.json",
      "personalities": [ {"name": "Karlach"},                       <- roster lookup
                         {"name": "high_E", "ocean": {"extraversion": 1.0}},
                         {"name": "H1", "ocean": {...}, "seeds": [42]} ],   <- per-entry seeds
      "policies":      [ {"name": "scorer"},
                         {"name": "nonlinear_2b",
                          "checkpoint": "results/rq2b/models/<file>.pt"}, ... ],
      "n_cycles": 10,
      "seeds": [42] }

A personality entry with an inline ``ocean`` dict stands alone; a name-only
entry is looked up in ``personalities_file``. An entry may override the global
``seeds`` with its own. ``unity_dir`` (optional) mirrors
the written ``S*.json`` into the Unity playback folder after generation, with
the same stale-file cleanup, so no manual copy step exists.

Sequences are crossed in world -> personality -> policy -> seed order and
numbered S001, S002, ... deterministically; ``manifest.csv`` in ``out_dir`` is
the stimulus ledger (one row per sequence). Files are permanent research
artefacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from npc_policy.policies import LearnedPolicyAdapter
from npc_policy.representation import Personality
from npc_policy.scorer import HandAuthoredScorer
from npc_policy.world import load_personalities, load_world

from .common import SequenceSpec, format_preview, generate_sequence, validate_sequence

DEFAULT_CONFIG_PATH = "configs/rq3_sequences.json"

MANIFEST_FIELDS = ["sequence_id", "file", "policy", "checkpoint",
                   "personality", "world", "seed", "n_cycles",
                   "selection_temperature", "generated_at"]


def build_policy(entry: dict):
    """'scorer' -> hand-authored scorer; anything else needs a checkpoint path."""
    if entry["name"] == "scorer":
        return HandAuthoredScorer()
    if "checkpoint" not in entry:
        raise ValueError(f"policy {entry['name']!r} needs a 'checkpoint' path")
    return LearnedPolicyAdapter(entry["checkpoint"])


def resolve_personality(entry: dict, roster: dict[str, Personality]) -> Personality:
    """Inline ``ocean`` wins; a name-only entry is looked up in the roster file."""
    if "ocean" in entry:
        return Personality.from_traits(**entry["ocean"])
    name = entry["name"]
    if name not in roster:
        raise ValueError(
            f"personality {name!r}: not in personalities_file and no inline ocean")
    return roster[name]


def sync_unity_dir(out_dir: Path, unity_dir: Path, written: set[str]) -> None:
    """Mirror this run's S*.json into the Unity playback folder (stale files
    from earlier, larger runs are removed so the dropdown matches the run)."""
    unity_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(written):
        shutil.copyfile(out_dir / name, unity_dir / name)
    for stale in unity_dir.glob("S*.json"):
        if stale.name not in written:
            stale.unlink()
            print(f"removed stale {stale.name} from unity_dir")
    print(f"synced {len(written)} sequences to {unity_dir}")


def run_config(config_path: str | Path, preview: bool = False) -> int:
    """Generate every configured sequence; returns the number written."""
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    policies = [(e, build_policy(e)) for e in cfg["policies"]]
    roster: dict[str, Personality] = {}
    if cfg.get("personalities_file"):
        roster = load_personalities(cfg["personalities_file"])

    rows: list[dict] = []
    i = 0
    for world_path in cfg["worlds"]:
        world = load_world(world_path)
        for pers in cfg["personalities"]:
            personality = resolve_personality(pers, roster)
            for entry, policy in policies:
                # A personality may carry its own seeds. The personality-agnostic
                # control ignores the personality input, so one global seed would
                # give it the identical trail for every persona — visibly a bug to
                # a participant. Per-persona seeds keep the three policies matched
                # (same persona -> same seed) while separating the personas.
                for seed in pers.get("seeds", cfg["seeds"]):
                    i += 1
                    spec = SequenceSpec(
                        sequence_id=f"S{i:03d}",
                        policy_name=entry["name"],
                        checkpoint=entry.get("checkpoint", ""),
                        personality_name=pers["name"],
                        personality=personality,
                        world_path=str(world_path),
                        n_cycles=int(cfg["n_cycles"]),
                        seed=int(seed),
                        # Per-policy override. One global temperature is not
                        # neutral across policies: it is applied to whatever
                        # distribution a policy emits, and the scorer's is far
                        # flatter than a hard-label-trained model's, so a value
                        # that sharpens one into character locks the other onto
                        # a single option. Matching post-temperature entropy
                        # keeps the comparison about *what* is chosen.
                        selection_temperature=float(entry.get(
                            "selection_temperature",
                            cfg.get("selection_temperature", 1.0))),
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
                        "selection_temperature": m["selection_temperature"],
                        "generated_at": m["generated_at"],
                    })
                    if preview:
                        print(format_preview(seq))

    # A re-run with a smaller config must not leave orphaned stimuli beside a
    # shrunken manifest — the manifest and the S*.json set stay in lockstep.
    written = {row["file"] for row in rows}
    for stale in out_dir.glob("S*.json"):
        if stale.name not in written:
            stale.unlink()
            print(f"removed stale {stale.name}")

    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {i} sequences + manifest.csv to {out_dir}")

    if cfg.get("unity_dir"):
        sync_unity_dir(out_dir, Path(cfg["unity_dir"]), written)
    return i


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                    help=f"run config (default: {DEFAULT_CONFIG_PATH})")
    ap.add_argument("--preview", action="store_true",
                    help="print each sequence as one text line for QC")
    args = ap.parse_args()
    run_config(args.config, preview=args.preview)


if __name__ == "__main__":
    main()
