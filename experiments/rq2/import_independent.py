"""Study 2B import: raw AI batches → validated, enriched, split dataset.

Rebuilds everything from ``data/rq2_independent/raw/*.json`` on every run
(idempotent; add files and re-run). Outputs ``cases.jsonl`` (pool format with
``gen`` tags), ``rejected.jsonl``, ``splits.json``, ``meta.json``, ``report.txt``.

Run from ``code/``:  python -m experiments.rq2.import_independent
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from npc_policy import IndependentCase

from .common import DATA, config_hash, write_pool
from .independent import (
    GENERAL_TARGETS,
    IMPORT_SEED,
    IND_DATA,
    KNOWN_KEYS,
    STRUCT_TARGETS,
    TRAITS,
    dedupe_key,
    enrich_case,
    in_pers_region,
    load_base_world,
    parse_raw_file,
    touches_arena,
    validate_case,
)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _assign_splits(records: list[tuple[IndependentCase, dict]],
                   rng: np.random.Generator) -> dict[str, list[str]]:
    """Spec §4: structured filters first (isolation enforced here, not by the
    AI following instructions), then proportional general-pool splitting."""
    pers = [t["id"] for c, t in records if in_pers_region(c)]
    arena = [t["id"] for c, t in records
             if touches_arena(c) and not in_pers_region(c)]
    general = [t["id"] for c, t in records
               if not in_pers_region(c) and not touches_arena(c)]
    for ids in (pers, arena, general):
        rng.shuffle(ids)

    splits = {"test_pers": pers[: STRUCT_TARGETS["test_pers"]],
              "test_arena": arena[: STRUCT_TARGETS["test_arena"]]}
    # structured surplus is dropped (never train/val — isolation), recorded in meta
    splits["dropped_structured"] = (pers[STRUCT_TARGETS["test_pers"]:]
                                    + arena[STRUCT_TARGETS["test_arena"]:])

    total = sum(GENERAL_TARGETS.values())               # 725
    n = len(general)
    n_iid = round(n * GENERAL_TARGETS["test_iid"] / total)
    n_val = round(n * GENERAL_TARGETS["val"] / total)
    splits["test_iid"] = general[:n_iid]
    splits["val"] = general[n_iid: n_iid + n_val]
    splits["train"] = general[n_iid + n_val:]
    return splits


def _coverage(records: list[tuple[IndependentCase, dict]]) -> list[str]:
    lines = []
    dts = Counter(c.decision_type for c, _ in records)
    lines.append(f"decision types: {dict(dts)}")
    for i, name in enumerate(TRAITS):
        vals = np.array([c.personality[i] for c, _ in records]) \
            if records else np.empty(0)
        lines.append(f"trait {name}: high(>0.3) {int((vals > 0.3).sum())}, "
                     f"mid {int(((vals >= -0.3) & (vals <= 0.3)).sum())}, "
                     f"low(<-0.3) {int((vals < -0.3).sum())}")
    empty = sum(1 for c, _ in records if c.candidate_history_features is None)
    lines.append(f"empty-history cases: {empty} ({empty / max(len(records), 1):.0%})")
    return lines


def run_import(raw_dir: Path = IND_DATA / "raw",
               out_dir: Path = IND_DATA) -> dict:
    world = load_base_world()
    files = sorted(raw_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no raw batches in {raw_dir}")

    records: list[tuple[IndependentCase, dict]] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    for f in files:
        source, cases, user_rejected = parse_raw_file(f)
        for item in user_rejected:
            rejected.append({"file": f.name, "reason": "user_rejected", "case": item})
        for idx, raw in cases:               # idx = raw array position (stable ids)
            extra = set(raw) - KNOWN_KEYS
            if extra:
                print(f"warning: {f.name}#{idx}: ignoring fields {sorted(extra)}")
            reason = validate_case(raw, world)
            if reason is None:
                key = dedupe_key(raw)
                if key in seen:
                    reason = "duplicate"
                seen.add(key)
            if reason is not None:
                rejected.append({"file": f.name, "reason": reason, "case": raw})
                continue
            case = enrich_case(raw, world, source)
            records.append((case, {"id": f"{f.stem}#{idx}", "source_file": f.name}))

    rng = np.random.default_rng(IMPORT_SEED)
    splits = _assign_splits(records, rng)
    group_of = {cid: g for g, ids in splits.items() for cid in ids}
    kept = []
    for case, tags in records:
        g = group_of.get(tags["id"])
        if g == "dropped_structured":
            rejected.append({"file": tags["source_file"], "reason": "structured_surplus",
                             "case": {"id": tags["id"]}})
            continue
        case.split = ("test" if g in ("test_iid", "test_pers", "test_arena") else g)
        kept.append((case, {**tags, "group": g}))

    out_dir.mkdir(parents=True, exist_ok=True)
    write_pool(out_dir / "cases.jsonl", kept)
    _atomic_write(out_dir / "rejected.jsonl",
                  "".join(json.dumps(r) + "\n" for r in rejected))
    public_splits = {k: v for k, v in splits.items() if k != "dropped_structured"}
    _atomic_write(out_dir / "splits.json",
                  json.dumps({"splits": public_splits}, indent=2))

    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": IMPORT_SEED,
        "config_hash": config_hash(),
        "world_hash": hashlib.sha256((DATA / "world.json").read_bytes()).hexdigest(),
        "raw_files": [f.name for f in files],
        "accepted": len(kept),
        "rejected": len(rejected),
        "split_sizes": {k: len(v) for k, v in public_splits.items()},
    }
    _atomic_write(out_dir / "meta.json", json.dumps(meta, indent=2))

    # report: aggregates only — test-case details stay blind (spec §4)
    trainval = [(c, t) for c, t in kept if t["group"] in ("train", "val")]
    lines = [f"accepted {len(kept)} / raw {len(kept) + len(rejected)}",
             "rejections: " + json.dumps(Counter(r["reason"] for r in rejected)),
             "split sizes: " + json.dumps(meta["split_sizes"]), "",
             "-- coverage (train+val pool only) --", *_coverage(trainval), ""]
    for grp, target in [("test_pers", STRUCT_TARGETS["test_pers"]),
                        ("test_arena", STRUCT_TARGETS["test_arena"]),
                        ("test_iid", GENERAL_TARGETS["test_iid"])]:
        got = meta["split_sizes"].get(grp, 0)
        if got < target:
            kind = {"test_pers": "personality-batch", "test_arena": "arena-batch",
                    "test_iid": "general-batch"}[grp]
            lines.append(f"SHORTFALL {grp}: {got}/{target} — request "
                         f"{target - got} more {kind} cases")
    _atomic_write(out_dir / "report.txt", "\n".join(lines) + "\n")
    print(f"imported {len(kept)} cases → {out_dir}; see report.txt")
    return meta


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Import Study 2B raw AI batches")
    ap.add_argument("--raw", type=Path, default=IND_DATA / "raw")
    ap.add_argument("--out", type=Path, default=IND_DATA)
    args = ap.parse_args(argv)
    run_import(raw_dir=args.raw, out_dir=args.out)


if __name__ == "__main__":
    main()
