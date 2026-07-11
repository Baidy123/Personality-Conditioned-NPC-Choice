"""Study 2B training — hard labels through the unchanged 2A loop.

One-hot targets make ``kl_loss`` equal cross-entropy, so ``train.train_one``
runs as-is; ``best_val_kl`` in the run results IS the validation NLL. The
nonlinear families sweep weight decay (chosen on val NLL at evaluation time);
the simple families train without weight decay.

Run from ``code/``:  python -m experiments.rq2.train_2b [--device auto|cpu|cuda]
                                                        [--only PREFIX]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from npc_policy import IndependentCase

from .common import SEEDS, RunSpec, config_hash, read_pool
from .independent import IND_DATA, IND_RESULTS, WD_GRID, independent_case_to_inputs
from .train import pick_device, run_all, train_one

SIMPLE_MODELS = ("simple", "agnostic_simple")
NONLINEAR_MODELS = ("nonlinear", "agnostic_nonlinear")


def run_matrix_2b() -> list[RunSpec]:
    runs = [RunSpec("IND", m, s) for m in SIMPLE_MODELS for s in SEEDS]        # 10
    runs += [RunSpec("IND", m, s, tag=f"wd{wd:g}")
             for m in NONLINEAR_MODELS for wd in WD_GRID for s in SEEDS]       # 30
    return runs


def wd_of(spec: RunSpec) -> float:
    return float(spec.tag[2:]) if spec.tag.startswith("wd") else 0.0


def train_2b_one(spec, train_cases, val_cases, device, **kw):
    return train_one(spec, train_cases, val_cases, device,
                     to_inputs=independent_case_to_inputs,
                     weight_decay=wd_of(spec), **kw)


def make_loader_2b(data_dir: Path):
    """Case loader over the imported pool. Loads once, lazily (2A pattern)."""
    state: dict = {}

    def load_cases(spec: RunSpec):
        if "by_id" not in state:
            state["by_id"] = {t["id"]: c for c, t in
                              read_pool(data_dir / "cases.jsonl",
                                        case_cls=IndependentCase)}
            state["splits"] = json.loads(
                (data_dir / "splits.json").read_text(encoding="utf-8"))["splits"]
        by_id, splits = state["by_id"], state["splits"]
        return ([by_id[i] for i in splits["train"]],
                [by_id[i] for i in splits["val"]])

    return load_cases


def train_main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train all Study 2B runs (resumable)")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--only", default=None,
                    help="only run ids starting with this prefix (debugging)")
    ap.add_argument("--data", type=Path, default=IND_DATA)
    ap.add_argument("--results", type=Path, default=IND_RESULTS)
    ap.add_argument("--max-epochs", type=int, default=500)
    args = ap.parse_args(argv)
    if not (args.data / "cases.jsonl").exists():
        raise SystemExit(f"dataset missing: {args.data} - run import_independent first")
    meta = json.loads((args.data / "meta.json").read_text(encoding="utf-8"))
    if config_hash() != meta["config_hash"]:
        raise SystemExit("relation config drifted since import; re-run "
                         "import_independent or restore the config")
    specs = run_matrix_2b()
    if args.only:
        specs = [s for s in specs if s.run_id.startswith(args.only)]
    run_all(specs, make_loader_2b(args.data), args.results, pick_device(args.device),
            train_fn=train_2b_one, batch_size=64,
            max_epochs=args.max_epochs, patience=30)
    print(f"done: {args.results / 'runs'}")


if __name__ == "__main__":
    train_main()
