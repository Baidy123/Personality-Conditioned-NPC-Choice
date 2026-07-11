"""Study 2A training runs — resumable, device-agnostic (design §3).

One run = split × model × seed (``common.run_matrix``, 130 runs full / 4 smoke).
Each run writes ``results/rq2/runs/<run_id>.json`` and ``models/<run_id>.pt``
on completion; re-invoking the command skips finished ids, so an interrupted
session continues where it left off.

Run from ``code/``:  python -m experiments.rq2.train [--smoke] [--device auto|cpu|cuda]
                                                     [--only PREFIX]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from npc_policy import ControlledCase
from npc_policy.learned import PolicyBatch, kl_loss

from .common import (
    RunSpec,
    build_model,
    case_to_inputs,
    config_hash,
    dirs,
    read_pool,
    run_matrix,
)


def pick_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def batch_to(b: PolicyBatch, device: torch.device, dtype: torch.dtype) -> PolicyBatch:
    """Move a batch to the training device/dtype (design §3: f32 CUDA, f64 CPU)."""
    return replace(
        b,
        p=b.p.to(device=device, dtype=dtype),
        d=b.d.to(device=device),
        ctx=b.ctx.to(device=device, dtype=dtype),
        cand=b.cand.to(device=device, dtype=dtype),
        rel=b.rel.to(device=device, dtype=dtype),
        mask=b.mask.to(device=device),
        target=None if b.target is None else b.target.to(device=device, dtype=dtype),
    )


def _device_batch_source(input_dicts: list[dict], device: torch.device,
                         dtype: torch.dtype):
    """One padded on-device tensor set; ``take(idx)`` slices out a ``PolicyBatch``.

    Per-step ``PolicyBatch.from_cases`` + host→device copies dominate wall time
    for these tiny models (the accelerator math is trivial); assembling once and
    indexing on-device removes that bottleneck. Batch composition and numerics
    are unchanged: the global padding width only adds mask-excluded columns,
    which contribute exact zeros through the masked softmax.
    """
    full = batch_to(PolicyBatch.from_cases(input_dicts), device, dtype)

    def take(idx) -> PolicyBatch:
        t = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=device)
        return replace(
            full, p=full.p[t], d=full.d[t], ctx=full.ctx[t], cand=full.cand[t],
            rel=full.rel[t], mask=full.mask[t],
            target=None if full.target is None else full.target[t],
        )

    return take, full.mask.shape[0]


def _mean_val_kl(model: torch.nn.Module, val_batches: list[PolicyBatch]) -> float:
    model.eval()
    tot = n = 0
    with torch.no_grad():
        for b in val_batches:
            tot += float(kl_loss(model(b), b.target, b.mask)) * b.mask.shape[0]
            n += b.mask.shape[0]
    return tot / n


def train_one(
    spec: RunSpec,
    train_cases: list[ControlledCase],
    val_cases: list[ControlledCase],
    device: torch.device,
    *,
    lr: float = 1e-3,
    batch_size: int = 256,
    max_epochs: int = 150,
    patience: int = 10,
    to_inputs=case_to_inputs,
    weight_decay: float | None = None,   # None → 2A rule (1e-4 nonlinear, else 0)
) -> tuple[dict, dict]:
    """Train one run; returns (result dict, best float64-CPU state_dict).

    ``to_inputs`` builds the input dicts (2B passes a one-hot-target builder:
    a one-hot target makes ``kl_loss`` compute exactly ``-log q[choice]``, i.e.
    cross-entropy, so hard labels train through this unchanged loop).
    """
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    take_train, n_train = _device_batch_source(
        [to_inputs(c, spec.ablation) for c in train_cases], device, dtype)
    take_val, n_val = _device_batch_source(
        [to_inputs(c, spec.ablation) for c in val_cases], device, dtype)
    val_batches = [take_val(np.arange(i, min(i + 512, n_val)))
                   for i in range(0, n_val, 512)]

    model = build_model(spec.model, spec.seed).to(device=device, dtype=dtype)
    if weight_decay is None:
        weight_decay = 1e-4 if "nonlinear" in spec.model else 0.0
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    shuffle_rng = np.random.default_rng(spec.seed)

    t0 = time.time()
    best_kl, best_state, best_epoch = float("inf"), None, -1
    epochs_run = 0
    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        model.train()
        order = shuffle_rng.permutation(n_train)
        for lo in range(0, len(order), batch_size):
            batch = take_train(order[lo:lo + batch_size])
            opt.zero_grad()
            loss = kl_loss(model(batch), batch.target, batch.mask)
            loss.backward()
            opt.step()
        val_kl = _mean_val_kl(model, val_batches)
        if val_kl < best_kl - 1e-6:
            best_kl, best_epoch = val_kl, epoch
            best_state = {k: v.detach().to("cpu", torch.float64).clone()
                          for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

    if best_state is None:      # e.g. NaN val KL from epoch 0 (diverged run)
        raise RuntimeError(f"{spec.run_id}: no finite val KL in {epochs_run} epochs")
    result = {
        "run_id": spec.run_id,
        **asdict(spec),
        "best_val_kl": best_kl,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "wall_time_s": round(time.time() - t0, 1),
        "device": device.type,
        "dtype": "float32" if dtype == torch.float32 else "float64",
        "n_train_cases": len(train_cases),
        "config_hash": config_hash(),
    }
    return result, best_state


def run_all(specs, load_cases, results_dir: Path, device: torch.device,
            train_fn=train_one, **train_kw) -> None:
    """Run every spec whose result file does not exist yet (resume = skip)."""
    runs_dir = results_dir / "runs"
    models_dir = results_dir / "models"
    runs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in specs if not (runs_dir / f"{s.run_id}.json").exists()]
    print(f"{len(specs)} runs, {len(specs) - len(todo)} already done, {len(todo)} to go "
          f"(device: {device.type})")
    for i, spec in enumerate(todo, 1):
        train_cases, val_cases = load_cases(spec)
        result, state = train_fn(spec, train_cases, val_cases, device, **train_kw)
        # atomic writes: a kill mid-write must not leave a truncated file that
        # marks the run complete (same pattern as common.write_pool)
        model_path = models_dir / f"{spec.run_id}.pt"
        tmp = model_path.with_suffix(model_path.suffix + ".tmp")
        torch.save({"model": spec.model, "state_dict": state}, tmp)
        os.replace(tmp, model_path)
        result_path = runs_dir / f"{spec.run_id}.json"
        tmp = result_path.with_suffix(result_path.suffix + ".tmp")
        tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
        os.replace(tmp, result_path)
        print(f"[{i}/{len(todo)}] {spec.run_id}: val KL {result['best_val_kl']:.4g} "
              f"({result.get('epochs_run', '?')} epochs, "
              f"{result.get('wall_time_s', '?')}s)")


def make_loader(data_dir: Path):
    """Case loader over the pool + manifests. Loads the pool once, lazily."""
    state: dict = {}

    def load_cases(spec: RunSpec):
        if "by_id" not in state:
            print("loading pool ...")
            state["by_id"] = {t["id"]: c for c, t in read_pool(data_dir / "pool.jsonl")}
            state["splits"] = json.loads(
                (data_dir / "splits.json").read_text(encoding="utf-8"))["splits"]
        part = state["splits"][spec.split]
        train_ids = part["train"]
        if spec.n_train is not None:
            train_ids = train_ids[: spec.n_train]      # nested subsets by construction
        by_id = state["by_id"]
        return [by_id[i] for i in train_ids], [by_id[i] for i in part["val"]]

    return load_cases


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train all Study 2A runs (resumable)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--only", default=None,
                    help="only run ids starting with this prefix (debugging)")
    args = ap.parse_args(argv)
    data_dir, results_dir = dirs(args.smoke)
    if not (data_dir / "pool.jsonl").exists():
        raise SystemExit(f"dataset missing: {data_dir} - run gen_controlled first")
    gen_meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    if config_hash() != gen_meta["config_hash"]:
        raise SystemExit(
            f"scorer config drifted since dataset generation "
            f"({config_hash()[:12]} != {gen_meta['config_hash'][:12]}); "
            "regenerate the dataset or restore the config")
    device = pick_device(args.device)
    specs = run_matrix(smoke=args.smoke)
    if args.only:
        specs = [s for s in specs if s.run_id.startswith(args.only)]
    kw = dict(max_epochs=8) if args.smoke else {}
    run_all(specs, make_loader(data_dir), results_dir, device, **kw)
    print(f"done: {results_dir / 'runs'}")


if __name__ == "__main__":
    main()
