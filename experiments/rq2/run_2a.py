"""Study 2A metrics: main table, figures, diagnostics.

Reads the run results and models written by ``train.py`` plus the test sets from
``gen_controlled.py``; writes CSVs and figures to ``results/rq2/``. G-split models
are evaluated on their own test set and on S0 (no-general-degradation check);
the S0 main models are evaluated on EVERY test set, so each G-split's
filtered-vs-S0-trained difference isolates the exclusion effect from the test
set's composition. All metrics per decision type and combined ("all").

Run from ``code/``:  python -m experiments.rq2.run_2a [--smoke]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.rq1.common import setup_style, write_csv
from npc_policy import ControlledCase
from npc_policy.learned import PolicyBatch, UniformBaseline

from .common import (
    ALL_SPLITS,
    DATA_SIZES,
    G_SPLITS,
    build_model,
    case_to_inputs,
    dirs,
    jsd_np,
    kl_np,
    read_pool,
    top1_agree,
)

FAMILY_COLORS = {"simple": "#0072B2", "nonlinear": "#E69F00",
                 "agnostic_simple": "#999999", "agnostic_nonlinear": "#CC79A7",
                 "uniform": "#CCCCCC"}
# What the figures call each family. The code key stays "simple" (run ids, CSV
# columns, lookups); the report calls that model the linear one.
FIG_LABEL = {"simple": "linear", "agnostic_simple": "agnostic_linear"}


def fig_label(name: str) -> str:
    return FIG_LABEL.get(name, name)


def load_student(results_dir: Path, run_id: str) -> torch.nn.Module:
    payload = torch.load(results_dir / "models" / f"{run_id}.pt", weights_only=False)
    model = build_model(payload["model"], seed=0)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def eval_cases(model: torch.nn.Module, cases: list[ControlledCase],
               ablation: str = "full", chunk: int = 512) -> list[dict]:
    """Per-case KL/JSD/top-1 rows; inputs get the run's ablation transform."""
    rows: list[dict] = []
    model.eval()
    for lo in range(0, len(cases), chunk):
        part = cases[lo:lo + chunk]
        batch = PolicyBatch.from_cases([case_to_inputs(c, ablation) for c in part])
        with torch.no_grad():
            probs = model(batch).exp().numpy()
        for i, c in enumerate(part):
            q = probs[i, : len(c.candidates)]
            t = np.asarray(c.target_distribution, dtype=float)
            rows.append({
                "decision_type": c.decision_type,
                "kl": kl_np(t, q), "jsd": jsd_np(t, q),
                "top1": int(top1_agree(t, q)),
            })
    return rows


def _summarise(rows: list[dict]) -> list[dict]:
    """Collapse per-case rows to per-decision-type (+ "all") means."""
    out = []
    for dt in ("location", "action", "all"):
        sel = rows if dt == "all" else [r for r in rows if r["decision_type"] == dt]
        if not sel:
            continue
        out.append({"decision_type": dt, "n_cases": len(sel),
                    "kl": float(np.mean([r["kl"] for r in sel])),
                    "jsd": float(np.mean([r["jsd"] for r in sel])),
                    "top1": float(np.mean([r["top1"] for r in sel]))})
    return out


def _std(xs: list[float]) -> float:
    """Sample std across seeds (ddof=1); 0.0 for a single seed."""
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def aggregate(per_run: list[dict]) -> list[dict]:
    """Mean ± sample std across seeds for each (split, model, ablation, n_train,
    eval_split, decision_type) group."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in per_run:
        key = (r["split"], r["model"], r["ablation"], r["n_train"],
               r["eval_split"], r["decision_type"])
        groups[key].append(r)
    table = []
    for (split, model, abl, n_train, eval_split, dt), rs in sorted(
            groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        table.append({
            "split": split, "model": model, "ablation": abl,
            "n_train": n_train, "eval_split": eval_split, "decision_type": dt,
            "n_cases": rs[0]["n_cases"], "n_seeds": len(rs),
            "kl_mean": float(np.mean([r["kl"] for r in rs])),
            "kl_std": _std([r["kl"] for r in rs]),
            "jsd_mean": float(np.mean([r["jsd"] for r in rs])),
            "jsd_std": _std([r["jsd"] for r in rs]),
            "top1_mean": float(np.mean([r["top1"] for r in rs])),
            "top1_std": _std([r["top1"] for r in rs]),
        })
    return table


def eval_splits_for(meta: dict, available: list[str]) -> list[str]:
    """Which test sets a run is evaluated on:

    S0 main runs (full context, full data) → every test set, so each G-split
    gets an S0-trained reference on identical cases; other S0 variants
    (ablations, data sizes) → S0 only; G-split runs → own test set + S0.
    """
    if (meta["split"] == "S0" and meta["ablation"] == "full"
            and meta["n_train"] is None):
        return list(available)
    if meta["split"] == "S0":
        return ["S0"]
    return [meta["split"], "S0"]


def _lookup(table, **want):
    """Rows of the aggregated table matching all given fields."""
    return [row for row in table if all(row[k] == v for k, v in want.items())]


def _read_main_table(path: Path) -> list[dict]:
    """Re-read an aggregated main_table.csv, restoring the value types the
    figure/diagnostic lookups expect (used by --figures-only, which redraws
    without the datasets and trained models being present)."""
    def cast(k: str, v: str):
        if k == "n_train":
            return None if v in ("", "None") else int(v)
        if k in ("split", "model", "ablation", "eval_split", "decision_type"):
            return v
        return float(v)
    with path.open(encoding="utf-8", newline="") as f:
        return [{k: cast(k, v) for k, v in row.items()} for row in csv.DictReader(f)]


def draw_gap_by_split(ax, table: list[dict]) -> None:
    """Student fidelity per split — standalone, and the left panel of the
    combined report figure drawn by ``make_combined_figure``.

    G2 (out-of-world spike, exclusion effect ~0) is dropped; the remaining
    splits are renumbered G1..G5 for display while lookups keep the real ids.
    """
    _kept = [g for g in G_SPLITS if g != "G2"]
    mains = {"S0": "S0"} | {f"G{i + 1}": g for i, g in enumerate(_kept)}
    x = np.arange(len(mains))
    series = [("linear", "simple", False), ("nonlinear", "nonlinear", False),
              ("linear (S0-trained)", "simple", True),
              ("nonlinear (S0-trained)", "nonlinear", True)]
    for i, (label, fam, s0_trained) in enumerate(series):
        ys, es = [], []
        for real in mains.values():
            if s0_trained and real == "S0":     # identical to the main S0 bar
                ys.append(np.nan)
                es.append(0.0)
                continue
            rows = _lookup(table, split="S0" if s0_trained else real, model=fam,
                           ablation="full", n_train=None, eval_split=real,
                           decision_type="all")
            ys.append(rows[0]["kl_mean"] if rows else np.nan)
            es.append(rows[0]["kl_std"] if rows else 0.0)
        ax.bar(x + (i - 1.5) * 0.21, ys, width=0.19, yerr=es,
               color=FAMILY_COLORS[fam], alpha=0.45 if s0_trained else 1.0,
               label=label, capsize=2)
    ax.set_xticks(x, list(mains))
    ax.set_ylabel("KL(teacher ‖ student), nats")
    ax.legend(frameon=False, loc="upper left", fontsize=7, ncol=2,
              handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Study 2A metrics and figures")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw the figures from an existing main_table.csv")
    args = ap.parse_args(argv)
    setup_style()
    data_dir, results_dir = dirs(args.smoke)

    if args.figures_only:
        table = _read_main_table(results_dir / "main_table.csv")
    else:
        runs_dir = results_dir / "runs"
        run_files = sorted(runs_dir.glob("*.json"))
        if not run_files:
            raise SystemExit(f"no run results in {runs_dir} - run train first")

        tests = {s: [c for c, _ in read_pool(data_dir / f"test_{s}.jsonl")]
                 for s in ALL_SPLITS if (data_dir / f"test_{s}.jsonl").exists()}

        # -- per-run evaluation ------------------------------------------------
        per_run: list[dict] = []
        for f in run_files:
            meta = json.loads(f.read_text(encoding="utf-8"))
            model = load_student(results_dir, meta["run_id"])
            for split in eval_splits_for(meta, list(tests)):
                for s in _summarise(eval_cases(model, tests[split], meta["ablation"])):
                    per_run.append({**s, "split": meta["split"], "model": meta["model"],
                                    "ablation": meta["ablation"], "n_train": meta["n_train"],
                                    "seed": meta["seed"], "eval_split": split})
            print(f"evaluated {meta['run_id']}")
        # untrained floor, once per split
        for split, cases in tests.items():
            for s in _summarise(eval_cases(UniformBaseline(), cases)):
                per_run.append({**s, "split": split, "model": "uniform", "ablation": "full",
                                "n_train": None, "seed": 0, "eval_split": split})

        table = aggregate(per_run)
        write_csv(results_dir / "main_table.csv", list(table[0].keys()),
                  [list(r.values()) for r in table])

    # -- figure: linear-vs-nonlinear gap across splits ---------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    draw_gap_by_split(ax, table)
    fig.savefig(results_dir / "gap_by_split.png", bbox_inches="tight")
    plt.close(fig)

    # -- figure: data-size curve -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sizes = list(DATA_SIZES)
    for fam in ("simple", "nonlinear"):
        xs, ys, es = [], [], []
        for n in sizes + [None]:                      # None = the full-data S0 runs
            rows = _lookup(table, split="S0", model=fam, ablation="full",
                           n_train=n, eval_split="S0", decision_type="all")
            if rows:
                xs.append(100_000 if n is None else n)   # None = the 100k S0 main runs
                ys.append(rows[0]["kl_mean"])
                es.append(rows[0]["kl_std"])
        ax.errorbar(xs, ys, yerr=es, marker="o", color=FAMILY_COLORS[fam],
                    label=fig_label(fam))
    ax.set_xscale("log")
    ax.set_xlabel("training cases")
    ax.set_ylabel("KL(teacher ‖ student), nats")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.savefig(results_dir / "data_size_curve.png", bbox_inches="tight")
    plt.close(fig)

    if args.figures_only:
        print("redrawn: gap_by_split.png, data_size_curve.png")
        return

    # -- diagnostics --------------------------------------------------------------
    diag_rows: list[list] = []
    for f in run_files:                                # c_L ⊙ o weight-norm check
        meta = json.loads(f.read_text(encoding="utf-8"))
        if meta["model"] not in ("simple", "agnostic_simple"):
            continue
        payload = torch.load(results_dir / "models" / f"{meta['run_id']}.pt",
                             weights_only=False)
        key = "w" if meta["model"] == "simple" else "inner.w"
        w = payload["state_dict"][key]
        # location row (w[0]) is identically 0.0 by construction: ctx is all-zero
        # for location cases, so the block never gets gradient from zero init.
        # The reportable diagnostic is the action row (w[1]).
        diag_rows.append(["w_ctx_norm", meta["run_id"],
                          float(w[0, -12:].norm()), float(w[1, -12:].norm())])
    for fam in ("simple", "nonlinear"):                # personality-information gap
        main_rows = _lookup(table, split="S0", model=fam, ablation="full",
                            n_train=None, eval_split="S0", decision_type="all")
        agn = _lookup(table, split="S0", model=f"agnostic_{fam}", ablation="full",
                      n_train=None, eval_split="S0", decision_type="all")
        if main_rows and agn:
            diag_rows.append(["agnostic_gap", fam,
                              agn[0]["kl_mean"] - main_rows[0]["kl_mean"], ""])
        for abl in ("no_context", "location_only"):    # ablation deltas
            row = _lookup(table, split="S0", model=fam, ablation=abl, n_train=None,
                          eval_split="S0", decision_type="all")
            if main_rows and row:
                diag_rows.append([f"ablation_delta_{abl}", fam,
                                  row[0]["kl_mean"] - main_rows[0]["kl_mean"], ""])
    write_csv(results_dir / "diagnostics.csv",
              ["kind", "who", "value_a", "value_b"], diag_rows)

    print(f"written: {results_dir / 'main_table.csv'}, gap_by_split.png, "
          f"data_size_curve.png, diagnostics.csv")


if __name__ == "__main__":
    main()
