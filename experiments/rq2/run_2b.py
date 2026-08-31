"""Study 2B / Study 3 automated comparison on the independent test set.

Every system predicts the same test cases; metrics are top-1 accuracy and
NLL of the labelled choice. Learned families aggregate over seeds (mean ± sd);
per (family, seed) the nonlinear weight-decay variant with the best val NLL is
selected. The hand-authored scorer appears here strictly as an evaluated
system — it took no part in labels or training.

Run from ``code/``:  python -m experiments.rq2.run_2b
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.rq1.common import setup_style, write_csv
from npc_policy import HandAuthoredScorer, IndependentCase
from npc_policy.learned import PolicyBatch, UniformBaseline
from npc_policy.representation import Personality

from .common import SEEDS, read_pool
from .independent import IND_DATA, IND_RESULTS, TEST_GROUPS, independent_case_to_inputs
from .run_2a import load_student
from .train_2b import NONLINEAR_MODELS, SIMPLE_MODELS

SYSTEM_COLORS = {"uniform": "#CCCCCC", "agnostic_simple": "#999999",
                 "agnostic_nonlinear": "#CC79A7", "scorer": "#009E73",
                 "simple": "#0072B2", "nonlinear": "#E69F00"}
# What the figures call each system. The code key stays "simple" (run ids, CSV
# columns, lookups); the report calls that model the linear one.
FIG_LABEL = {"simple": "linear", "agnostic_simple": "agnostic_linear"}
# RQ2 asks about acquisition and generalisation, not about which policy is better.
# The scorer is fit to the author's intuitions and the learned models to these
# labels, so a horse race between them is confounded (revised 2026-07-12; policy
# comparison belongs to the human study). Its per-case rows stay in main_table.csv
# and diagnostics.csv as a representation diagnostic, but it is kept out of the
# figures, which are RQ2 evidence.
PLOT_SYSTEMS = ("uniform", "agnostic_simple", "agnostic_nonlinear",
                "simple", "nonlinear")
_TINY = np.finfo(float).tiny


def fig_label(name: str) -> str:
    return FIG_LABEL.get(name, name)


def scorer_probs(case: IndependentCase) -> np.ndarray:
    scorer = HandAuthoredScorer()
    return scorer.distribution(
        Personality(np.array(case.personality, dtype=float)),
        case.candidates, relations=case.candidate_history_features,
        level=case.decision_type)


def model_probs(model: torch.nn.Module, cases: list[IndependentCase],
                chunk: int = 512) -> list[np.ndarray]:
    out = []
    model.eval()
    for lo in range(0, len(cases), chunk):
        part = cases[lo:lo + chunk]
        batch = PolicyBatch.from_cases(
            [independent_case_to_inputs(c) for c in part])
        with torch.no_grad():
            probs = model(batch).exp().numpy()
        out.extend(probs[i, : len(c.candidates)] for i, c in enumerate(part))
    return out


def case_metrics(q: np.ndarray, case: IndependentCase) -> dict:
    y = case.target_choice
    return {"top1": int(int(np.argmax(q)) == y),
            "nll": float(-np.log(max(q[y], _TINY))),
            "decision_type": case.decision_type}


def best_nonlinear_runs(results_dir: Path, family: str) -> list[str]:
    """Per seed, the weight-decay variant with the lowest val NLL."""
    best: dict[int, tuple[float, str]] = {}
    for f in (results_dir / "runs").glob(f"IND__{family}__wd*.json"):
        meta = json.loads(f.read_text(encoding="utf-8"))
        seed, val = meta["seed"], meta["best_val_kl"]
        if seed not in best or val < best[seed][0]:
            best[seed] = (val, meta["run_id"])
    return [run_id for _, run_id in sorted(best.values(), key=lambda t: t[1])]


def _rows_for(system: str, seed, per_case: list[dict],
              groups: dict[str, str], ids: list[str]) -> list[dict]:
    """Aggregate one prediction set to (group × decision_type) means."""
    rows = []
    for grp in ("all",) + TEST_GROUPS:
        sel = [m for m, cid in zip(per_case, ids)
               if grp == "all" or groups[cid] == grp]
        if not sel:
            continue
        for dt in ("location", "action", "all"):
            part = sel if dt == "all" else [m for m in sel if m["decision_type"] == dt]
            if not part:
                continue
            rows.append({"system": system, "seed": seed, "group": grp,
                         "decision_type": dt, "n_cases": len(part),
                         "top1": float(np.mean([m["top1"] for m in part])),
                         "nll": float(np.mean([m["nll"] for m in part]))})
    return rows


def _std(xs):
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def draw_group_bars(ax, table: list[dict]) -> None:
    """Top-1 per system × test group — standalone, and the right panel of the
    combined report figure drawn by ``make_combined_figure``."""
    plot_groups = ("all",) + TEST_GROUPS
    x = np.arange(len(plot_groups))
    systems = [s for s in PLOT_SYSTEMS if any(r["system"] == s for r in table)]
    for i, sys_ in enumerate(systems):
        ys, es = [], []
        for grp in plot_groups:
            rows = [r for r in table if r["system"] == sys_ and r["group"] == grp
                    and r["decision_type"] == "all"]
            ys.append(rows[0]["top1_mean"] if rows else np.nan)
            es.append(rows[0]["top1_std"] if rows else 0.0)
        off = (i - (len(systems) - 1) / 2) * 0.8 / len(systems)
        ax.bar(x + off, ys, width=0.75 / len(systems), yerr=es,
               color=SYSTEM_COLORS[sys_], label=fig_label(sys_), capsize=2)
    ax.set_xticks(x, plot_groups)
    ax.set_ylabel("top-1 accuracy")
    # Headroom so the legend clears the bars; both report panels carry their
    # legend in the same corner (upper left).
    ax.set_ylim(0, ax.get_ylim()[1] * 1.30)
    ax.legend(frameon=False, loc="upper left", fontsize=7, ncol=2,
              handlelength=1.2, labelspacing=0.3, columnspacing=1.0,
              borderaxespad=0.2)


def eval_main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Study 2B comparative evaluation")
    ap.add_argument("--data", type=Path, default=IND_DATA)
    ap.add_argument("--results", type=Path, default=IND_RESULTS)
    args = ap.parse_args(argv)
    setup_style()

    pool = read_pool(args.data / "cases.jsonl", case_cls=IndependentCase)
    by_id = {t["id"]: c for c, t in pool}
    groups = {t["id"]: t["group"] for _, t in pool}
    ids = [t["id"] for _, t in pool if t["group"] in TEST_GROUPS]
    cases = [by_id[i] for i in ids]
    if not cases:
        raise SystemExit("no test cases in the pool - run import_independent first")

    per_seed_rows: list[dict] = []
    diag_rows: list[list] = []          # per-case records for error analysis

    def record(system: str, seed, per_case: list[dict]) -> None:
        per_seed_rows.extend(_rows_for(system, seed, per_case, groups, ids))
        diag_rows.extend([system, seed, cid, groups[cid], m["decision_type"],
                          m["top1"], round(m["nll"], 4)]
                         for m, cid in zip(per_case, ids))

    # fixed systems (no seeds)
    record("uniform", 0, [case_metrics(q, c) for q, c in
                          zip(model_probs(UniformBaseline(), cases), cases)])
    record("scorer", 0, [case_metrics(scorer_probs(c), c) for c in cases])
    # learned systems
    run_ids = {m: [f"IND__{m}__s{s}" for s in SEEDS] for m in SIMPLE_MODELS}
    run_ids |= {m: best_nonlinear_runs(args.results, m) for m in NONLINEAR_MODELS}
    for system, rids in run_ids.items():
        for rid in rids:
            if not (args.results / "runs" / f"{rid}.json").exists():
                continue
            model = load_student(args.results, rid)
            seed = json.loads((args.results / "runs" / f"{rid}.json")
                              .read_text(encoding="utf-8"))["seed"]
            record(system, seed, [case_metrics(q, c) for q, c in
                                  zip(model_probs(model, cases), cases)])

    # aggregate over seeds
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in per_seed_rows:
        grouped[(r["system"], r["group"], r["decision_type"])].append(r)
    table = [{"system": sys_, "group": grp, "decision_type": dt,
              "n_cases": rs[0]["n_cases"], "n_seeds": len(rs),
              "top1_mean": float(np.mean([r["top1"] for r in rs])),
              "top1_std": _std([r["top1"] for r in rs]),
              "nll_mean": float(np.mean([r["nll"] for r in rs])),
              "nll_std": _std([r["nll"] for r in rs])}
             for (sys_, grp, dt), rs in sorted(grouped.items())]
    write_csv(args.results / "main_table.csv", list(table[0].keys()),
              [list(r.values()) for r in table])

    # per-case records (spec §6): which cases each system gets wrong
    write_csv(args.results / "diagnostics.csv",
              ["system", "seed", "case_id", "group", "decision_type", "top1", "nll"],
              diag_rows)

    # data-size curve (amendment 2026-07-12): test top-1 vs training size;
    # full-data points reuse the main-table "all/all" per-seed rows
    curve: list[dict] = []
    for f in (args.results / "runs").glob("IND__*__n*.json"):
        meta = json.loads(f.read_text(encoding="utf-8"))
        if meta.get("n_train") is None:
            continue
        model = load_student(args.results, meta["run_id"])
        per = [case_metrics(q, c) for q, c in zip(model_probs(model, cases), cases)]
        curve.append({"model": meta["model"], "n_train": meta["n_train"],
                      "top1": float(np.mean([m["top1"] for m in per])),
                      "nll": float(np.mean([m["nll"] for m in per]))})
    n_full = len(json.loads((args.data / "splits.json")
                            .read_text(encoding="utf-8"))["splits"]["train"])
    for r in per_seed_rows:
        if (r["system"] in ("simple", "nonlinear") and r["group"] == "all"
                and r["decision_type"] == "all"):
            curve.append({"model": r["system"], "n_train": n_full,
                          "top1": r["top1"], "nll": r["nll"]})
    if curve:
        cgroups: dict[tuple, list[dict]] = defaultdict(list)
        for r in curve:
            cgroups[(r["model"], r["n_train"])].append(r)
        crows = [{"model": m, "n_train": n, "n_seeds": len(rs),
                  "top1_mean": float(np.mean([r["top1"] for r in rs])),
                  "top1_std": _std([r["top1"] for r in rs]),
                  "nll_mean": float(np.mean([r["nll"] for r in rs])),
                  "nll_std": _std([r["nll"] for r in rs])}
                 for (m, n), rs in sorted(cgroups.items())]
        write_csv(args.results / "data_size_curve.csv", list(crows[0].keys()),
                  [list(r.values()) for r in crows])
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for fam in ("simple", "nonlinear"):
            pts = [r for r in crows if r["model"] == fam]
            ax.errorbar([r["n_train"] for r in pts],
                        [r["top1_mean"] for r in pts],
                        yerr=[r["top1_std"] for r in pts],
                        marker="o", color=SYSTEM_COLORS[fam],
                        label=fig_label(fam))
        agn = [r for r in table if r["system"] == "agnostic_nonlinear"
               and r["group"] == "all" and r["decision_type"] == "all"]
        if agn:
            ax.axhline(agn[0]["top1_mean"], ls="--",
                       color=SYSTEM_COLORS["agnostic_nonlinear"],
                       label="personality-agnostic control")
        ax.set_xscale("log")
        ax.set_xlabel("training cases")
        ax.set_ylabel("top-1 accuracy (all test groups)")
        lo, hi = ax.get_ylim()                  # headroom so the legend clears the curves
        ax.set_ylim(lo, hi + 0.22 * (hi - lo))
        ax.legend(frameon=False, loc="upper right", fontsize=7,
                  handlelength=1.4, labelspacing=0.3, borderaxespad=0.2)
        fig.savefig(args.results / "data_size_curve.png", bbox_inches="tight")
        plt.close(fig)

    # figure: top-1 per system × group
    fig, ax = plt.subplots(figsize=(9, 4.5))
    draw_group_bars(ax, table)
    fig.savefig(args.results / "group_bars.png", bbox_inches="tight")
    plt.close(fig)
    print(f"written: {args.results / 'main_table.csv'}, group_bars.png, diagnostics.csv")


if __name__ == "__main__":
    eval_main()
