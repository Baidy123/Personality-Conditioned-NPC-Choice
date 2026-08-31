"""Study 2B label-property probe — what memory rule do the independent labels follow?

Run BEFORE training (pre-registration order): this describes the *labels*, not any
model. For every case with a non-empty buffer it asks whether the chosen option is
more or less familiar / repeated than the case's candidate average, and whether
that tendency varies with personality. The same measurement is applied to the
hand-authored scorer's argmax on the identical cases, giving a direct
teacher-vs-independent contrast of the three designed memory channels
(O novelty-seeking, C routine preference, N anxious clinging; E/A have no memory
channel by design).

The scorer is used here strictly as a *measured system*, never as a label source.

Run from ``code/``:  python -m experiments.rq2.run_label_probe
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.rq1.common import setup_style, write_csv
from npc_policy import HandAuthoredScorer, IndependentCase
from npc_policy.representation import Personality

from .common import read_pool
from .independent import IND_DATA, IND_RESULTS

TRAITS = ("O", "C", "E", "A", "N")
# designed channel signs (scorer equation v1.3); "." = no memory channel
EXPECTED = {"O": "-", "C": "+", "E": ".", "A": ".", "N": "+"}
BANDS = (("high", 0.3, 1.01), ("mid", -0.3, 0.3), ("low", -1.01, -0.3))


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.asarray(a, dtype=float).argsort()
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(len(a), dtype=float)
    return r


def spearman(x, y) -> float:
    """Rank correlation; 0.0 when either side is constant (no scipy dependency)."""
    rx, ry = _rank(np.asarray(x, float)), _rank(np.asarray(y, float))
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def chosen_deltas(case: IndependentCase, idx: int) -> tuple[float, float]:
    """(Δsim, Δrep) of option ``idx`` against this case's candidate means.

    Centring within the case removes the scenario's own familiarity level, so the
    number reads as "chose something more (+) or less (−) familiar than what was
    on offer" and is comparable across cases with different buffers.
    """
    rel = case.candidate_history_features
    return (float(rel.sim[idx] - rel.sim.mean()),
            float(rel.rep[idx] - rel.rep.mean()))


def probe(cases: list[IndependentCase]) -> tuple[list[dict], list[dict]]:
    """Per-case records + the per-(trait, band) summary table."""
    scorer = HandAuthoredScorer()
    records = []
    for c in cases:
        q = scorer.distribution(
            Personality(np.array(c.personality, dtype=float)), c.candidates,
            relations=c.candidate_history_features, level=c.decision_type)
        ai_sim, ai_rep = chosen_deltas(c, c.target_choice)
        sc_sim, sc_rep = chosen_deltas(c, int(np.argmax(q)))
        records.append({"p": np.array(c.personality, dtype=float),
                        "decision_type": c.decision_type,
                        "ai_sim": ai_sim, "ai_rep": ai_rep,
                        "sc_sim": sc_sim, "sc_rep": sc_rep})

    table = []
    for t, name in enumerate(TRAITS):
        xs = [r["p"][t] for r in records]
        row = {"trait": name, "expected_sign": EXPECTED[name], "n": len(records)}
        for who in ("ai", "sc"):
            for what in ("sim", "rep"):
                row[f"{who}_{what}_rho"] = spearman(xs, [r[f"{who}_{what}"] for r in records])
        table.append(row)
        for band, lo, hi in BANDS:
            sel = [r for r in records if lo <= r["p"][t] < hi]
            if not sel:
                continue
            band_row = {"trait": f"{name}_{band}", "expected_sign": "", "n": len(sel)}
            for who in ("ai", "sc"):
                for what in ("sim", "rep"):
                    band_row[f"{who}_{what}_rho"] = float(
                        np.mean([r[f"{who}_{what}"] for r in sel]))   # band means, not rho
            table.append(band_row)
    return records, table


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="2B label-property probe (run before training)")
    ap.add_argument("--data", type=Path, default=IND_DATA)
    ap.add_argument("--results", type=Path, default=IND_RESULTS)
    args = ap.parse_args(argv)
    setup_style()

    pool = [c for c, _ in read_pool(args.data / "cases.jsonl", case_cls=IndependentCase)]
    cases = [c for c in pool if c.candidate_history_features is not None]
    if not cases:
        raise SystemExit("no buffered cases - run import_independent first")
    print(f"{len(cases)} / {len(pool)} cases carry a non-empty buffer")

    records, table = probe(cases)
    args.results.mkdir(parents=True, exist_ok=True)
    write_csv(args.results / "label_probe.csv", list(table[0].keys()),
              [list(r.values()) for r in table])

    # figure: channel strength, independent labels vs scorer (rank correlations)
    rho_rows = [r for r in table if "_" not in r["trait"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    x = np.arange(len(TRAITS))
    for ax, what, xlabel in [(axes[0], "sim", "familiarity (sim)"),
                             (axes[1], "rep", "exact repetition (rep)")]:
        ax.bar(x - 0.2, [r[f"ai_{what}_rho"] for r in rho_rows], width=0.38,
               color="#0072B2", label="independent labels")
        ax.bar(x + 0.2, [r[f"sc_{what}_rho"] for r in rho_rows], width=0.38,
               color="#009E73", label="hand-authored scorer")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, [f"{t}\n({EXPECTED[t]})" for t in TRAITS])
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Spearman ρ (trait vs Δ of chosen option)\n+ clings   −  avoids")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(args.results / "label_probe.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\n{'trait':<7}{'exp':<5}{'AI sim':>8}{'AI rep':>8}{'scorer sim':>12}{'scorer rep':>12}")
    for r in rho_rows:
        print(f"{r['trait']:<7}{r['expected_sign']:<5}{r['ai_sim_rho']:>+8.3f}"
              f"{r['ai_rep_rho']:>+8.3f}{r['sc_sim_rho']:>+12.3f}{r['sc_rep_rho']:>+12.3f}")
    print(f"\nmean Δrep — labels {np.mean([r['ai_rep'] for r in records]):+.3f}, "
          f"scorer {np.mean([r['sc_rep'] for r in records]):+.3f}")
    print(f"written: {args.results / 'label_probe.csv'}, label_probe.png")


if __name__ == "__main__":
    main()
