"""The RQ2 report figure: controlled fidelity beside independent-label agreement.

Left  — Study 2A: KL(teacher ‖ student) per split, i.e. how faithfully each
        student reproduces the hand-authored policy, including the structured
        exclusions (drawn by ``run_2a.draw_gap_by_split``).
Right — Study 2B: top-1 agreement with the independent labels per test group
        (drawn by ``run_2b.draw_group_bars``).

Both panels read the aggregated ``main_table.csv`` that their own experiment
already wrote, so this script draws only — it never evaluates a model, and the
numbers cannot drift from the standalone figures. Nothing is printed above
either panel; the report caption names left and right.

Run from ``code/`` after run_2a and run_2b have written their tables:

    python -m experiments.rq2.make_combined_figure

Output: results/rq2/rq2_learning.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.rq1.common import setup_style

from .common import dirs
from .independent import IND_RESULTS
from .run_2a import _read_main_table, draw_gap_by_split
from .run_2b import draw_group_bars


def read_2b_table(path: Path) -> list[dict]:
    """Re-read 2B's main_table.csv with the types ``draw_group_bars`` expects."""
    def cast(k: str, v: str):
        return v if k in ("system", "group", "decision_type") else float(v)

    with path.open(encoding="utf-8", newline="") as f:
        return [{k: cast(k, v) for k, v in row.items()} for row in csv.DictReader(f)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Combined RQ2 report figure")
    ap.add_argument("--rq2a-results", type=Path, default=dirs(False)[1])
    ap.add_argument("--rq2b-results", type=Path, default=IND_RESULTS)
    ap.add_argument("--out", type=Path, default=None,
                    help="output path (default: <rq2a-results>/rq2_learning.png)")
    args = ap.parse_args(argv)
    setup_style()

    table_a = _read_main_table(args.rq2a_results / "main_table.csv")
    table_b = read_2b_table(args.rq2b_results / "main_table.csv")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 4.5))
    draw_gap_by_split(ax_a, table_a)
    draw_group_bars(ax_b, table_b)
    fig.tight_layout()

    out = args.out or (args.rq2a_results / "rq2_learning.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
