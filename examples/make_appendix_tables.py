"""Render the appendix parameter tables of the supervisor summary as images.

Markdown tables render without cell borders in the VS Code -> PDF pipeline, so
the supervisor document embeds these as figures instead. Values are read live
from ``npc_policy.weights`` / ``npc_policy.config``, so regenerating after a
tuning round keeps the document in sync with the source of truth:

    python -m examples.make_appendix_tables
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from npc_policy.config import ScorerConfig
from npc_policy.weights import (
    default_b_location,
    default_C_location,
    default_w_location,
    default_W_rel_action,
)

DOCS = Path(__file__).resolve().parent.parent / "docs"

LOC_TAGS = ["social", "stimulation", "structure", "cognitive", "physical",
            "risk", "exploration", "privacy", "conflict"]
REL_TAGS = ["cooperation", "helping", "conflict", "control"]
TRAITS = ["O", "C", "E", "A", "N"]

HEADER_BG = "#e8e8e8"
ROW_BG = {"b": "#f4f4f4", "w": "#f4f4f4"}


def _grid(ax, cell_text, col_labels, row_labels, col_width, row_colors=None):
    """A plain bordered grid: every cell gets all four edges."""
    table = ax.table(cellText=cell_text, colLabels=col_labels, rowLabels=row_labels,
                     cellLoc="center", rowLoc="right", loc="center",
                     colWidths=[col_width] * len(col_labels))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#555555")
        cell.set_linewidth(0.8)
        if r == 0:                                   # column header
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(weight="bold")
        if c == -1:                                  # row label
            cell.set_text_props(weight="bold")
        if row_colors and r - 1 in row_colors:
            cell.set_facecolor(row_colors[r - 1])
    ax.axis("off")
    return table


def make_location_table() -> Path:
    b, C, w = default_b_location(), default_C_location(), default_w_location()

    rows = [[f"{v:.3g}" for v in b]]
    rows += [[("0" if v == 0 else f"{v:+.2f}") for v in C[i]] for i in range(len(TRAITS))]
    rows += [[f"{v:.1f}" for v in w]]
    row_labels = ["b  (neutral ideal)"] + [f"C — {t}" for t in TRAITS] + ["w  (deviation cost)"]

    fig, ax = plt.subplots(figsize=(13, 3.6))
    _grid(ax, rows, LOC_TAGS, row_labels, col_width=0.105,
          row_colors={0: ROW_BG["b"], len(row_labels) - 1: ROW_BG["w"]})
    ax.set_title("Location level: neutral ideal levels $b$, trait shifts $C$, "
                 "deviation costs $w$  (all values provisional)",
                 fontsize=13, pad=16)
    out = DOCS / "appendix_location_tables.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def make_relational_table() -> Path:
    W_rel = default_W_rel_action()
    rows = [[("0" if v == 0 else f"{v:+.1f}") for v in W_rel[i]] for i in range(len(TRAITS))]

    fig, ax = plt.subplots(figsize=(7, 2.6))
    _grid(ax, rows, REL_TAGS, TRAITS, col_width=0.22)
    ax.set_title("Action level: relational-tag weights $W_{rel}$  (all values provisional)",
                 fontsize=13, pad=16)
    out = DOCS / "appendix_relational_table.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def make_scalar_table() -> Path:
    cfg = ScorerConfig()
    loc, act = cfg.location, cfg.action
    rows = [
        ["$K_L$, $K_A$", "memory lengths (locations, actions)", f"{cfg.K_L}, {cfg.K_A}"],
        [r"$\delta$", r"recency decay in $\alpha_j$", f"{cfg.recency_decay:g}"],
        [r"$\tau_0$", "base temperature", f"location {loc.tau_0:.1f}, action {act.tau_0:.1f}"],
        [r"$\lambda_R$", "satiation strength", f"{loc.lambda_R:.1f}"],
        [r"$\kappa_C$", "C-modulation of satiation", f"{loc.kappa_C:.1f}"],
        [r"$\lambda_O$", "O familiarity aversion", f"{loc.lambda_O:.1f}"],
        [r"$\lambda_C$", "C familiarity preference", f"{loc.lambda_C:.1f}"],
        [r"$\lambda_{Nf}$", "N familiarity clinging", f"{loc.lambda_Nf:.1f}"],
        [r"$\lambda_N$", "N temperature scale", f"{loc.lambda_N:.1f}"],
    ]

    fig, ax = plt.subplots(figsize=(9, 3.4))
    table = ax.table(cellText=rows, colLabels=["symbol", "meaning", "current value"],
                     cellLoc="left", loc="center", colWidths=[0.16, 0.52, 0.32])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#555555")
        cell.set_linewidth(0.8)
        cell.PAD = 0.04
        if r == 0:
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(weight="bold")
    ax.axis("off")
    ax.set_title("Scalar coefficients of Eqs. 3–6  (all values provisional)",
                 fontsize=13, pad=16)
    out = DOCS / "appendix_scalar_table.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    for path in (make_location_table(), make_relational_table(), make_scalar_table()):
        print(f"written: {path}")
