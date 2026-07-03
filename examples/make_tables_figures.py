"""Render the current scorer tables as figures (docs/v12_tables.png etc.).

The figures read the live tables in ``npc_policy.weights``, so regenerating
them after any tuning round keeps the docs in sync with the source of truth:

    python -m examples.make_tables_figures

Replaces the hand-drawn v1.1 drafts (``v11_tables.png`` /
``v11_ideal_levels_demo.png``), which predated the round-1 tuning and the
v1.2 ``conflict`` column.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from npc_policy.representation import Personality
from npc_policy.weights import (
    default_b_location,
    default_C_location,
    default_w_location,
    default_W_rel_action,
)

DOCS = Path(__file__).resolve().parent.parent / "docs"
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

LOC_LABELS = [
    "social\n社交", "stimulation\n刺激", "structure\n秩序", "cognitive\n认知",
    "physical\n体力", "risk\n风险", "exploration\n探索", "privacy\n私密",
    "conflict\n冲突",
]
REL_LABELS = ["cooperation\n合作", "helping\n帮助", "conflict\n冲突", "control\n控制"]
TRAIT_LABELS = ["O 开放", "C 尽责", "E 外向", "A 宜人", "N 神经质"]


def _row_panel(ax, values, labels, title, cmap, vmin, vmax) -> None:
    ax.imshow(values[None, :], cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for j, v in enumerate(values):
        ax.text(j, 0, f"{v:.2f}", ha="center", va="center", fontweight="bold", fontsize=13)
    ax.set_xticks(range(len(labels)), labels, fontsize=11)
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, loc="left", pad=10)
    for s in ax.spines.values():
        s.set_visible(True)


def _matrix_panel(ax, M, col_labels, title, vmax) -> None:
    ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt = "0" if v == 0 else f"{v:+.2f}"
            strong = abs(v) >= 0.6 * vmax
            ax.text(j, i, txt, ha="center", va="center", fontsize=11,
                    fontweight="bold" if strong else "normal",
                    color="white" if strong else "black")
    ax.set_xticks(range(len(col_labels)), col_labels, fontsize=11)
    ax.set_yticks(range(M.shape[0]), TRAIT_LABELS, fontsize=12)
    ax.set_title(title, fontsize=14, pad=10)
    for s in ax.spines.values():
        s.set_visible(True)


def make_tables_png() -> Path:
    b, C, w = default_b_location(), default_C_location(), default_w_location()
    W_rel = default_W_rel_action()

    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 3.4, 1, 2.6], hspace=0.75)
    fig.suptitle("方程 v1.2 系数表（location 层；action 层同结构、取前 7 列）— 数值均为 PROVISIONAL",
                 fontsize=17, y=0.985)

    _row_panel(fig.add_subplot(gs[0]), b, LOC_LABELS,
               "表1  b：中性人格的理想水平（性格全 0 的人觉得多少最舒服）— risk 基线 0.2，conflict 基线 0.1",
               "Greens", 0.0, 0.8)
    _matrix_panel(fig.add_subplot(gs[1]), C, LOC_LABELS,
                  "表2  C 主表：trait 每 +1 把该特征的理想水平推多少（红=往高推，蓝=往低推）\n"
                  "理想水平 μ = clip(b + C 列加权和, 0, 1)；conflict 列由低 A（凶悍）低 N（无畏）抬高",
                  0.6)
    _row_panel(fig.add_subplot(gs[2]), w, LOC_LABELS,
               "表3  w：偏离理想水平的扣分力度（越大越要紧）", "Oranges", 0.0, 1.2)
    _matrix_panel(fig.add_subplot(gs[3]), W_rel, REL_LABELS,
                  "表4  W_rel：action 独有 4 个关系特征，保持线性（红=越多越加分，蓝=越多越扣分）\n"
                  "A 宜人性主要活在这张表：爱合作、爱帮人、强烈厌恶冲突（round 3 加强到 −0.9）",
                  0.9)

    out = DOCS / "v12_tables.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def make_ideal_levels_png() -> Path:
    b, C = default_b_location(), default_C_location()

    profiles = [
        ("外向者 E=+0.8", Personality.from_traits(extraversion=0.8), "tab:red", "o"),
        ("尽责+神经质 C=+0.7, N=+0.8",
         Personality.from_traits(conscientiousness=0.7, neuroticism=0.8), "tab:blue", "s"),
        ("开放者 O=+0.9", Personality.from_traits(openness=0.9), "tab:green", "^"),
        ("凶悍无畏 A=−0.9, N=−0.7",
         Personality.from_traits(agreeableness=-0.9, neuroticism=-0.7), "tab:purple", "D"),
    ]

    fig, ax = plt.subplots(figsize=(15, 7))
    x = np.arange(len(LOC_LABELS))
    ax.bar(x, b, width=0.55, color="0.85", label="中性人格的理想水平 b", zorder=1)
    for label, p, color, marker in profiles:
        mu = np.clip(b + p.vector @ C, 0.0, 1.0)
        ax.scatter(x, mu, s=110, color=color, marker=marker, label=label, zorder=3,
                   edgecolors="white", linewidths=1.2)
    ax.set_xticks(x, LOC_LABELS, fontsize=12)
    ax.set_ylabel("理想水平 μ", fontsize=13)
    ax.set_ylim(0, 1.02)
    ax.set_title("同一张 C 表下，不同性格的理想水平 μ = clip(b + C 加权和, 0, 1)"
                 "（点离灰柱越远 = 该性格对这个特征的偏好越偏离常人）", fontsize=14)
    ax.legend(fontsize=12, loc="upper right")

    out = DOCS / "v12_ideal_levels_demo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    for path in (make_tables_png(), make_ideal_levels_png()):
        print(f"written: {path}")
