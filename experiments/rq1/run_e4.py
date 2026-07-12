"""E4 — memory-context ablation (the "decision structure" part of RQ1).

The bounded recent-choice context is a structural component of the proposal.
E4 reruns trajectories under three conditions — no memory, location memory
only, location + action memory — and measures what the component contributes,
and through which traits. By design the memory term works through O, C, and
(since v1.3) N: the O/C familiarity affinity, the C-modulated satiation, and
the N familiarity clinging. E and A should be largely unaffected. E4 verifies
that attribution.

Outputs: results/rq1/e4_ablation.png, e4_ablation.csv

Run from ``code/``:  python -m experiments.rq1.run_e4
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from npc_policy import HandAuthoredScorer
from npc_policy.representation import Personality

from .common import (
    CONDITION_COLORS, RESULTS, TRAIT_SHORT, TRAITS, TRAJ_SEEDS,
    load_cases, personality_of, run_trajectory, setup_style,
    trajectory_metrics, world_for, write_csv,
)

CONDITIONS = ("none", "location_only", "full")
METRICS = ("repeat_rate", "visit_entropy", "max_share", "distinct",
           "action_repeat_rate")


def systematic_profiles() -> list[tuple[str, Personality]]:
    out = [("neutral", Personality.from_traits())]
    for trait in TRAITS:
        for v in (-1.0, 1.0):
            out.append((f"{TRAIT_SHORT[trait]}{v:+.0f}",
                        Personality.from_traits(**{trait: v})))
    return out


def main() -> None:
    setup_style()
    cases = load_cases()
    scorer = HandAuthoredScorer()
    world = world_for("full")

    profiles = systematic_profiles() + [
        (e["id"], personality_of(e)) for e in cases["profiles"]["named"]
    ]

    rows = []
    agg: dict[tuple[str, str], dict[str, float]] = {}
    for pid, p in profiles:
        for cond in CONDITIONS:
            per_seed = {m: [] for m in METRICS}
            for seed in TRAJ_SEEDS:
                met = trajectory_metrics(
                    *run_trajectory(scorer, world, p, seed, memory=cond))
                for m in METRICS:
                    per_seed[m].append(met[m])
                rows.append([pid, cond, seed] + [f"{met[m]:.4f}" for m in METRICS])
            agg[(pid, cond)] = {m: float(np.mean(per_seed[m])) for m in METRICS}

    write_csv(RESULTS / "e4_ablation.csv",
              ["profile", "condition", "seed", *METRICS], rows)

    # ---- figure: repeat rate + variety, systematic profiles x conditions ----
    sys_ids = [pid for pid, _ in systematic_profiles()]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10.5), sharex=True)
    panel = (("repeat_rate", "stickiness: immediate location-repeat rate"),
             ("visit_entropy", "variety: entropy of visit shares"),
             ("action_repeat_rate",
              "action stickiness: action-repeat rate on same-location stays "
              "(separates location_only vs full)"))
    x = np.arange(len(sys_ids))
    for ax, (metric, title) in zip(axes, panel):
        for i, cond in enumerate(CONDITIONS):
            vals = [agg[(pid, cond)][metric] for pid in sys_ids]
            ax.bar(x + (i - 1) * 0.27, vals, width=0.25,
                   color=CONDITION_COLORS[cond], label=cond)
        ax.set_title(title)
        ax.set_xticks(x, sys_ids)
    axes[0].legend(frameon=False, ncol=3, title="memory condition")
    fig.suptitle("What the bounded recent-choice context contributes "
                 f"(50 rounds x {len(TRAJ_SEEDS)} seeds)", y=0.995)
    fig.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "e4_ablation.png", bbox_inches="tight")
    plt.close(fig)

    # ---- attribution check: which traits does memory act through? -----------
    print("E4 memory effect per trait (repeat_rate: none -> full):")
    for pid in sys_ids:
        none, full = agg[(pid, "none")]["repeat_rate"], agg[(pid, "full")]["repeat_rate"]
        print(f"  {pid:<8} {none:.3f} -> {full:.3f}   (delta {full - none:+.3f})")
    print(f"figure: {RESULTS / 'e4_ablation.png'}")


if __name__ == "__main__":
    main()
