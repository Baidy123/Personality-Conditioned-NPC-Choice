"""E4 — memory-context ablation (the "decision structure" part of RQ1).

The bounded recent-choice context is a structural component of the proposal.
E4 measures what it contributes and through which traits, at both levels:
location choice is compared on free trajectories with H_L off vs live, action
choice on fixed-location rollouts with H_A off vs live. Holding the location
fixed is what makes the action side readable — on a free trajectory a profile
that rarely stays put leaves the same-location denominator empty.

By design the memory term works through O, C, and (since v1.3) N: the O/C
familiarity affinity, the C-modulated satiation, and the N familiarity
clinging. E and A should be largely unaffected. E4 verifies that attribution.

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
    action_repeat_rate, load_cases, personality_of, run_action_trajectory,
    run_trajectory, setup_style, trajectory_metrics, world_for, write_csv,
)

CONDITIONS = ("none", "full")
METRICS = ("repeat_rate", "visit_entropy", "max_share", "distinct")
ACTION_METRIC = "action_repeat_rate"


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

    location_ids = [o.id for o in world.resolve()]

    rows = []
    agg: dict[tuple[str, str], dict[str, float]] = {}
    for pid, p in profiles:
        for cond in CONDITIONS:
            per_seed = {m: [] for m in METRICS}
            per_seed[ACTION_METRIC] = []
            for seed in TRAJ_SEEDS:
                visits, _ = run_trajectory(scorer, world, p, seed, memory=cond)
                met = trajectory_metrics(visits)
                # Action side: one fixed-location rollout per location, averaged
                # over the six. Staying put keeps H_A alive for the whole run, so
                # every profile contributes a rate rather than an empty denominator.
                met[ACTION_METRIC] = float(np.mean([
                    action_repeat_rate(run_action_trajectory(
                        scorer, world, p, loc, seed, memory=cond))
                    for loc in location_ids
                ]))
                for m in (*METRICS, ACTION_METRIC):
                    per_seed[m].append(met[m])
                rows.append([pid, cond, seed]
                            + [f"{met[m]:.4f}" for m in (*METRICS, ACTION_METRIC)])
            agg[(pid, cond)] = {m: float(np.mean(per_seed[m]))
                                for m in (*METRICS, ACTION_METRIC)}

    write_csv(RESULTS / "e4_ablation.csv",
              ["profile", "condition", "seed", *METRICS, ACTION_METRIC], rows)

    # ---- figure: location + action repeat rate, systematic profiles x conditions ----
    sys_ids = [pid for pid, _ in systematic_profiles()]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    # Title names the level, y label names the rate being plotted; the profile
    # ticks are repeated on both panels so neither has to be read off the other.
    panel = (("repeat_rate", "immediate location-repeat rate"),
             (ACTION_METRIC,
              "action-repeat rate at a fixed location\n(mean over the 6 locations)"))
    x = np.arange(len(sys_ids))
    for ax, (metric, ylabel) in zip(axes, panel):
        for i, cond in enumerate(CONDITIONS):
            vals = [agg[(pid, cond)][metric] for pid in sys_ids]
            ax.bar(x + (i - 0.5) * 0.42, vals, width=0.38,
                   color=CONDITION_COLORS[cond], label=cond)
        ax.set_xticks(x, sys_ids)
        ax.tick_params(labelbottom=True)  # sharex would hide the top panel's
        ax.set_xlabel("personality profile (neutral + single-trait endpoints)")
        ax.set_ylabel(ylabel)
    axes[0].legend(frameon=False, ncol=2, title="memory condition")
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
