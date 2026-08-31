"""E3 — trajectory-level consistency (the "consistent patterns" part of RQ1).

Static distributions can differ while sampled behaviour washes out (or the
reverse: memory dynamics could erase personality differences). E3 runs
50-round controller trajectories (10 seeds each) and measures the *pattern*
statistics — concentration, routine, variety — as a function of each trait.

Profiles: the full single-trait sweep + neutral + the 6 named dev profiles +
50 random profiles (seed-robustness sample).

Outputs: results/rq1/e3_patterns.png (all four metrics),
e3_patterns_main.png (the two the report shows), e3_trajectories.csv

Run from ``code/``:  python -m experiments.rq1.run_e3
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from npc_policy import HandAuthoredScorer

from .common import (
    RESULTS, TRAIT_COLORS, TRAIT_SHORT, TRAITS, TRAJ_SEEDS,
    load_cases, personality_of, run_trajectory, setup_style,
    trajectory_metrics, world_for, write_csv,
)

METRICS = ("visit_entropy", "max_share", "distinct", "repeat_rate",
           "action_repeat_rate")
FIG_METRICS = ("visit_entropy", "max_share", "distinct", "repeat_rate")
MAIN_METRICS = ("max_share", "repeat_rate")  # the pair shown in the report
N_RANDOM_SUBSET = 50

# The write-up calls these four metrics variety / routine / coverage /
# stickiness; the panels print the measured quantity, and the caption supplies
# the concept words.
YLABELS = {  # the quantity actually plotted
    "visit_entropy": "entropy of visit shares",
    "max_share": "Proportion of cycles at the\nmost-selected location",
    "distinct": "distinct locations visited",
    "repeat_rate": "Proportion of adjacent cycles\nselecting the same location",
}


def main() -> None:
    setup_style()
    cases = load_cases()
    scorer = HandAuthoredScorer()
    world = world_for("full")

    profiles = (
        [dict(e, group="sweep") for e in cases["profiles"]["sweep"]]
        + [dict(cases["profiles"]["neutral"], group="neutral")]
        + [dict(e, group="named") for e in cases["profiles"]["named"]]
        + [dict(e, group="random") for e in cases["profiles"]["random"][:N_RANDOM_SUBSET]]
    )

    rows = []
    results: dict[str, dict[str, np.ndarray]] = {}
    for entry in profiles:
        p = personality_of(entry)
        per_seed = {m: [] for m in METRICS}
        for seed in TRAJ_SEEDS:
            met = trajectory_metrics(*run_trajectory(scorer, world, p, seed))
            for m in METRICS:
                per_seed[m].append(met[m])
            rows.append([entry["id"], entry["group"], entry.get("trait", ""),
                         entry.get("value", ""), seed]
                        + [f"{met[m]:.4f}" for m in METRICS])
        results[entry["id"]] = {m: np.array(per_seed[m]) for m in METRICS}

    write_csv(RESULTS / "e3_trajectories.csv",
              ["profile", "group", "trait", "value", "seed", *METRICS], rows)

    # ---- figures: each pattern metric vs trait value, one line per trait -----
    # The panel title names the concept, the y label names the quantity plotted,
    # so neither axis is left for the caption to explain.
    sweep = cases["profiles"]["sweep"]
    values = sorted({e["value"] for e in sweep})

    def draw(ax, metric: str) -> None:
        for trait in TRAITS:
            ids = [e["id"] for e in sweep if e["trait"] == trait]
            mean = np.array([results[i][metric].mean() for i in ids])
            sd = np.array([results[i][metric].std() for i in ids])
            ax.plot(values, mean, color=TRAIT_COLORS[trait],
                    label=TRAIT_SHORT[trait])
            ax.fill_between(values, mean - sd, mean + sd,
                            color=TRAIT_COLORS[trait], alpha=0.12,
                            edgecolor="none")
        ax.set_xlabel("trait value (others 0)")
        ax.set_ylabel(YLABELS[metric])
        ax.tick_params(labelbottom=True)  # keep the x values under sharex

    RESULTS.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True)
    for ax, metric in zip(axes.flat, FIG_METRICS):
        draw(ax, metric)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(RESULTS / "e3_patterns.png", bbox_inches="tight")
    plt.close(fig)

    # Report figure: the two metrics the RQ1 write-up reads, at report width.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric in zip(axes, MAIN_METRICS):
        draw(ax, metric)
    axes[0].legend(ncol=5, frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(RESULTS / "e3_patterns_main.png", bbox_inches="tight")
    plt.close(fig)

    # ---- console summary -----------------------------------------------------
    print("E3 named-profile patterns (mean over seeds):")
    print(f"{'profile':<14}" + "".join(f"{m:>20}" for m in METRICS))
    for entry in profiles:
        if entry["group"] not in ("named", "neutral"):
            continue
        r = results[entry["id"]]
        print(f"{entry['id']:<14}"
              + "".join(f"{r[m].mean():>20.3f}" for m in METRICS))
    print(f"figures: {RESULTS / 'e3_patterns.png'}, "
          f"{RESULTS / 'e3_patterns_main.png'}")


if __name__ == "__main__":
    main()
