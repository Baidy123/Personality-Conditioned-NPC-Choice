"""E3 — trajectory-level consistency (the "consistent patterns" part of RQ1).

Static distributions can differ while sampled behaviour washes out (or the
reverse: memory dynamics could erase personality differences). E3 runs
50-round controller trajectories (10 seeds each) and measures the *pattern*
statistics — concentration, routine, variety — as a function of each trait.

Profiles: the full single-trait sweep + neutral + the 6 named dev profiles +
50 random profiles (seed-robustness sample).

Outputs: results/rq1/e3_patterns.png, e3_trajectories.csv

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
N_RANDOM_SUBSET = 50


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

    # ---- figure: each pattern metric vs trait value, one line per trait -----
    sweep = cases["profiles"]["sweep"]
    values = sorted({e["value"] for e in sweep})
    titles = {
        "visit_entropy": "variety: entropy of visit shares",
        "max_share": "routine: share of the favourite location",
        "distinct": "coverage: distinct locations visited",
        "repeat_rate": "stickiness: immediate-repeat rate",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, metric in zip(axes.flat, FIG_METRICS):
        for trait in TRAITS:
            ids = [e["id"] for e in sweep if e["trait"] == trait]
            mean = [results[i][metric].mean() for i in ids]
            sd = [results[i][metric].std() for i in ids]
            ax.plot(values, mean, color=TRAIT_COLORS[trait],
                    label=TRAIT_SHORT[trait])
            ax.fill_between(values,
                            np.array(mean) - np.array(sd),
                            np.array(mean) + np.array(sd),
                            color=TRAIT_COLORS[trait], alpha=0.12,
                            edgecolor="none")
        ax.set_title(titles[metric])
        ax.set_xlabel("trait value (others 0)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(f"Trajectory patterns, 50 rounds x {len(TRAJ_SEEDS)} seeds "
                 "(band = ±1 sd over seeds)", y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "e3_patterns.png", bbox_inches="tight")
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
    print(f"figure: {RESULTS / 'e3_patterns.png'}")


if __name__ == "__main__":
    main()
