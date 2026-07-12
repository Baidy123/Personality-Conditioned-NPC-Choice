"""E2 — profile distinguishability (the "distinguishable" part of RQ1).

For 300 random OCEAN profiles, compares personality distance (Euclidean in
trait space) against behavioural distance (JSD between choice distributions,
averaged over all matched contexts, location and action level). If the
representation preserves personality structure, the two distances correlate:
similar profiles behave similarly, dissimilar profiles behave differently.

Statistics: Spearman rho on all pairs + Mantel permutation test (the pairs of
a distance matrix are not independent, so a naive p-value would overstate
significance; Mantel permutes profiles, not pairs).

Outputs: results/rq1/e2_scatter.png, e2_binned.csv, e2_summary.csv

Run from ``code/``:  python -m experiments.rq1.run_e2
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from npc_policy import HandAuthoredScorer

from .common import (
    RESULTS, action_buffer, load_cases, location_buffer, mantel,
    pairwise_jsd, personality_of, setup_style, spearman, world_for, write_csv,
)

N_BINS = 12
MANTEL_PERMS = 999


def main() -> None:
    setup_style()
    cases = load_cases()
    scorer = HandAuthoredScorer()
    profiles = cases["profiles"]["random"]
    vecs = np.array([e["vector"] for e in profiles])
    n = len(profiles)

    # Behavioural distance: mean JSD per decision level, then combined —
    # the split answers how much distinguishability each level carries.
    D_loc = np.zeros((n, n))
    for ctx in cases["location_contexts"]:
        w = world_for(ctx["world"])
        cand = w.resolve()
        buf = location_buffer(w, ctx["memory"])
        P = np.stack([scorer.distribution(personality_of(e), cand, buffer=buf,
                                          level="location") for e in profiles])
        D_loc += pairwise_jsd(P)
    n_loc = len(cases["location_contexts"])
    D_loc /= n_loc

    D_act = np.zeros((n, n))
    for ctx in cases["action_contexts"]:
        w = world_for(ctx["world"])
        acts = w.actions_at(ctx["location"])
        buf = action_buffer(w, ctx["location"], ctx["memory"])
        P = np.stack([scorer.distribution(personality_of(e), acts, buffer=buf,
                                          level="action") for e in profiles])
        D_act += pairwise_jsd(P)
    n_act = len(cases["action_contexts"])
    D_act /= n_act

    n_ctx = n_loc + n_act
    D_beh = (n_loc * D_loc + n_act * D_act) / n_ctx

    D_pers = np.sqrt(((vecs[:, None, :] - vecs[None, :, :]) ** 2).sum(axis=2))

    iu = np.triu_indices(n, k=1)
    x, y = D_pers[iu], D_beh[iu]
    rho = spearman(x, y)
    rho_loc = spearman(x, D_loc[iu])
    rho_act = spearman(x, D_act[iu])
    rho_m, p_m = mantel(D_pers, D_beh, n_perm=MANTEL_PERMS, seed=0)

    # Channel-strength-weighted personality distance: the plain Euclidean
    # metric counts channels a level deliberately leaves silent (A at the
    # location level), which drags the correlation down. Weighting each trait
    # by its measured E1 endpoint TVD is a *consistency check* (weights come
    # from the same scorer via E1, so it is not independent validation) —
    # report both numbers.
    def weighted_rho(level_col: str, D_level: np.ndarray) -> float:
        e1 = (RESULTS / "e1_sensitivity.csv").read_text().strip().splitlines()
        header = e1[0].split(",")
        w = np.array([float(line.split(",")[header.index(level_col)])
                      for line in e1[1:]])
        v = vecs * w
        d = np.sqrt(((v[:, None, :] - v[None, :, :]) ** 2).sum(axis=2))
        return spearman(d[iu], D_level[iu])

    try:
        rho_loc_w = weighted_rho("tvd_rule_location", D_loc)
        rho_act_w = weighted_rho("tvd_rule_action_avg", D_act)
    except FileNotFoundError:  # run_e1 not run yet
        rho_loc_w = rho_act_w = float("nan")

    # Binned means for the trend line and the CSV.
    edges = np.linspace(x.min(), x.max(), N_BINS + 1)
    centers, means, stds, counts = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi if hi < edges[-1] else x <= hi)
        if m.sum() == 0:
            continue
        centers.append(0.5 * (lo + hi))
        means.append(float(y[m].mean()))
        stds.append(float(y[m].std()))
        counts.append(int(m.sum()))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(x, y, s=4, alpha=0.08, color="#0072B2", edgecolors="none",
               rasterized=True)
    ax.plot(centers, means, color="#D55E00", marker="o", markersize=5,
            label="binned mean")
    ax.set_xlabel("personality distance (Euclidean, OCEAN space)")
    ax.set_ylabel("behavioural distance (mean JSD over matched contexts)")
    ax.set_title(f"Distinguishability: {n} random profiles, "
                 f"{n * (n - 1) // 2} pairs, {n_ctx} contexts")
    ax.text(0.02, 0.95, f"Spearman rho = {rho:.3f}\nMantel p = {p_m:.3f} "
            f"({MANTEL_PERMS} perms)", transform=ax.transAxes, va="top",
            fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="0.8"))
    ax.legend(frameon=False, loc="lower right")
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "e2_scatter.png", bbox_inches="tight")
    plt.close(fig)

    write_csv(RESULTS / "e2_binned.csv",
              ["pers_dist_bin_center", "beh_dist_mean", "beh_dist_std", "n_pairs"],
              [[f"{c:.3f}", f"{m:.4f}", f"{s:.4f}", k]
               for c, m, s, k in zip(centers, means, stds, counts)])
    write_csv(RESULTS / "e2_summary.csv",
              ["n_profiles", "n_pairs", "n_contexts", "spearman_rho",
               "spearman_rho_location_only", "spearman_rho_action_only",
               "spearman_rho_location_tvd_weighted",
               "spearman_rho_action_tvd_weighted",
               "mantel_rho", "mantel_p", "mantel_perms"],
              [[n, n * (n - 1) // 2, n_ctx, f"{rho:.4f}",
                f"{rho_loc:.4f}", f"{rho_act:.4f}",
                f"{rho_loc_w:.4f}", f"{rho_act_w:.4f}",
                f"{rho_m:.4f}", f"{p_m:.4f}", MANTEL_PERMS]])

    print(f"E2: Spearman rho = {rho:.3f} combined "
          f"(location-only {rho_loc:.3f} over {n_loc} contexts, "
          f"action-only {rho_act:.3f} over {n_act} contexts), "
          f"Mantel p = {p_m:.3f}, {n * (n - 1) // 2} pairs")
    print(f"    channel-strength-weighted personality distance: "
          f"location {rho_loc_w:.3f}, action {rho_act_w:.3f} "
          "(consistency check, weights from E1)")
    print(f"figure: {RESULTS / 'e2_scatter.png'}")


if __name__ == "__main__":
    main()
