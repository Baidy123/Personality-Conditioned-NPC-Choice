"""E2 — profile distinguishability (the "distinguishable" part of RQ1).

For 300 random OCEAN profiles, compares personality distance (Euclidean in
trait space) against behavioural distance (JSD between choice distributions,
averaged over all matched contexts, location and action level). If the
representation preserves personality structure, the two distances correlate:
similar profiles behave similarly, dissimilar profiles behave differently.

Statistics: Spearman rho on all pairs + Mantel permutation test (the pairs of
a distance matrix are not independent, so a naive p-value would overstate
significance; Mantel permutes profiles, not pairs).

Outputs: results/rq1/e2_scatter.png, e2_binned.csv, e2_summary.csv, and the
combined RQ1 expression figure e1e2_expression.png — panel (a) is E1's
per-trait channel strength (read back from e1_sensitivity.csv, so run_e1 must
have run), panel (b) is this experiment's scatter.

Run from ``code/``:  python -m experiments.rq1.run_e2
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from npc_policy import HandAuthoredScorer

from .common import (
    RESULTS, TRAITS, action_buffer, load_cases, location_buffer, mantel,
    pairwise_jsd, personality_of, plot_expression_strength, read_csv,
    setup_style, spearman, world_for, write_csv,
)

N_BINS = 12
MANTEL_PERMS = 9999

# E1 CSV columns -> the three bars of the expression-strength panel.
STRENGTH_COLS = {"base": "tvd_base_location", "rule": "tvd_rule_location",
                 "act": "tvd_rule_action_empty"}


def read_strength() -> dict[str, dict[str, float]]:
    """E1's endpoint TVDs, keyed as ``plot_expression_strength`` expects."""
    rows = {r["trait"]: r for r in read_csv(RESULTS / "e1_sensitivity.csv")}
    return {t: {k: float(rows[t][c]) for k, c in STRENGTH_COLS.items()}
            for t in TRAITS}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="E2 profile distinguishability")
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw the figures from an existing e2_summary.csv; "
                         "skips the Mantel permutations (the slow part) and "
                         "leaves the CSVs untouched")
    args = ap.parse_args(argv)
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
    if args.figures_only:      # the permutations are ~13 min; the annotation
        prev = read_csv(RESULTS / "e2_summary.csv")[0]      # is what needs them
        rho_m, p_m = float(prev["mantel_rho"]), float(prev["mantel_p"])
    else:
        rho_m, p_m = mantel(D_pers, D_beh, n_perm=MANTEL_PERMS, seed=0)

    # Channel-strength-weighted personality distance: the plain Euclidean
    # metric counts channels a level deliberately leaves silent (A at the
    # location level), which drags the correlation down. Weighting each trait
    # by its measured E1 endpoint TVD is a *consistency check* (weights come
    # from the same scorer via E1, so it is not independent validation) —
    # report both numbers.
    def weighted_rho(level_col: str, D_level: np.ndarray) -> float:
        w = np.array([float(r[level_col])
                      for r in read_csv(RESULTS / "e1_sensitivity.csv")])
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

    def draw_scatter(ax) -> None:
        ax.scatter(x, y, s=4, alpha=0.08, color="#0072B2", edgecolors="none",
                   rasterized=True)
        # Binned means, not a fitted trend: the bars are the within-bin spread
        # of the same points, which is what keeps the polyline readable as a
        # summary of the cloud. (A standard error would be both invisible —
        # thousands of pairs per bin — and unjustified, since pairs are not
        # independent; that non-independence is what the Mantel test handles.)
        ax.errorbar(centers, means, yerr=stds, color="#D55E00", marker="o",
                    markersize=5, capsize=3, elinewidth=1.0,
                    label="binned mean ±1 sd")
        ax.set_xlabel("personality distance (Euclidean, OCEAN space)")
        ax.set_ylabel("behavioural distance (mean JSD over matched contexts)")
        ax.text(0.02, 0.95, f"Spearman rho = {rho:.3f}\nMantel p = {p_m:.4f} "
                f"({MANTEL_PERMS} perms)", transform=ax.transAxes, va="top",
                fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="0.8"))
        ax.legend(frameon=False, loc="lower right")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    draw_scatter(ax)
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "e2_scatter.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------- combined expression figure --
    # E1's per-trait channel strengths and E2's profile-level distinguishability
    # are the two halves of one claim, so the report reads them as one figure.
    # Nothing is printed above either panel: the caption names left and right.
    try:
        strength = read_strength()
    except FileNotFoundError:
        print("    (e1e2_expression.png skipped — run run_e1 first)")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8),
                                 gridspec_kw={"width_ratios": (1.15, 1.0)})
        plot_expression_strength(axes[0], strength)
        draw_scatter(axes[1])
        fig.tight_layout()
        fig.savefig(RESULTS / "e1e2_expression.png", bbox_inches="tight")
        plt.close(fig)

    if args.figures_only:
        print(f"redrawn: {RESULTS / 'e2_scatter.png'}, "
              f"{RESULTS / 'e1e2_expression.png'}")
        return

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
          f"Mantel p = {p_m:.4f}, {n * (n - 1) // 2} pairs")
    print(f"    channel-strength-weighted personality distance: "
          f"location {rho_loc_w:.3f}, action {rho_act_w:.3f} "
          "(consistency check, weights from E1)")
    print(f"figures: {RESULTS / 'e2_scatter.png'}, "
          f"{RESULTS / 'e1e2_expression.png'}")


if __name__ == "__main__":
    main()
