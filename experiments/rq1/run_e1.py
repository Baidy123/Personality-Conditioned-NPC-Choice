"""E1 — single-trait sensitivity (the "personality-conditioned" part of RQ1).

Sweeps each OCEAN trait from -1 to +1 (others 0) over matched contexts and
measures how strongly the choice distribution responds:

  - sensitivity curves: P_rule per location vs trait value (empty memory);
    the figures label it "probability of choosing this location/action" —
    P_rule is the internal name, not a label a reader of the report can decode;
  - expression strength: endpoint TVD per trait, on P_base and on P_rule
    (their gap quantifies how much the N temperature dilutes each trait's
    preference signal), plus the action level (empty memory, mean over the
    six locations' action sets). Context-averaged columns stay in the CSV
    only as channel weights for E2's weighted consistency check;

Outputs: results/rq1/e1_curves.png, e1_action_curves.png, e1_strength.png,
e1_sensitivity.csv

Run from ``code/``:  python -m experiments.rq1.run_e1
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from npc_policy import HandAuthoredScorer

from .common import (
    LOC_COLORS, RESULTS, TRAITS,
    action_buffer, load_cases, location_buffer, personality_of,
    plot_expression_strength, setup_style, tvd, world_for, write_csv,
)


def main() -> None:
    setup_style()
    cases = load_cases()
    scorer = HandAuthoredScorer()
    sweep = cases["profiles"]["sweep"]
    values = sorted({e["value"] for e in sweep})

    def sweep_entry(trait: str, v: float) -> dict:
        return next(e for e in sweep if e["trait"] == trait and e["value"] == v)

    # ---------------------------------------------------------------- curves --
    world = world_for("full")
    locs = world.resolve()
    ids = [o.id for o in locs]

    # One panel per trait. The N entropy curve that used to sit in a sixth panel
    # is not evidence for E1 (whose metric is TVD); it belongs to the RQ2 E-diag
    # overlay, which plots the same teacher curve against the students'.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for ax, trait in zip(axes.flat, TRAITS):
        P = np.stack([
            scorer.distribution(personality_of(sweep_entry(trait, v)), locs, level="location")
            for v in values
        ])
        for k, lid in enumerate(ids):
            ax.plot(values, P[:, k], color=LOC_COLORS[lid], label=lid)
        # No panel title (the report caption numbers and names the figure);
        # the trait a panel sweeps is named on its x-axis instead.
        ax.set_xlabel(f"{trait} value (other traits at 0)")
        ax.set_ylabel("probability of choosing this location")
        # sharex/sharey hide the inner panels' tick values; every panel is read
        # on its own, so put them back rather than leaving bare axis labels.
        ax.tick_params(labelbottom=True, labelleft=True)
    axes.flat[5].axis("off")     # five traits, six slots

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(ids), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "e1_curves.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------- action-level curves ---
    # One diagnostic (trait, location) pair per panel: the action set where the
    # trait's designed channel should be most visible. A gets two panels — its
    # home level is actions (checklist §E2), so it carries the heaviest claim.
    panels = [
        ("openness", "library"),
        ("conscientiousness", "arena"),
        ("extraversion", "tavern"),
        ("agreeableness", "tavern"),
        ("agreeableness", "arena"),
        ("neuroticism", "arena"),
    ]
    # Okabe-Ito order, assigned per panel (action sets differ between panels;
    # identity is carried by each panel's own legend).
    action_cycle = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, (trait, loc_id) in zip(axes.flat, panels):
        acts = world.actions_at(loc_id)
        P = np.stack([
            scorer.distribution(personality_of(sweep_entry(trait, v)), acts, level="action")
            for v in values
        ])
        for k, a in enumerate(acts):
            ax.plot(values, P[:, k], color=action_cycle[k % len(action_cycle)],
                    label=a.id)
        ax.set_ylabel("probability of choosing this action")
        ax.set_xlabel(f"{trait} value, actions at the {loc_id}")
        ax.tick_params(labelbottom=True)  # sharex hides the top row's values
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "e1_action_curves.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------- expression strength (TVD) --
    rows = []
    strength: dict[str, dict[str, float]] = {}
    for trait in TRAITS:
        p_lo, p_hi = (personality_of(sweep_entry(trait, v)) for v in (-1.0, 1.0))

        base_lo = scorer.trace(p_lo, locs, level="location").P_base
        base_hi = scorer.trace(p_hi, locs, level="location").P_base
        tvd_base = tvd(base_lo, base_hi)
        tvd_rule = tvd(scorer.distribution(p_lo, locs, level="location"),
                       scorer.distribution(p_hi, locs, level="location"))

        # context-averaged: all location contexts (memory states + variants)
        ctx_tvds = []
        for ctx in cases["location_contexts"]:
            w = world_for(ctx["world"])
            cand = w.resolve()
            buf = location_buffer(w, ctx["memory"])
            ctx_tvds.append(tvd(
                scorer.distribution(p_lo, cand, buffer=buf, level="location"),
                scorer.distribution(p_hi, cand, buffer=buf, level="location"),
            ))
        tvd_ctx = float(np.mean(ctx_tvds))

        # action level, empty memory: one action set per location, averaged
        # over the six. Memory contexts stay out of E1 — what memory adds is
        # E4's subject.
        act_empty_tvds = []
        for o in locs:
            acts = world.actions_at(o.id)
            act_empty_tvds.append(tvd(
                scorer.distribution(p_lo, acts, level="action"),
                scorer.distribution(p_hi, acts, level="action"),
            ))
        tvd_act = float(np.mean(act_empty_tvds))

        # context-averaged action TVD, kept only because E2's weighted
        # consistency check reads this column as a channel weight.
        act_tvds = []
        for ctx in cases["action_contexts"]:
            w = world_for(ctx["world"])
            acts = w.actions_at(ctx["location"])
            buf = action_buffer(w, ctx["location"], ctx["memory"])
            act_tvds.append(tvd(
                scorer.distribution(p_lo, acts, buffer=buf, level="action"),
                scorer.distribution(p_hi, acts, buffer=buf, level="action"),
            ))
        tvd_act_ctx = float(np.mean(act_tvds))

        strength[trait] = {"base": tvd_base, "rule": tvd_rule,
                           "ctx": tvd_ctx, "act": tvd_act}
        rows.append([trait, f"{tvd_base:.4f}", f"{tvd_rule:.4f}",
                     f"{tvd_ctx:.4f}", f"{tvd_act_ctx:.4f}", f"{tvd_act:.4f}"])

    write_csv(RESULTS / "e1_sensitivity.csv",
              ["trait", "tvd_base_location", "tvd_rule_location",
               "tvd_rule_location_ctx_avg", "tvd_rule_action_avg",
               "tvd_rule_action_empty"], rows)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    plot_expression_strength(ax, strength)
    fig.savefig(RESULTS / "e1_strength.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------- summary ---
    print("E1 expression strength (endpoint TVD):")
    print(f"{'trait':<18} {'P_base':>7} {'P_rule':>7} {'ctx-avg':>8} {'action(empty)':>13}")
    for t in TRAITS:
        s = strength[t]
        print(f"{t:<18} {s['base']:>7.3f} {s['rule']:>7.3f} {s['ctx']:>8.3f} {s['act']:>13.3f}")
    n = strength["neuroticism"]
    print(f"\nN dilution: preference signal {n['base']:.3f} (P_base) -> "
          f"{n['rule']:.3f} (P_rule) = {100 * (1 - n['rule'] / n['base']):.0f}% absorbed"
          " by N's own temperature" if n["base"] > n["rule"] else "")
    print(f"figures: {RESULTS / 'e1_curves.png'}, "
          f"{RESULTS / 'e1_action_curves.png'}, {RESULTS / 'e1_strength.png'}")


if __name__ == "__main__":
    main()
