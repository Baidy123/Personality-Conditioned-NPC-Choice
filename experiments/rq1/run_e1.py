"""E1 — single-trait sensitivity (the "personality-conditioned" part of RQ1).

Sweeps each OCEAN trait from -1 to +1 (others 0) over matched contexts and
measures how strongly the choice distribution responds:

  - sensitivity curves: P_rule per location vs trait value (empty memory);
  - expression strength: endpoint TVD per trait, on P_base and on P_rule
    (their gap quantifies how much the N temperature dilutes each trait's
    preference signal), plus the context-averaged and action-level values;
  - the N entropy curve (bidirectional temperature, checklist A5 formalised).

Outputs: results/rq1/e1_curves.png, e1_strength.png, e1_sensitivity.csv

Run from ``code/``:  python -m experiments.rq1.run_e1
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from npc_policy import HandAuthoredScorer

from .common import (
    LOC_COLORS, RESULTS, TRAIT_COLORS, TRAIT_SHORT, TRAITS,
    action_buffer, entropy, load_cases, location_buffer, personality_of,
    setup_style, tvd, world_for, write_csv,
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

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=False)
    for ax, trait in zip(axes.flat, TRAITS):
        P = np.stack([
            scorer.distribution(personality_of(sweep_entry(trait, v)), locs, level="location")
            for v in values
        ])
        for k, lid in enumerate(ids):
            ax.plot(values, P[:, k], color=LOC_COLORS[lid], label=lid)
        ax.set_title(f"{TRAIT_SHORT[trait]} ({trait})")
        ax.set_ylabel("P_rule")
        ax.set_xlabel("trait value")

    ax = axes.flat[5]  # N temperature: entropy of P_rule over the N sweep
    ent = [entropy(scorer.distribution(personality_of(sweep_entry("neuroticism", v)),
                                       locs, level="location")) for v in values]
    ax.plot(values, ent, color=TRAIT_COLORS["neuroticism"])
    ax.set_title("N temperature: entropy of P_rule")
    ax.set_ylabel("entropy (nats)")
    ax.set_xlabel("N value")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(ids), frameon=False)
    fig.suptitle("E1 — location-choice sensitivity to single traits (empty memory)", y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "e1_curves.png", bbox_inches="tight")
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

        # action level: mean endpoint TVD over all action contexts
        act_tvds = []
        for ctx in cases["action_contexts"]:
            w = world_for(ctx["world"])
            acts = w.actions_at(ctx["location"])
            buf = action_buffer(w, ctx["location"], ctx["memory"])
            act_tvds.append(tvd(
                scorer.distribution(p_lo, acts, buffer=buf, level="action"),
                scorer.distribution(p_hi, acts, buffer=buf, level="action"),
            ))
        tvd_act = float(np.mean(act_tvds))

        strength[trait] = {"base": tvd_base, "rule": tvd_rule,
                           "ctx": tvd_ctx, "act": tvd_act}
        rows.append([trait, f"{tvd_base:.4f}", f"{tvd_rule:.4f}",
                     f"{tvd_ctx:.4f}", f"{tvd_act:.4f}"])

    write_csv(RESULTS / "e1_sensitivity.csv",
              ["trait", "tvd_base_location", "tvd_rule_location",
               "tvd_rule_location_ctx_avg", "tvd_rule_action_avg"], rows)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(TRAITS))
    bars = [("P_base (location)", "base", "#BBBBBB"),
            ("P_rule (location)", "rule", "#0072B2"),
            ("P_rule (action, avg)", "act", "#E69F00")]
    for i, (label, key, color) in enumerate(bars):
        vals = [strength[t][key] for t in TRAITS]
        ax.bar(x + (i - 1) * 0.26, vals, width=0.24, color=color, label=label)
    ax.axhline(0.15, color="0.4", linewidth=1, linestyle="--")
    ax.text(len(TRAITS) - 0.5, 0.155, "B1 floor (0.15)", fontsize=8, color="0.4")
    ax.set_xticks(x, [TRAIT_SHORT[t] for t in TRAITS])
    ax.set_ylabel("endpoint TVD (trait -1 vs +1)")
    ax.set_title("E1 — expression strength per trait channel")
    ax.legend(frameon=False)
    fig.savefig(RESULTS / "e1_strength.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------- summary ---
    print("E1 expression strength (endpoint TVD):")
    print(f"{'trait':<18} {'P_base':>7} {'P_rule':>7} {'ctx-avg':>8} {'action':>7}")
    for t in TRAITS:
        s = strength[t]
        print(f"{t:<18} {s['base']:>7.3f} {s['rule']:>7.3f} {s['ctx']:>8.3f} {s['act']:>7.3f}")
    n = strength["neuroticism"]
    print(f"\nN dilution: preference signal {n['base']:.3f} (P_base) -> "
          f"{n['rule']:.3f} (P_rule) = {100 * (1 - n['rule'] / n['base']):.0f}% absorbed"
          " by N's own temperature" if n["base"] > n["rule"] else "")
    print(f"figures: {RESULTS / 'e1_curves.png'}, {RESULTS / 'e1_strength.png'}")


if __name__ == "__main__":
    main()
