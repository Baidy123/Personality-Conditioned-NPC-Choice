"""RQ3 figure: does a rating point at the intended persona, trait by trait?

Draws the per-trait half of analyse_survey's section 3b. For every rated trait
the quantity is

    mean |rating - other persona's value|  -  |rating - intended value|

so 0 means the rating sits no closer to the intended persona than to the other
five, and a positive value means it points the right way. The difference form
is what makes the number readable at all: raw distance rewards a participant
who never moves a slider off the midpoint, while a constant answer shifts both
terms together and leaves the difference unchanged.

Only the two personality-conditioned policies are drawn. The agnostic control
has no intended personality to point at, so its bar would not answer this
question; its numbers stay in the text tables.

Produces
  results/rq3/survey/fig_trait_pointing.png / .pdf
  and prints mean, bootstrap CI and permutation p for every bar.

Usage (from code/):
  python -m experiments.rq3.plot_trait_pointing --export "results/rq3/survey/dissertation_ver1.51_August 8, 2026_17.25.tsv"

Previews are kept by default, matching analysis_all21.txt (--drop-previews to
exclude them).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analyse_survey import (SURVEY_DIR, TRAITS, build_trials, load_assignment,
                             load_export, load_personas, perm_test_paired)
from .plot_meeting6 import COLOURS

POLICIES = ["scorer", "nonlinear_2b"]          # the agnostic control is excluded
LEGEND = {"scorer": "hand-crafted scorer", "nonlinear_2b": "learned model"}
TRAIT_NAME = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion",
              "A": "Agreeableness", "N": "Neuroticism"}


def pointing_values(blind: list[dict], personas: dict[str, dict],
                    policy: str, trait: str) -> np.ndarray:
    """One value per trial: distance to the other five minus distance to the intended."""
    i = TRAITS.index(trait)
    vec = {p: np.array(personas[p]["vector"]) for p in personas}
    out = []
    for t in blind:
        if t["policy"] != policy or t["ratings"][trait] is None:
            continue
        r = t["ratings"][trait]
        others = np.mean([abs(r - vec[p][i]) for p in vec if p != t["persona"]])
        out.append(others - abs(r - t["intended"][trait]))
    return np.array(out)


def boot_ci(v: np.ndarray, rng, sims: int = 10_000) -> tuple[float, float]:
    draws = rng.choice(v, size=(sims, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", required=True, type=Path)
    ap.add_argument("--drop-previews", action="store_true",
                    help="exclude Status=4 responses (analysis_all21 keeps them)")
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    _, _, records = load_export(args.export)
    kept = [r for r in records
            if r.get("consent_status") == "accepted" and r.get("version")
            and not (args.drop_previews and r["Status"] == "4")]
    personas = load_personas()
    blind, _ = build_trials(kept, load_assignment(), personas)

    stats = {pol: {tr: pointing_values(blind, personas, pol, tr) for tr in TRAITS}
             for pol in POLICIES}
    # Resampled once, then reused by the table and the bars, so the printed
    # interval is the interval drawn.
    ci = {pol: {tr: boot_ci(stats[pol][tr], rng) for tr in TRAITS}
          for pol in POLICIES}

    print("3b per trait — mean distance to the other five, minus distance to "
          "the intended one")
    print(f"{'policy':<22}{'trait':<20}{'n':>4}{'mean':>9}"
          f"{'95% CI':>18}{'p vs 0':>9}")
    for pol in POLICIES:
        for tr in TRAITS:
            v = stats[pol][tr]
            lo, hi = ci[pol][tr]
            p = perm_test_paired(v, np.zeros_like(v), rng, 8000)
            print(f"{LEGEND[pol]:<22}{TRAIT_NAME[tr]:<20}{len(v):>4}{v.mean():>+9.3f}"
                  f"{f'[{lo:+.3f}, {hi:+.3f}]':>18}{p:>9.4f}")

    xs = np.arange(len(TRAITS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    lo_all, hi_all = [], []
    for k, pol in enumerate(POLICIES):
        off = (k - 0.5) * width
        means = np.array([stats[pol][tr].mean() for tr in TRAITS])
        cis = np.array([ci[pol][tr] for tr in TRAITS])
        lo_all.append(cis[:, 0].min())
        hi_all.append(cis[:, 1].max())
        ax.bar(xs + off, means, width=width, color=COLOURS[pol],
               label=LEGEND[pol], zorder=2)
        ax.errorbar(xs + off, means,
                    yerr=[means - cis[:, 0], cis[:, 1] - means],
                    fmt="none", ecolor="black", elinewidth=1.2, capsize=4, zorder=4)
        # Above the interval's cap, not above the bar: printed at the bar height
        # the error bar runs straight through the digits.
        for x, m, (lo, hi) in zip(xs + off, means, cis):
            ax.text(x, hi + 0.015, f"{m:+.2f}", ha="center", va="bottom",
                    fontsize=8.5)
    # zorder 3: above the bars, below the error bars — a zero line hidden behind
    # a bar cannot be read off, and this line is the whole reference.
    ax.axhline(0, color="black", linewidth=1.1, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([TRAIT_NAME[t] for t in TRAITS], fontsize=9)
    # Two lines: at one line this label is ~70 characters and runs past the top
    # of the axes, where the figure edge clips it.
    ax.set_ylabel("Distance to the other five personas\n"
                  "minus distance to the intended one", fontsize=9)
    ax.text(1.01, 0, "0 = no closer to the\nintended persona",
            transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8.5)
    # Headroom for the value labels and for a legend that clears the tallest
    # interval; matplotlib's own margins measure the bars, not the text.
    ax.set_ylim(min(lo_all) - 0.05, max(hi_all) + 0.20)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper center")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    # Room on the right for the zero-line label, which is drawn outside the axes
    # and so is not measured by tight_layout.
    fig.subplots_adjust(left=0.13, right=0.80, bottom=0.12, top=0.95)

    SURVEY_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(SURVEY_DIR / f"fig_trait_pointing.{ext}", dpi=200)
    plt.close(fig)
    print(f"\nwrote {SURVEY_DIR / 'fig_trait_pointing.png'} (+.pdf)")


if __name__ == "__main__":
    main()
