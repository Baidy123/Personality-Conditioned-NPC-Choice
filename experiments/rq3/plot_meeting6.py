"""Meeting-6 RQ3 figures: per-policy identification and agreement bars.

Supervisor request (meeting 6, 2026-08-18): for every stimulus, take the share
of participants who picked the MOST-CHOSEN description (whichever it is, right
or wrong), average per policy, and draw it as a bar with the spread across
stimuli as the error bar. Shown next to the identification-rate panel so
"reads as having a personality" (agreement) and "reads as the intended
personality" (accuracy) stay two separate claims.

Produces
  results/rq3/survey/fig_id_agreement.png / .pdf   two-panel figure
  results/rq3/survey/meeting6_summary.txt          per-stimulus / per-policy
                                                   tables + the control's modal
                                                   personas ("what is the
                                                   neutral personality?")

Usage (from code/):
  python -m experiments.rq3.plot_meeting6 --export "results/rq3/survey/dissertation_ver1.51_August 8, 2026_17.25.tsv"

Matches analysis_all21.txt: previews are kept by default (--drop-previews to
exclude them, mirroring analyse_survey's opposite flag).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analyse_survey import (POLICY_LABEL, POLICY_ORDER, SURVEY_DIR,
                             build_trials, load_assignment, load_export,
                             load_personas)

COLOURS = {"scorer": "#1f77b4", "nonlinear_2b": "#ff7f0e",
           "agnostic_nonlinear_2b": "#8c8c8c"}

# Bar labels for the report figure only. The text tables keep POLICY_LABEL's
# names so they still line up with analysis_all21.txt; a reader of the figure
# needs words that say what each system is without the codebase's vocabulary.
FIG_LABEL = {"scorer": "hand-crafted\nscorer",
             "nonlinear_2b": "learned\nmodel",
             "agnostic_nonlinear_2b": "personality-agnostic\ncontrol"}


def stimulus_stats(blind: list[dict]) -> list[dict]:
    """One row per stimulus: identification rate, modal-agreement share, modal pick."""
    by_seq: dict[str, list[dict]] = defaultdict(list)
    for t in blind:
        if t["answered"]:
            by_seq[t["sequence_id"]].append(t)
    out = []
    for seq, trials in sorted(by_seq.items()):
        votes = Counter(t["chosen"] for t in trials)
        modal, modal_n = votes.most_common(1)[0]
        out.append({
            "seq": seq,
            "persona": trials[0]["persona"],
            "policy": trials[0]["policy"],
            "n": len(trials),
            "id_rate": sum(t["correct"] for t in trials) / len(trials),
            "agreement": modal_n / len(trials),
            "modal": modal,
            "votes": dict(votes),
        })
    return out


def uniform_agreement_baseline(n: int, k: int, rng, sims: int = 200_000) -> float:
    """E[max category share] when n participants guess uniformly over k options."""
    draws = rng.multinomial(n, [1 / k] * k, size=sims)
    return float(draws.max(axis=1).mean() / n)


def per_policy(rows: list[dict], key: str) -> dict[str, dict]:
    out = {}
    for pol in POLICY_ORDER:
        vals = [r[key] for r in rows if r["policy"] == pol]
        out[pol] = {"vals": vals, "mean": float(np.mean(vals)),
                    "min": min(vals), "max": max(vals)}
    return out


def draw_panel(ax, stats: dict[str, dict], baseline: float, baseline_label: str,
               ylabel: str) -> None:
    xs = np.arange(len(POLICY_ORDER))
    for x, pol in zip(xs, POLICY_ORDER):
        s = stats[pol]
        ax.bar(x, s["mean"], width=0.62, color=COLOURS[pol], zorder=2)
        ax.errorbar(x, s["mean"],
                    yerr=[[s["mean"] - s["min"]], [s["max"] - s["mean"]]],
                    fmt="none", ecolor="black", elinewidth=1.4, capsize=6, zorder=4)
        ax.text(x, 1.02, f"{s['mean']:.2f}", ha="center", va="bottom", fontsize=10)
    # zorder 3: above the bars (a reference line hidden behind a bar cannot be
    # read off), below the error bars and their caps.
    ax.axhline(baseline, color="black", linestyle="--", linewidth=1.1, zorder=3)
    # The label sits outside the axes at the line's height, so it covers neither
    # an error bar nor a bar; main() reserves the right margin for it.
    ax.text(1.015, baseline, baseline_label, transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([FIG_LABEL[p] for p in POLICY_ORDER], fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    # 8.5pt, not 10: the two labels are ~45 characters and at 10pt they run
    # past the top of the axes and get clipped by the figure edge.
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


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
    blind, _ = build_trials(kept, load_assignment(), load_personas())
    rows = stimulus_stats(blind)

    n_per = rows[0]["n"]
    id_stats = per_policy(rows, "id_rate")
    ag_stats = per_policy(rows, "agreement")
    ag_base = uniform_agreement_baseline(n_per, 3, rng)

    # Nothing is printed above either panel: the report caption names the figure
    # and says which panel is which (left = identification, right = agreement).
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.2, 4.0))
    draw_panel(ax_a, id_stats, 1 / 3, "chance (1/3)",
               "Proportion selecting the intended personality description")
    draw_panel(ax_b, ag_stats, ag_base, f"uniform guessing ({ag_base:.2f})",
               "Agreement on most common personality description")
    # Room to the right of each panel for its reference-line label; tight_layout
    # does not measure text drawn outside an axes, so the margins are set here.
    fig.subplots_adjust(left=0.075, right=0.845, bottom=0.15, top=0.94, wspace=0.62)
    for ext in ("png", "pdf"):
        fig.savefig(SURVEY_DIR / f"fig_id_agreement.{ext}", dpi=200)

    lines = [f"RQ3 meeting-6 figures — {args.export.name}",
             f"responses used: {len(kept)}  (participants per stimulus: {n_per})",
             "",
             "per policy (mean over its 6 stimuli; error bar in the figure = "
             "min..max over those 6)",
             f"{'policy':<26}{'id mean':>9}{'id range':>15}{'agr mean':>10}{'agr range':>15}"]
    for pol in POLICY_ORDER:
        i, a = id_stats[pol], ag_stats[pol]
        lines.append(f"{POLICY_LABEL[pol]:<26}{i['mean']:>9.3f}"
                     f"{i['min']:>7.2f}–{i['max']:<7.2f}{a['mean']:>9.3f}"
                     f"{a['min']:>7.2f}–{a['max']:<7.2f}")
    lines += ["",
              f"agreement baseline: E[modal share] under uniform guessing over 3 options, "
              f"n = {n_per} -> {ag_base:.3f} (simulated, 200k draws). Chance for "
              f"identification = 1/3. The two are different because the maximum of a "
              f"small sample sits above its mean.",
              "",
              "per stimulus",
              f"{'seq':<6}{'persona':<9}{'policy':<26}{'id':>6}{'agree':>7}   modal (votes)"]
    for r in sorted(rows, key=lambda r: (POLICY_ORDER.index(r["policy"]), r["persona"])):
        votes = " ".join(f"{k}:{v}" for k, v in
                         sorted(r["votes"].items(), key=lambda kv: -kv[1]))
        lines.append(f"{r['seq']:<6}{r['persona']:<9}{POLICY_LABEL[r['policy']]:<26}"
                     f"{r['id_rate']:>6.2f}{r['agreement']:>7.2f}   {r['modal']} ({votes})")
    lines += ["",
              "what does the agnostic control read as? (supervisor's 'neutral "
              "personality' question)",
              "per-stimulus modal picks under the control:"]
    ctl = [r for r in rows if r["policy"] == "agnostic_nonlinear_2b"]
    for r in sorted(ctl, key=lambda r: r["persona"]):
        lines.append(f"  intended {r['persona']}: modal pick {r['modal']} "
                     f"({r['agreement']:.2f} of {r['n']})")
    pooled = Counter()
    for r in ctl:
        pooled.update(r["votes"])
    lines.append("pooled control votes over all its trials: "
                 + " ".join(f"{k}:{v}" for k, v in pooled.most_common()))
    out = SURVEY_DIR / "meeting6_summary.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {SURVEY_DIR / 'fig_id_agreement.png'} (+.pdf) and {out}")


if __name__ == "__main__":
    main()
