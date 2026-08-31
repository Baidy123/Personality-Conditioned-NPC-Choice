"""Select the Study-3 personality set by measured behavioural separation.

The human study shows each participant one 10-cycle sequence and asks which of
three personality descriptions produced it. That task is only meaningful if the
profiles behind the descriptions actually behave differently *at the length the
participant sees*. Two profiles that differ in OCEAN numbers but produce
near-identical 10-step sequences turn their trials into a coin flip, dragging
every policy's identification rate down for a reason that has nothing to do
with the policies. This script finds such pairs before the questionnaire is
built.

Two levels of evidence, both against the hand-authored scorer (the reference
implementation of the representation — a pair it cannot separate will not be
separated by a policy trained to imitate it):

  distribution level  mean JSD between choice distributions over the RQ1
                      matched contexts. Ceiling: what the representation can
                      express when no sampling noise intervenes.

  sequence level      the number that decides. Each profile is rolled S times
                      at the stimulus length; a sequence is summarised as its
                      joint (location, action) share vector, and

                          separation(i,j) = between(i,j) / mean(within_i, within_j)

                      where ``within`` is the mean JSD between two sequences of
                      the *same* profile (different seeds) and ``between`` is
                      the mean JSD across profiles. A ratio near 1 means the
                      gap between two profiles is no larger than the gap two
                      rolls of one profile already show — a participant seeing
                      a single sequence cannot tell them apart, however
                      different the underlying vectors are.

The final set maximises the *worst* pair (maximin): a persona set is only as
good as its most confusable pair, because that pair is what produces the
guessing trials.

Outputs (results/rq3/):
  persona_separation.csv    every pair, both levels
  persona_separation.png    sequence-level ratio heatmap, selected set marked
  persona_examples.txt      sample sequences as questionnaire text, for eyeballing
  persona_selection.json    the chosen set + runner-up sets + parameters

Run from ``code/``:
  python -m experiments.rq3.select_personas
  python -m experiments.rq3.select_personas --select 6 --seeds 30 --temperature 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from npc_policy import HandAuthoredScorer, load_world
from npc_policy.representation import Personality

from ..rq1.common import CASES, action_buffer, location_buffer, pairwise_jsd
from .common import SequenceSpec, generate_sequence

CODE = Path(__file__).resolve().parents[2]
DATA = CODE / "data"
RESULTS = CODE / "results" / "rq3"

CANDIDATES = DATA / "rq3_persona_candidates.json"
LABELS = DATA / "rq3_labels_zh.json"
WORLD = DATA / "world.json"

# Stimulus conditions — these must match what gen_sequences.py will later use,
# otherwise the selection is made under conditions the participants never see.
N_CYCLES = 10          # sequence length shown to a participant
N_SEEDS = 30           # rolls per profile; only affects the noise estimate
SELECT = 6             # profiles to keep
TEMPERATURE = 0.1      # DecisionController selection_temperature (tuning-log setting)
N_EXAMPLES = 3         # sample sequences printed per profile
COVERAGE = 0.5         # --enforce-coverage: each trait must span +/- this in the set


# --------------------------------------------------------------- candidates --
def load_candidates() -> list[dict]:
    return json.loads(CANDIDATES.read_text(encoding="utf-8"))["candidates"]


def check_design_rules(cands: list[dict]) -> list[str]:
    """R1/R2 from the candidate file, as warnings rather than hard failures —
    a rule-violating candidate is allowed to compete and be rejected on data."""
    warnings = []
    for c in cands:
        v = np.asarray(c["vector"])
        if (np.abs(v) >= 0.6).sum() < 2:
            warnings.append(
                f"R1: {c['id']} ({c['archetype']}) has "
                f"{(np.abs(v) >= 0.6).sum()} trait(s) at |value| >= 0.6")
    V = np.array([c["vector"] for c in cands])
    traits = ["O", "C", "E", "A", "N"]
    for k, t in enumerate(traits):
        if V[:, k].max() < 0.5 or V[:, k].min() > -0.5:
            warnings.append(
                f"R2: trait {t} spans only [{V[:, k].min():+.2f}, {V[:, k].max():+.2f}]")
    return warnings


# -------------------------------------------------------- distribution level --
def distribution_distances(cands: list[dict], scorer: HandAuthoredScorer) -> np.ndarray | None:
    """Mean pairwise JSD over the RQ1 matched contexts; None if they are absent.

    Reuses the RQ1 case file so the two studies talk about the same contexts.
    Location and action contexts are pooled by count, as in run_e2.
    """
    case_file = CASES / "cases.json"
    if not case_file.exists():
        return None
    cases = json.loads(case_file.read_text(encoding="utf-8"))
    worlds: dict[str, object] = {}

    def world_for(name: str):
        if name not in worlds:
            worlds[name] = load_world(CASES / "worlds" / f"{name}.json")
        return worlds[name]

    pers = [Personality(np.asarray(c["vector"], dtype=float)) for c in cands]
    n = len(pers)

    D_loc = np.zeros((n, n))
    for ctx in cases["location_contexts"]:
        w = world_for(ctx["world"])
        cand = w.resolve()
        buf = location_buffer(w, ctx["memory"])
        P = np.stack([scorer.distribution(p, cand, buffer=buf, level="location")
                      for p in pers])
        D_loc += pairwise_jsd(P)
    n_loc = len(cases["location_contexts"])

    D_act = np.zeros((n, n))
    for ctx in cases["action_contexts"]:
        w = world_for(ctx["world"])
        acts = w.actions_at(ctx["location"])
        buf = action_buffer(w, ctx["location"], ctx["memory"])
        P = np.stack([scorer.distribution(p, acts, buffer=buf, level="action")
                      for p in pers])
        D_act += pairwise_jsd(P)
    n_act = len(cases["action_contexts"])

    return (D_loc + D_act) / (n_loc + n_act)


# ------------------------------------------------------------ sequence level --
def option_index(world) -> dict[tuple[str, str], int]:
    """Fixed (location, action) -> column order for sequence signatures."""
    pairs = [(loc.id, a.id) for loc in world.resolve() for a in world.actions_at(loc.id)]
    return {p: i for i, p in enumerate(pairs)}


def signature(seq: dict, index: dict[tuple[str, str], int]) -> np.ndarray:
    """A sequence as its joint (location, action) share vector.

    The joint form carries both what the participant reads off the trail: which
    places, and what was done there. Marginal location shares would erase the
    action level entirely, where agreeableness and conflict live.
    """
    v = np.zeros(len(index))
    for step in seq["steps"]:
        v[index[(step["location"], step["action"])]] += 1.0
    return v / v.sum()


def sequence_separation(
    cands: list[dict], scorer: HandAuthoredScorer, world, n_seeds: int,
    n_cycles: int, temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[dict]]]:
    """Returns (separation_ratio, between, within, sequences[i][seed])."""
    index = option_index(world)
    n = len(cands)
    sigs = np.zeros((n, n_seeds, len(index)))
    seqs: list[list[dict]] = []

    for i, c in enumerate(cands):
        personality = Personality(np.asarray(c["vector"], dtype=float))
        per_profile = []
        for s in range(n_seeds):
            spec = SequenceSpec(
                sequence_id=f"{c['id']}_s{s}", policy_name="scorer", checkpoint="",
                personality_name=c["id"], personality=personality,
                world_path=str(WORLD), n_cycles=n_cycles, seed=s,
                selection_temperature=temperature,
            )
            seq = generate_sequence(spec, scorer, world)
            per_profile.append(seq)
            sigs[i, s] = signature(seq, index)
        seqs.append(per_profile)

    # One JSD pass over all n*n_seeds signatures, then read off blocks.
    flat = sigs.reshape(n * n_seeds, -1)
    D = pairwise_jsd(flat).reshape(n, n_seeds, n, n_seeds)

    within = np.zeros(n)
    for i in range(n):
        block = D[i, :, i, :]
        iu = np.triu_indices(n_seeds, k=1)
        within[i] = block[iu].mean()

    between = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            between[i, j] = D[i, :, j, :].mean() if i != j else within[i]

    denom = 0.5 * (within[:, None] + within[None, :])
    ratio = between / np.maximum(denom, 1e-12)
    np.fill_diagonal(ratio, 1.0)
    return ratio, between, within, seqs


# ----------------------------------------------------------------- selection --
def trait_span(vectors: np.ndarray, sub: tuple[int, ...]) -> np.ndarray:
    """Per-trait (min, max) over a subset -> (5, 2)."""
    V = vectors[list(sub)]
    return np.stack([V.min(axis=0), V.max(axis=0)], axis=1)


def covers_all_traits(vectors: np.ndarray, sub: tuple[int, ...], thr: float) -> bool:
    """Every trait takes both a high (>= thr) and a low (<= -thr) value in the set.

    Without this, maximin can legitimately drop a whole trait: profiles whose
    dominant trait makes their *own* behaviour noisy (high N raises the softmax
    temperature) have a large within-profile spread, so every ratio involving
    them is small and they lose every subset comparison. The set that results
    is the most identifiable one, and it silently removes the ability to say
    anything about that trait in RQ3.
    """
    span = trait_span(vectors, sub)
    return bool((span[:, 1] >= thr).all() and (span[:, 0] <= -thr).all())


def maximin_subset(
    ratio: np.ndarray, k: int, top: int = 5,
    vectors: np.ndarray | None = None, coverage: float | None = None,
) -> list[tuple[float, tuple[int, ...]]]:
    """All k-subsets ranked by their worst pairwise ratio; best first.

    With ``coverage`` set, only subsets spanning every trait are eligible.
    """
    n = ratio.shape[0]
    subs = combinations(range(n), k)
    if coverage is not None:
        assert vectors is not None
        subs = (s for s in subs if covers_all_traits(vectors, s, coverage))
    scored = [
        (min(ratio[i, j] for i, j in combinations(sub, 2)), sub)
        for sub in subs
    ]
    scored.sort(key=lambda t: -t[0])
    return scored[:top]


# ------------------------------------------------------------------- outputs --
def render(seq: dict, labels: dict) -> str:
    """One sequence as the questionnaire text: 地点（动作）→ 地点（动作）→ ..."""
    L, A = labels["locations"], labels["actions"]
    return " → ".join(
        f"{L.get(s['location'], s['location'])}（{A.get(s['action'], s['action'])}）"
        for s in seq["steps"]
    )


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


def heatmap(ratio: np.ndarray, cands: list[dict], chosen: tuple[int, ...], path: Path) -> None:
    n = len(cands)
    labels = [f"{c['id']}\n{c['archetype']}" for c in cands]
    fig, ax = plt.subplots(figsize=(1.0 * n + 2.5, 1.0 * n + 1.5))
    m = ratio.copy()
    np.fill_diagonal(m, np.nan)
    im = ax.imshow(m, cmap="viridis", vmin=1.0)
    ax.set_xticks(range(n), labels, fontsize=7)
    ax.set_yticks(range(n), labels, fontsize=7)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            both = i in chosen and j in chosen
            ax.text(j, i, f"{ratio[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if ratio[i, j] < m[~np.isnan(m)].mean() else "black",
                    fontweight="bold" if both else "normal")
    for i in chosen:
        ax.add_patch(plt.Rectangle((i - 0.5, -0.5), 1, n, fill=False,
                                   edgecolor="#D55E00", lw=1.5))
        ax.add_patch(plt.Rectangle((-0.5, i - 0.5), n, 1, fill=False,
                                   edgecolor="#D55E00", lw=1.5))
    fig.colorbar(im, ax=ax, shrink=0.75,
                 label="separation ratio (between / within)")
    # Nothing above the axes: the caption says that the orange frames mark the
    # selected set and that a ratio of 1.0 is indistinguishable from sampling
    # noise.
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--select", type=int, default=SELECT, help="how many profiles to keep")
    ap.add_argument("--seeds", type=int, default=N_SEEDS, help="rolls per profile")
    ap.add_argument("--cycles", type=int, default=N_CYCLES, help="sequence length")
    ap.add_argument("--temperature", type=float, default=TEMPERATURE,
                    help="DecisionController selection_temperature")
    ap.add_argument("--enforce-coverage", action="store_true",
                    help=f"require every trait to span +/-{COVERAGE} in the selected set")
    args = ap.parse_args()

    # Console is cp936 on this machine; the Chinese descriptions would mojibake.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cands = load_candidates()
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    world = load_world(WORLD)
    scorer = HandAuthoredScorer()
    n = len(cands)

    for w in check_design_rules(cands):
        print(f"  warning  {w}")

    print(f"rolling {n} candidates x {args.seeds} seeds x {args.cycles} cycles "
          f"(temperature {args.temperature}) ...")
    ratio, between, within, seqs = sequence_separation(
        cands, scorer, world, args.seeds, args.cycles, args.temperature)
    D_dist = distribution_distances(cands, scorer)
    if D_dist is None:
        print("  note: data/rq1_cases/cases.json missing — distribution level skipped "
              "(run: python -m experiments.rq1.gen_cases)")

    vectors = np.array([c["vector"] for c in cands], dtype=float)
    ranked = maximin_subset(
        ratio, args.select, vectors=vectors,
        coverage=COVERAGE if args.enforce_coverage else None)
    if not ranked:
        raise SystemExit(
            f"no {args.select}-subset spans every trait at +/-{COVERAGE}; "
            "add candidates or lower COVERAGE")
    best_score, chosen = ranked[0]

    # ---- console ------------------------------------------------------------
    print(f"\nwithin-profile sequence variability (sampling noise floor):")
    for i, c in enumerate(cands):
        print(f"  {c['id']:<4} {c['archetype']:<16} within = {within[i]:.4f}")

    print(f"\nmost confusable pairs (whole candidate set):")
    pairs = sorted(((ratio[i, j], i, j) for i, j in combinations(range(n), 2)))
    for r, i, j in pairs[:5]:
        print(f"  {cands[i]['id']}-{cands[j]['id']:<4} ratio {r:5.2f}   "
              f"{cands[i]['archetype']} / {cands[j]['archetype']}")

    print(f"\nselected set (maximin worst-pair ratio = {best_score:.2f}):")
    for i in chosen:
        print(f"  {cands[i]['id']:<4} {cands[i]['archetype']:<16} "
              f"{cands[i]['description_zh']}")
    dropped = [cands[i]['id'] for i in range(n) if i not in chosen]
    print(f"  dropped: {', '.join(dropped)}")

    # Trait coverage of the chosen set: a trait that ends up one-sided cannot be
    # asked about in RQ3, whatever the identification rate turns out to be.
    span = trait_span(vectors, chosen)
    print(f"\ntrait coverage of the selected set"
          f"{' (enforced)' if args.enforce_coverage else ''}:")
    for k, t in enumerate(("O", "C", "E", "A", "N")):
        flag = "" if (span[k, 1] >= COVERAGE and span[k, 0] <= -COVERAGE) else "   <- one-sided"
        print(f"  {t}  [{span[k, 0]:+.2f}, {span[k, 1]:+.2f}]{flag}")

    print(f"\nrunner-up sets:")
    for score, sub in ranked[1:]:
        print(f"  {score:5.2f}  {', '.join(cands[i]['id'] for i in sub)}")

    # ---- files --------------------------------------------------------------
    rows = []
    for i, j in combinations(range(n), 2):
        rows.append([
            cands[i]["id"], cands[j]["id"], cands[i]["archetype"], cands[j]["archetype"],
            f"{ratio[i, j]:.4f}", f"{between[i, j]:.4f}",
            f"{within[i]:.4f}", f"{within[j]:.4f}",
            f"{D_dist[i, j]:.4f}" if D_dist is not None else "",
            f"{np.linalg.norm(np.array(cands[i]['vector']) - np.array(cands[j]['vector'])):.4f}",
            int(i in chosen and j in chosen),
        ])
    write_csv(RESULTS / "persona_separation.csv",
              ["a", "b", "archetype_a", "archetype_b", "separation_ratio",
               "between_jsd", "within_a", "within_b", "distribution_jsd",
               "ocean_distance", "both_selected"], rows)

    heatmap(ratio, cands, chosen, RESULTS / "persona_separation.png")

    lines = [f"Sample sequences — {args.cycles} cycles, temperature {args.temperature}, "
             f"hand-authored scorer\n"]
    for i, c in enumerate(cands):
        mark = "SELECTED" if i in chosen else "dropped "
        lines.append(f"[{mark}] {c['id']} {c['archetype']} — {c['description_zh']}")
        lines.append(f"           OCEAN {np.asarray(c['vector'])}")
        for s in range(min(N_EXAMPLES, args.seeds)):
            lines.append(f"    seed {s}: {render(seqs[i][s], labels)}")
        lines.append("")
    (RESULTS / "persona_examples.txt").write_text("\n".join(lines), encoding="utf-8")

    (RESULTS / "persona_selection.json").write_text(json.dumps({
        "parameters": {"n_cycles": args.cycles, "n_seeds": args.seeds,
                       "selection_temperature": args.temperature,
                       "select": args.select, "policy": "hand_authored_scorer",
                       "world": str(WORLD),
                       "enforce_coverage": args.enforce_coverage,
                       "coverage_threshold": COVERAGE},
        "criterion": "maximin pairwise sequence-level separation ratio"
                     + (f", subject to every trait spanning +/-{COVERAGE}"
                        if args.enforce_coverage else ""),
        "selected": [cands[i]["id"] for i in chosen],
        "worst_pair_ratio": round(float(best_score), 4),
        "trait_span": {t: [round(float(span[k, 0]), 3), round(float(span[k, 1]), 3)]
                       for k, t in enumerate(("O", "C", "E", "A", "N"))},
        "within_profile_noise": {c["id"]: round(float(within[i]), 4)
                                 for i, c in enumerate(cands)},
        "dropped": dropped,
        "runner_up_sets": [{"ids": [cands[i]["id"] for i in sub],
                            "worst_pair_ratio": round(float(sc), 4)}
                           for sc, sub in ranked[1:]],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nwritten to {RESULTS}:")
    for f in ("persona_separation.csv", "persona_separation.png",
              "persona_examples.txt", "persona_selection.json"):
        print(f"  {f}")
    print("\nRead persona_examples.txt before accepting the set: if you cannot "
          "tell two selected profiles apart by eye, participants will not either.")


if __name__ == "__main__":
    main()
