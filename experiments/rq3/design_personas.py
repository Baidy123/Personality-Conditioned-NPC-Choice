"""Design the Study-3 personality set backwards: from behaviour to OCEAN.

``select_personas.py`` picks from hand-written candidates. That fails in a way
the pilot exposed: an author writes profiles that read as distinct people, but
several OCEAN paths converge on the same observable behaviour. Measured over
the six-location world, the trait effects overlap heavily —

    quiet places      <- low E, high N, high C
    cooperative acts  <- high A, high E, high C
    conflict acts     <- low A, low N, low C

so "curious scholar" (high C, low E) and "anxious recluse" (high N, low E)
produce near-identical trails. A participant reading one trail cannot recover
which profile made it, however different the two descriptions look.

This script inverts the problem. It samples OCEAN space, measures where each
profile lands in the *observable* channels a participant can actually read off
a location/action trail, and picks the set that is maximally spread out there.
The description can then be written from the measured profile, which is what
makes description and behaviour agree by construction rather than by hope.

Channels (all in [0, 1], one vector per profile):

    loud_share      fraction of cycles in a busy place (tavern/market/arena)
    conflict_share  fraction of cycles doing a conflict action
    coop_share      fraction of cycles doing a cooperative action
    move_rate       fraction of cycles that change location
    coverage        distinct locations visited / locations available

Selection is maximin on ``distance / noise``, where noise is the spread of a
single profile's own channel vector across seeds: a pair separated by less than
one profile's own run-to-run variation is not separable from a single trail.

Outputs (results/rq3/):
  designed_personas.json    the chosen set: OCEAN, channels, per-channel rank
  designed_personas.csv     every sampled profile that passed the filters
  designed_personas.txt     sample trails for the chosen set, for eyeballing

Run from ``code/``:
  python -m experiments.rq3.design_personas
  python -m experiments.rq3.design_personas --samples 1200 --select 6
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from npc_policy import HandAuthoredScorer, load_world
from npc_policy.representation import Personality

from .common import SequenceSpec, generate_sequence

CODE = Path(__file__).resolve().parents[2]
DATA = CODE / "data"
RESULTS = CODE / "results" / "rq3"
WORLD = DATA / "world.json"
LABELS = DATA / "rq3_labels_zh.json"

# Observable channels. Editing these sets changes what "distinguishable" means,
# so they belong here rather than inline: they are the study's claim about what
# a participant can read off a trail.
LOUD_LOCATIONS = {"tavern", "market", "arena"}
CONFLICT_ACTIONS = {"brawl", "fight", "spar", "haggle", "raid"}
COOP_ACTIONS = {"discuss", "trade", "coach", "confess", "chat"}
CHANNELS = ("loud_share", "conflict_share", "coop_share", "move_rate", "coverage")
CHANNEL_ZH = {
    "loud_share": "热闹地点", "conflict_share": "冲突动作", "coop_share": "合作动作",
    "move_rate": "换地方", "coverage": "地点覆盖",
}

N_SAMPLES = 1000       # OCEAN vectors to try
N_SEEDS = 20           # rolls per sampled profile (channel estimate + its noise)
N_CYCLES = 10          # stimulus length
TEMPERATURE = 0.1      # DecisionController selection_temperature
SELECT = 6
TRAIT_CAP = 0.8        # |trait| bound: a profile of five extremes reads as a caricature
MIN_STRONG = 2         # at least this many traits at |value| >= 0.5 (no flat profiles)
NOISE_CAP = 0.12       # max across-seed channel spread: above this, one trail is
                       # not representative of the profile the participant judges
COVERAGE = 0.5         # every trait must span +/- this across the selected set
GEN_SEED = 20260804


def sample_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform OCEAN vectors, rejecting flat ones (no trait to describe)."""
    out = []
    while len(out) < n:
        v = rng.uniform(-TRAIT_CAP, TRAIT_CAP, size=5)
        if (np.abs(v) >= 0.5).sum() >= MIN_STRONG:
            out.append(np.round(v, 2))
    return np.array(out)


def channel_vector(seq: dict) -> np.ndarray:
    steps = seq["steps"]
    n = len(steps)
    locs = {s["location"] for s in steps}
    return np.array([
        sum(s["location"] in LOUD_LOCATIONS for s in steps) / n,
        sum(s["action"] in CONFLICT_ACTIONS for s in steps) / n,
        sum(s["action"] in COOP_ACTIONS for s in steps) / n,
        sum(bool(s["moved"]) for s in steps) / n,
        len(locs) / 6.0,
    ])


def measure(vec: np.ndarray, scorer, world, n_seeds: int, n_cycles: int,
            temperature: float) -> tuple[np.ndarray, float, list[dict]]:
    """Mean channel vector, its across-seed spread, and the sequences."""
    p = Personality(np.asarray(vec, dtype=float))
    per_seed, seqs = [], []
    for s in range(n_seeds):
        seq = generate_sequence(
            SequenceSpec("x", "scorer", "", "x", p, str(WORLD), n_cycles, s, temperature),
            scorer, world)
        seqs.append(seq)
        per_seed.append(channel_vector(seq))
    M = np.stack(per_seed)
    return M.mean(axis=0), float(np.linalg.norm(M.std(axis=0))), seqs


def force_pool(vectors: np.ndarray, noise: np.ndarray, trait: str,
               threshold: float, size: int) -> list[int]:
    """Least-noisy profiles with ``trait >= threshold``, exempt from the cap.

    Used to seat one profile the noise cap would otherwise always reject. High
    N is the case this exists for: it raises the softmax temperature, so a high-N
    profile is *defined* by unpredictable behaviour and can never be run-to-run
    stable. Excluding it would let the study report per-trait recognisability
    while having quietly removed the trait least likely to be recognised — the
    finding "high-N profiles are hardest to identify, because the model makes
    their behaviour unpredictable" is only available if one is in the set.
    """
    k = "OCEAN".index(trait)
    pool = [i for i in range(len(vectors)) if vectors[i][k] >= threshold]
    if not pool:
        raise SystemExit(f"no sampled profile has {trait} >= {threshold}")
    return sorted(pool, key=lambda i: noise[i])[:size]


def maximin_select(
    mu: np.ndarray, noise: np.ndarray, vectors: np.ndarray, k: int,
    noise_cap: float, coverage: float, forced: list[int] | None = None,
) -> tuple[int, ...]:
    """Farthest-point set in channel space, subject to two hard constraints.

    Objective is *absolute* channel distance, not distance/noise. A ratio
    objective is dominated by its denominator: profiles whose behaviour is a
    fixed loop have near-zero seed-to-seed spread, so their ratio diverges and
    they win every comparison regardless of where they sit. Run-to-run
    stability belongs in the eligibility test (``noise_cap``), not the score.

    ``coverage`` forces every trait to take both a high and a low value in the
    set. Without it the search drifts to uniformly low N, because low N sharpens
    the softmax, which shrinks noise, which the objective would otherwise reward
    — and RQ3 would lose the ability to say anything about neuroticism.
    """
    n = len(mu)
    D = np.linalg.norm(mu[:, None, :] - mu[None, :, :], axis=2)
    eligible = [i for i in range(n) if noise[i] <= noise_cap]
    if len(eligible) < k:
        raise SystemExit(f"only {len(eligible)} profiles under noise cap {noise_cap}")

    # With a forced pool, run the free search for k-1 members once per candidate
    # seat-holder and keep the best whole set.
    if forced:
        best: tuple[int, ...] | None = None
        best_worst = -1.0
        for f in forced:
            rest = _search(D, vectors, [i for i in eligible if i != f],
                           k - 1, coverage, fixed=[f])
            if rest is None:
                continue
            sub = tuple(sorted(rest, key=lambda i: (i != f, i)))
            w = min(D[a, b] for a, b in combinations(sub, 2))
            if w > best_worst:
                best, best_worst = sub, w
        if best is None:
            raise SystemExit("no set containing a forced profile satisfies coverage")
        return best

    got = _search(D, vectors, eligible, k, coverage, fixed=[])
    if got is None:
        raise SystemExit("no set satisfies the coverage constraint")
    return tuple(got)


def _search(D: np.ndarray, vectors: np.ndarray, eligible: list[int], k: int,
            coverage: float, fixed: list[int]) -> list[int] | None:
    """Greedy farthest-point over ``eligible`` plus the immovable ``fixed`` seats,
    repaired for coverage, then improved by swaps that never break it."""

    def covers(sub: list[int]) -> bool:
        V = vectors[sub]
        return bool((V.max(axis=0) >= coverage).all() and (V.min(axis=0) <= -coverage).all())

    def worst(sub: list[int]) -> float:
        return min(D[a, b] for a, b in combinations(sub, 2))

    # Greedy farthest-point seeding among eligible profiles.
    chosen = list(fixed)
    if not chosen:
        i, j = max(combinations(eligible, 2), key=lambda p: D[p[0], p[1]])
        chosen = [i, j]
    while len(chosen) < k + len(fixed):
        rest = [x for x in eligible if x not in chosen]
        if not rest:
            return None
        chosen.append(max(rest, key=lambda x: min(D[x, c] for c in chosen)))

    movable = list(range(len(fixed), len(chosen)))

    # Repair coverage first (accept the distance cost), then improve distance
    # without ever breaking coverage again.
    if not covers(chosen):
        for _ in range((k + 1) * 4):
            if covers(chosen):
                break
            best = None
            for pos in movable:
                for cand in eligible:
                    if cand in chosen:
                        continue
                    trial = list(chosen)
                    trial[pos] = cand
                    if covers(trial) and (best is None or worst(trial) > worst(best)):
                        best = trial
            if best is None:
                break
            chosen = best
        if not covers(chosen):
            return None

    improved = True
    while improved:
        improved = False
        base = worst(chosen)
        for pos in movable:
            for cand in eligible:
                if cand in chosen:
                    continue
                trial = list(chosen)
                trial[pos] = cand
                if covers(trial) and worst(trial) > base:
                    chosen, base, improved = trial, worst(trial), True
    return chosen


def render(seq: dict, labels: dict) -> str:
    L, A = labels["locations"], labels["actions"]
    return " → ".join(f"{L.get(s['location'], s['location'])}"
                      f"（{A.get(s['action'], s['action'])}）" for s in seq["steps"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=N_SAMPLES)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--cycles", type=int, default=N_CYCLES)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--select", type=int, default=SELECT)
    ap.add_argument("--noise-cap", type=float, default=NOISE_CAP,
                    help="reject profiles whose own runs vary more than this")
    ap.add_argument("--coverage", type=float, default=COVERAGE,
                    help="every trait must span +/- this across the set")
    ap.add_argument("--force-trait", default=None, metavar="TRAIT:VALUE",
                    help="seat one profile with TRAIT >= VALUE, exempt from the "
                         "noise cap (e.g. N:0.5)")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rng = np.random.default_rng(GEN_SEED)
    world, scorer = load_world(WORLD), HandAuthoredScorer()
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    vectors = sample_vectors(args.samples, rng)

    print(f"measuring {len(vectors)} profiles x {args.seeds} seeds "
          f"x {args.cycles} cycles (temperature {args.temperature}) ...")
    mu = np.zeros((len(vectors), len(CHANNELS)))
    noise = np.zeros(len(vectors))
    seqs_by_profile: dict[int, list[dict]] = {}
    for i, v in enumerate(vectors):
        mu[i], noise[i], seqs = measure(v, scorer, world, args.seeds,
                                        args.cycles, args.temperature)
        seqs_by_profile[i] = seqs
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(vectors)}")

    forced = None
    if args.force_trait:
        t, v = args.force_trait.split(":")
        forced = force_pool(vectors, noise, t.strip().upper(), float(v), size=20)
        print(f"  forced seat: {len(forced)} candidates with {t.upper()} >= {v}, "
              f"noise {noise[forced].min():.3f}-{noise[forced].max():.3f} "
              f"(cap {args.noise_cap})")

    chosen = maximin_select(mu, noise, vectors, args.select,
                            args.noise_cap, args.coverage, forced=forced)
    D = np.linalg.norm(mu[:, None, :] - mu[None, :, :], axis=2)
    R = D / np.maximum(0.5 * (noise[:, None] + noise[None, :]), 1e-9)
    worst = min(D[a, b] for a, b in combinations(chosen, 2))
    print(f"  {(noise <= args.noise_cap).sum()}/{len(vectors)} profiles under "
          f"noise cap {args.noise_cap}")

    # ---- console -------------------------------------------------------------
    hdr = f"{'#':<4}{'O':>6}{'C':>6}{'E':>6}{'A':>6}{'N':>6}   "
    hdr += "".join(f"{CHANNEL_ZH[c]:>10}" for c in CHANNELS) + f"{'噪声':>8}"
    print(f"\nselected set (worst pair separation = {worst:.2f}):")
    print(hdr)
    for rank, i in enumerate(chosen, 1):
        row = f"{rank:<4}" + "".join(f"{x:>+6.2f}" for x in vectors[i]) + "   "
        row += "".join(f"{mu[i][k]:>10.2f}" for k in range(len(CHANNELS)))
        print(row + f"{noise[i]:>8.3f}")

    print(f"\nchannel spread across the set (how much of each channel is used):")
    for k, c in enumerate(CHANNELS):
        col = mu[list(chosen), k]
        print(f"  {CHANNEL_ZH[c]:<8} [{col.min():.2f}, {col.max():.2f}]   "
              f"range {col.max() - col.min():.2f}")

    print(f"\ntrait spread across the set:")
    V = vectors[list(chosen)]
    for k, t in enumerate("OCEAN"):
        print(f"  {t}  [{V[:, k].min():+.2f}, {V[:, k].max():+.2f}]")

    print(f"\nclosest pairs inside the set:")
    for a, b in sorted(combinations(chosen, 2), key=lambda p: D[p[0], p[1]])[:3]:
        ra, rb = chosen.index(a) + 1, chosen.index(b) + 1
        print(f"  #{ra}-#{rb}  distance {D[a, b]:.2f}  (= {R[a, b]:.1f}x their own run-to-run spread)")

    # ---- files ---------------------------------------------------------------
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "designed_personas.json").write_text(json.dumps({
        "parameters": {"samples": len(vectors), "n_seeds": args.seeds,
                       "n_cycles": args.cycles, "selection_temperature": args.temperature,
                       "trait_cap": TRAIT_CAP, "min_strong_traits": MIN_STRONG,
                       "gen_seed": GEN_SEED, "policy": "hand_authored_scorer"},
        "channels": list(CHANNELS),
        "criterion": "maximin channel-space distance / within-profile channel noise",
        "worst_pair_channel_distance": round(float(worst), 4),
        "noise_cap": args.noise_cap, "coverage": args.coverage,
        "personas": [{
            "rank": r,
            "vector": [float(x) for x in vectors[i]],
            "ocean": dict(zip("OCEAN", (float(x) for x in vectors[i]))),
            "channels": {c: round(float(mu[i][k]), 4) for k, c in enumerate(CHANNELS)},
            "channel_noise": round(float(noise[i]), 4),
        } for r, i in enumerate(chosen, 1)],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(RESULTS / "designed_personas.csv", "w", encoding="utf-8", newline="") as f:
        f.write("idx,O,C,E,A,N," + ",".join(CHANNELS) + ",noise,selected\n")
        for i in range(len(vectors)):
            f.write(f"{i}," + ",".join(f"{x:.2f}" for x in vectors[i]) + ","
                    + ",".join(f"{mu[i][k]:.4f}" for k in range(len(CHANNELS)))
                    + f",{noise[i]:.4f},{int(i in chosen)}\n")

    lines = [f"Designed personas — {args.cycles} cycles, temperature "
             f"{args.temperature}, hand-authored scorer\n"]
    for rank, i in enumerate(chosen, 1):
        ch = "  ".join(f"{CHANNEL_ZH[c]}{mu[i][k]:.0%}" for k, c in enumerate(CHANNELS))
        lines.append(f"#{rank}  OCEAN {vectors[i]}")
        lines.append(f"    {ch}")
        for s in range(3):
            lines.append(f"    seed {s}: {render(seqs_by_profile[i][s], labels)}")
        lines.append("")
    (RESULTS / "designed_personas.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nwritten to {RESULTS}: designed_personas.{{json,csv,txt}}")
    print("Descriptions are written from the measured channels — read the .txt "
          "and check each trail supports the profile you would write for it.")


if __name__ == "__main__":
    main()
