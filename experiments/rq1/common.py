"""Shared utilities for the RQ1 personality-expression experiments.

RQ1 asks to what extent the representation and decision structure produce
*consistent* and *distinguishable* personality-conditioned choice patterns
(project_flow.md §10 steps 2-3). Four analyses share the matched cases written
by ``gen_cases.py``:

  E1 (run_e1)  single-trait sensitivity        -> "personality-conditioned"
  E2 (run_e2)  profile distinguishability      -> "distinguishable"
  E3 (run_e3)  trajectory-level consistency    -> "consistent patterns"
  E4 (run_e4)  memory-context ablation         -> "the decision structure"

Everything here analyses the hand-authored scorer (v1.2, provisional values);
the perceptual side of RQ1 comes from the later human study.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from npc_policy import DecisionController, HandAuthoredScorer, load_world
from npc_policy.representation import Personality, RecentBuffer
from npc_policy.world import World

CODE = Path(__file__).resolve().parents[2]
DATA = CODE / "data"
CASES = DATA / "rq1_cases"
RESULTS = CODE / "results" / "rq1"

TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
TRAIT_SHORT = {
    "openness": "O", "conscientiousness": "C", "extraversion": "E",
    "agreeableness": "A", "neuroticism": "N",
}

# Okabe-Ito palette (colourblind-safe). Colour follows the entity: every RQ1
# figure uses the same fixed assignment, never cycled defaults.
TRAIT_COLORS = {
    "openness": "#0072B2", "conscientiousness": "#009E73", "extraversion": "#E69F00",
    "agreeableness": "#CC79A7", "neuroticism": "#D55E00",
}
LOC_COLORS = {
    "tavern": "#E69F00", "library": "#0072B2", "chapel": "#CC79A7",
    "market": "#D55E00", "forest": "#009E73", "arena": "#56B4E9",
    "enemy_camp": "#737373",
}
CONDITION_COLORS = {  # E4 ablation conditions
    "none": "#CCCCCC", "location_only": "#56B4E9", "full": "#0072B2",
}

# Trajectory convention: matches the acceptance checklist C3 setting
# (sample mode, selection_temperature 0.1, 50 rounds).
TRAJ_ROUNDS = 50
TRAJ_TEMPERATURE = 0.1
TRAJ_SEEDS = tuple(range(10))


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.color": "0.9",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2.0,
    })


# ---------------------------------------------------------------- cases I/O --
def load_cases() -> dict:
    with open(CASES / "cases.json", encoding="utf-8") as f:
        return json.load(f)


_world_cache: dict[str, World] = {}


def world_for(variant: str) -> World:
    if variant not in _world_cache:
        _world_cache[variant] = load_world(CASES / "worlds" / f"{variant}.json")
    return _world_cache[variant]


def personality_of(entry: dict) -> Personality:
    return Personality(np.array(entry["vector"], dtype=float))


def location_buffer(world: World, ids: list[str], maxlen: int = 3) -> RecentBuffer | None:
    """Build H_L from location ids listed oldest -> newest; None if empty."""
    if not ids:
        return None
    buf = RecentBuffer(maxlen=maxlen)
    for lid in ids:
        buf.push(world.effective_location(lid))
    return buf


def action_buffer(world: World, loc_id: str, ids: list[str], maxlen: int = 3) -> RecentBuffer | None:
    """Build H_A from action ids at ``loc_id``, oldest -> newest; None if empty."""
    if not ids:
        return None
    actions = {a.id: a for a in world.actions_at(loc_id)}
    buf = RecentBuffer(maxlen=maxlen)
    for aid in ids:
        buf.push(actions[aid])
    return buf


# ------------------------------------------------------------------ metrics --
def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    return float(-(p * np.log(p + 1e-12)).sum())


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats (symmetric, bounded by ln 2)."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    kl = lambda a, b: float((a * (np.log(a + 1e-12) - np.log(b + 1e-12))).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def pairwise_jsd(P: np.ndarray) -> np.ndarray:
    """All-pairs JSD for an ``(n, k)`` matrix of distributions -> ``(n, n)``."""
    logp = np.log(P + 1e-12)
    M = 0.5 * (P[:, None, :] + P[None, :, :])
    logm = np.log(M + 1e-12)
    kl_pm = (P[:, None, :] * (logp[:, None, :] - logm)).sum(axis=2)
    kl_qm = (P[None, :, :] * (logp[None, :, :] - logm)).sum(axis=2)
    return 0.5 * kl_pm + 0.5 * kl_qm


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties averaged), enough for Spearman without scipy."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx**2).sum() * (ry**2).sum()))


def mantel(D1: np.ndarray, D2: np.ndarray, n_perm: int = 999, seed: int = 0) -> tuple[float, float]:
    """Mantel test (Spearman statistic) between two square distance matrices.

    Permutes the rows/columns of ``D2`` jointly, which preserves its internal
    structure — the null is "no association between the two distance spaces".
    Returns (rho, one-sided p-value).
    """
    n = D1.shape[0]
    iu = np.triu_indices(n, k=1)
    obs = spearman(D1[iu], D2[iu])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        if spearman(D1[iu], D2[perm][:, perm][iu]) >= obs:
            hits += 1
    return obs, (hits + 1) / (n_perm + 1)


# ------------------------------------------------------------- trajectories --
def run_trajectory(
    scorer: HandAuthoredScorer,
    world: World,
    personality: Personality,
    seed: int,
    rounds: int = TRAJ_ROUNDS,
    memory: str = "full",
) -> list[str]:
    """One sampled behaviour sequence; returns the visited location ids.

    ``memory`` ablates the decision structure's recent-choice context:
      "full"          — both buffers live (the real controller behaviour);
      "location_only" — H_A cleared after every cycle (no action context);
      "none"          — both buffers cleared after every cycle (memory-free).
    """
    if memory not in ("full", "location_only", "none"):
        raise ValueError(f"unknown memory condition {memory!r}")
    ctrl = DecisionController(
        scorer, mode="sample", rng=np.random.default_rng(seed),
        selection_temperature=TRAJ_TEMPERATURE,
    )
    locs = world.resolve()
    visits: list[str] = []
    acts: list[str] = []
    for _ in range(rounds):
        d = ctrl.choose_location(personality, locs)
        a = ctrl.choose_action(personality, world.actions_at(d.option.id))
        visits.append(d.option.id)
        acts.append(a.option.id)
        if memory == "none":
            ctrl.H_L.clear()
            ctrl.H_A.clear()
        elif memory == "location_only":
            ctrl.H_A.clear()
    return visits, acts


def run_action_trajectory(
    scorer: HandAuthoredScorer,
    world: World,
    personality: Personality,
    location_id: str,
    seed: int,
    rounds: int = TRAJ_ROUNDS,
    memory: str = "full",
) -> list[str]:
    """One action sequence at a fixed location; returns the chosen action ids.

    The location choice is skipped, so ``H_A`` is never reset by a move: every
    consecutive pair is a same-location stay and the action-repeat rate has a
    fixed denominator of ``rounds - 1``. A free trajectory instead leaves that
    denominator empty for any profile that rarely stays put, which is why the
    action conditions are compared here rather than on ``run_trajectory``.

    ``memory``: "full" — H_A live; "none" — H_A cleared after every cycle.
    """
    if memory not in ("full", "none"):
        raise ValueError(f"unknown memory condition {memory!r}")
    ctrl = DecisionController(
        scorer, mode="sample", rng=np.random.default_rng(seed),
        selection_temperature=TRAJ_TEMPERATURE,
    )
    pool = world.actions_at(location_id)
    acts: list[str] = []
    for _ in range(rounds):
        a = ctrl.choose_action(personality, pool)
        acts.append(a.option.id)
        if memory == "none":
            ctrl.H_A.clear()
    return acts


def action_repeat_rate(acts: list[str]) -> float:
    """Fraction of consecutive pairs that repeat the action.

    For a fixed-location sequence every pair counts, so the denominator is
    always ``len(acts) - 1``.
    """
    return sum(1 for a, b in zip(acts, acts[1:]) if a == b) / (len(acts) - 1)


def trajectory_metrics(visits: list[str], acts: list[str] | None = None) -> dict[str, float]:
    """Pattern statistics for one sequence (the RQ1 'consistency' measures).

    ``action_repeat_rate`` is defined only over consecutive *same-location*
    cycles (the only case where H_A persists); it is the metric that separates
    the ``location_only`` and ``full`` memory conditions in E4.
    """
    n = len(visits)
    counts = np.array([visits.count(v) for v in sorted(set(visits))], dtype=float)
    shares = counts / n
    repeats = sum(1 for a, b in zip(visits, visits[1:]) if a == b)
    out = {
        "visit_entropy": entropy(shares),
        "max_share": float(shares.max()),
        "distinct": float(len(counts)),
        "repeat_rate": repeats / (n - 1),
    }
    if acts is not None:
        stays = [(acts[i - 1], acts[i]) for i in range(1, n)
                 if visits[i - 1] == visits[i]]
        out["action_repeat_rate"] = (
            sum(1 for a, b in stays if a == b) / len(stays) if stays else 0.0
        )
    return out


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")
