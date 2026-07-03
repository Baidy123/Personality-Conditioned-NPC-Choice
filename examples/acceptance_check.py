"""Phase-2 acceptance scorecard (docs/specs/2026-07-02-phase2-acceptance-checklist.md).

Evaluates checks A1-A6 (directions), B1-B4 (magnitudes), C1-C4 (memory
behaviour), and F1-F6 (combined BG3-inspired dev profiles) against the current
tables and coefficients, and prints a pass/fail report with measured values.
Section D (structural regressions) is covered separately by ``pytest tests``.

Run from the ``code`` folder:
    python -m examples.acceptance_check
Exit code: number of failed checks (0 = all green).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from npc_policy import (
    DecisionController,
    HandAuthoredScorer,
    load_personalities,
    load_world,
)
from npc_policy.representation import Personality, RecentBuffer

DATA = Path(__file__).resolve().parent.parent / "data"
SWEEP = (-1.0, -0.5, 0.0, 0.5, 1.0)
TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")

SOCIAL = {"tavern", "market", "arena"}
QUIET = {"library", "chapel", "forest"}
STRUCT = {"library", "chapel"}

results: list[tuple[str, bool, str]] = []


def check(cid: str, passed: bool, detail: str) -> None:
    results.append((cid, bool(passed), detail))


def entropy(p: np.ndarray) -> float:
    return float(-(p * np.log(p + 1e-12)).sum())


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def strictly_increasing(xs: list[float]) -> bool:
    return all(b > a for a, b in zip(xs, xs[1:]))


def main(scorer: HandAuthoredScorer | None = None) -> int:
    world = load_world(DATA / "world.json")
    npcs = load_personalities(DATA / "personalities.json")
    scorer = scorer if scorer is not None else HandAuthoredScorer()
    results.clear()
    locs = world.resolve()
    ids = [o.id for o in locs]
    idx = {i: k for k, i in enumerate(ids)}

    def sweep_profiles(trait: str):
        return [Personality.from_traits(**{trait: v}) for v in SWEEP]

    def p_base(p: Personality) -> np.ndarray:
        return scorer.trace(p, locs, level="location").P_base

    def p_rule(p: Personality, buffer: RecentBuffer | None = None) -> np.ndarray:
        return scorer.distribution(p, locs, buffer=buffer, level="location")

    def p_actions(p: Personality, loc_id: str, buffer: RecentBuffer | None = None) -> dict[str, float]:
        acts = world.actions_at(loc_id)
        d = scorer.distribution(p, acts, buffer=buffer, level="action")
        return {a.id: float(x) for a, x in zip(acts, d)}

    def group(d: np.ndarray, names: set[str]) -> float:
        return float(sum(d[idx[n]] for n in names if n in idx))

    # ---------------- A. directions -------------------------------------------
    xs = [group(p_base(p), SOCIAL) for p in sweep_profiles("extraversion")]
    top_lo = ids[int(np.argmax(p_base(sweep_profiles("extraversion")[0])))]
    top_hi = ids[int(np.argmax(p_base(sweep_profiles("extraversion")[-1])))]
    check("A1 E->social", strictly_increasing(xs) and top_lo in QUIET and top_hi in SOCIAL,
          f"P(SOCIAL)={['%.2f' % x for x in xs]}, top(E-1)={top_lo}, top(E+1)={top_hi}")

    xs = [float(p_base(p)[idx["forest"]]) for p in sweep_profiles("openness")]
    check("A2 O->forest", strictly_increasing(xs), f"P(forest)={['%.2f' % x for x in xs]}")

    xs = [group(p_base(p), STRUCT) for p in sweep_profiles("conscientiousness")]
    ys = [float(p_base(p)[idx["arena"]]) for p in sweep_profiles("conscientiousness")]
    check("A3 C->structure", strictly_increasing(xs) and strictly_increasing(ys[::-1]),
          f"P(STRUCT)={['%.2f' % x for x in xs]}, P(arena)={['%.2f' % y for y in ys]}")

    xs = [float(p_base(p)[idx["arena"]]) for p in sweep_profiles("neuroticism")]
    ys = [float(p_base(p)[idx["chapel"]]) for p in sweep_profiles("neuroticism")]
    check("A4 N pref", strictly_increasing(xs[::-1]) and strictly_increasing(ys),
          f"P(arena)={['%.2f' % x for x in xs]}, P(chapel)={['%.2f' % y for y in ys]}")

    hs = [entropy(p_rule(p)) for p in sweep_profiles("neuroticism")]
    check("A5 N temp", strictly_increasing(hs), f"entropy={['%.2f' % h for h in hs]}")

    tav = [p_actions(p, "tavern") for p in sweep_profiles("agreeableness")]
    yard = [p_actions(p, "arena") for p in sweep_profiles("agreeableness")]
    check("A6 A actions",
          strictly_increasing([d["chat"] for d in tav])
          and strictly_increasing([d["brawl"] for d in tav][::-1])
          and strictly_increasing([d["coach"] for d in yard])
          and strictly_increasing([d["spar"] for d in yard][::-1]),
          f"chat={['%.2f' % d['chat'] for d in tav]}, brawl={['%.2f' % d['brawl'] for d in tav]}")

    # ---------------- B. magnitudes -------------------------------------------
    for trait in TRAITS:
        if trait == "agreeableness":
            lo = p_actions(Personality.from_traits(agreeableness=-1.0), "tavern")
            hi = p_actions(Personality.from_traits(agreeableness=1.0), "tavern")
            t = tvd(np.array(list(lo.values())), np.array(list(hi.values())))
        else:
            t = tvd(p_rule(Personality.from_traits(**{trait: -1.0})),
                    p_rule(Personality.from_traits(**{trait: 1.0})))
        check(f"B1 {trait[:12]}", t >= 0.15, f"endpoint TVD={t:.2f} (>=0.15)")

    worst = max(
        (float(p_rule(p).max()) for trait in TRAITS for p in sweep_profiles(trait)),
    )
    check("B2 no caricature", worst <= 0.50, f"worst single-trait max P={worst:.2f} (<=0.50)")

    d0 = p_rule(Personality.from_traits())
    check("B3 neutral", d0.max() <= 0.25 and d0.min() >= 0.03,
          f"neutral max={d0.max():.2f} (<=0.25), min={d0.min():.2f} (>=0.03)")

    h_hi = entropy(p_rule(Personality.from_traits(neuroticism=1.0)))
    h_lo = entropy(p_rule(Personality.from_traits(neuroticism=-1.0)))
    m_lo = float(p_rule(Personality.from_traits(neuroticism=-1.0)).max())
    check("B4 N range", (h_hi - h_lo) >= 0.4 and m_lo <= 0.70,
          f"entropy gap={h_hi - h_lo:.2f} (>=0.4), maxP@N-1={m_lo:.2f} (<=0.70)")

    # ---------------- C. memory behaviour -------------------------------------
    tavern_opt = world.effective_location("tavern")

    def with_tavern_buffer() -> RecentBuffer:
        buf = RecentBuffer(maxlen=3)
        buf.push(tavern_opt)
        return buf

    neutral = Personality.from_traits()
    before = float(p_rule(neutral)[idx["tavern"]])
    after = float(p_rule(neutral, with_tavern_buffer())[idx["tavern"]])
    ratio = after / before
    check("C1 satiation", 0.30 <= ratio <= 0.80,
          f"P(tavern) {before:.2f} -> {after:.2f}, ratio={ratio:.2f} (0.30-0.80)")

    hiC = Personality.from_traits(conscientiousness=1.0)
    r_c = float(p_rule(hiC, with_tavern_buffer())[idx["market"]]) / float(p_rule(hiC)[idx["market"]])
    hiO = Personality.from_traits(openness=1.0)
    r_o = float(p_rule(hiO, with_tavern_buffer())[idx["market"]]) / float(p_rule(hiO)[idx["market"]])
    check("C2 routine C", 1.15 <= r_c <= 2.0, f"P(market) ratio for C+1 = {r_c:.2f} (1.15-2.0)")
    check("C2 novelty O", 0.50 <= r_o <= 0.87, f"P(market) ratio for O+1 = {r_o:.2f} (0.50-0.87)")

    # v1.2 revision: C-modulated satiation deliberately lets routine (high-C)
    # personalities concentrate, so named profiles get looser bounds; the neutral
    # profile keeps the strict bound as the mechanism-degeneracy guard.
    traj_profiles = dict(npcs)
    traj_profiles["Neutral"] = neutral
    worst_share, worst_who, min_distinct = 0.0, "", 99
    neutral_share, neutral_distinct = 0.0, 0
    for name, p in traj_profiles.items():
        ctrl = DecisionController(scorer, mode="sample", rng=np.random.default_rng(42),
                                  selection_temperature=0.1)
        visits: dict[str, int] = {}
        for _ in range(50):
            loc = ctrl.choose_location(p, locs)
            ctrl.choose_action(p, world.actions_at(loc.option.id))
            visits[loc.option.id] = visits.get(loc.option.id, 0) + 1
        share = max(visits.values()) / 50.0
        if name == "Neutral":
            neutral_share, neutral_distinct = share, len(visits)
        else:
            if share > worst_share:
                worst_share, worst_who = share, name
            min_distinct = min(min_distinct, len(visits))
    check("C3 no collapse",
          worst_share <= 0.75 and min_distinct >= 3
          and neutral_share <= 0.60 and neutral_distinct >= 4,
          f"named worst share={worst_share:.2f} ({worst_who}) (<=0.75), named min distinct={min_distinct} (>=3); "
          f"neutral share={neutral_share:.2f} (<=0.60), neutral distinct={neutral_distinct} (>=4)")

    acts0 = p_actions(neutral, "tavern")
    buf_a = RecentBuffer(maxlen=3)
    chat = next(a for a in world.actions_at("tavern") if a.id == "chat")
    buf_a.push(chat)
    acts1 = p_actions(neutral, "tavern", buffer=buf_a)
    ratio_a = acts1["chat"] / acts0["chat"]
    check("C4 action satiation", 0.30 <= ratio_a <= 0.80,
          f"P(chat) {acts0['chat']:.2f} -> {acts1['chat']:.2f}, ratio={ratio_a:.2f} (0.30-0.80)")

    # ---------------- F. combined profiles -------------------------------------
    def top_loc(p: Personality) -> str:
        return ids[int(np.argmax(p_rule(p)))]

    def top_action(p: Personality, loc_id: str) -> str:
        d = p_actions(p, loc_id)
        return max(d, key=d.get)

    gale = npcs["Gale"]
    d = p_rule(gale)
    lib = p_actions(gale, "library")
    bookish = lib["read"] + lib["research"]
    check("F1 Gale", top_loc(gale) == "library" and d[idx["tavern"]] >= 0.08
          and bookish >= lib["discuss"],
          f"top={top_loc(gale)}, P(tavern)={d[idx['tavern']]:.2f} (>=0.08), "
          f"read+research={bookish:.2f} vs discuss={lib['discuss']:.2f}")

    lae = npcs["Laezel"]
    arena_d = p_actions(lae, "arena")
    combat = arena_d["fight"] + arena_d["spar"] + arena_d["drill"]
    check("F2 Laezel", top_loc(lae) == "arena" and min(arena_d, key=arena_d.get) == "coach"
          and combat >= 0.60 and arena_d["fight"] >= arena_d["spectate"],
          f"top={top_loc(lae)}, arena={ {k: round(v, 2) for k, v in arena_d.items()} }, "
          f"combat mass={combat:.2f} (>=0.60)")

    sh = npcs["Shadowheart"]
    check("F3 Shadowheart", top_loc(sh) == "chapel" and top_action(sh, "chapel") in {"pray", "meditate"},
          f"top={top_loc(sh)}, chapel action={top_action(sh, 'chapel')}")

    ast = npcs["Astarion"]
    d = p_rule(ast)
    tav_a = p_actions(ast, "tavern")
    check("F4 Astarion",
          top_loc(ast) in {"tavern", "market"} and d[idx["arena"]] >= d[idx["chapel"]]
          and (tav_a["chat"] + tav_a["drink"]) >= tav_a["brawl"] and tav_a["brawl"] >= 0.15,
          f"top={top_loc(ast)}, P(arena)={d[idx['arena']]:.2f} vs P(chapel)={d[idx['chapel']]:.2f}, "
          f"tavern={ {k: round(v, 2) for k, v in tav_a.items()} }")

    kar = npcs["Karlach"]
    yard_k = p_actions(kar, "arena")
    tav_k = p_actions(kar, "tavern")
    check("F5 Karlach",
          top_loc(kar) in {"tavern", "arena"}
          and max(yard_k, key=yard_k.get) in {"spar", "coach"}
          and min(tav_k, key=tav_k.get) == "brawl",
          f"top={top_loc(kar)}, arena action={max(yard_k, key=yard_k.get)}, "
          f"tavern min={min(tav_k, key=tav_k.get)}")

    hal = npcs["Halsin"]
    check("F6 Halsin", top_loc(hal) == "forest" and top_action(hal, "forest") in {"explore", "forage"},
          f"top={top_loc(hal)}, forest action={top_action(hal, 'forest')}")

    # ---------------- report ---------------------------------------------------
    fails = 0
    print(f"{'check':<22} result  measured")
    print("-" * 78)
    for cid, ok, detail in results:
        fails += 0 if ok else 1
        print(f"{cid:<22} {'PASS' if ok else 'FAIL':<6}  {detail}")
    print("-" * 78)
    print(f"{len(results) - fails}/{len(results)} passed")
    return fails


if __name__ == "__main__":
    sys.exit(main())
