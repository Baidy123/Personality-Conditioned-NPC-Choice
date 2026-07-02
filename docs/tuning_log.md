# Phase-2 Tuning Log

Every change to tables, coefficients, world data, or acceptance criteria is
recorded here with its reason and its effect on the scorecard
(`python -m examples.acceptance_check`, checklist:
`docs/specs/2026-07-02-phase2-acceptance-checklist.md`).

## Round 1 — 2026-07-02 · baseline 21/25 → 25/25

Baseline failures: B4 (N entropy gap 0.31 < 0.4), C3 (Shadowheart visited only
3 locations in 50 rounds), F1 (Gale: top = market; library action = help_reader),
F6 (Halsin: top = market).

### Scalar changes (config.py, grid-searched)

| knob | 1.0 → | reason |
|---|---|---|
| `lambda_N` | 1.5 | B4: entropy at N=+1 is near the ln(7) ceiling, so the gap can only widen by sharpening the stable end. |
| `lambda_R` | 1.2 | C3: stronger satiation pushes narrow profiles to more distinct locations (min distinct 3 → ≥4). |

### Table changes (weights.py, user-approved)

| cell | from → to | reason |
|---|---|---|
| `b_social` | 0.50 → 0.475 | "Everyone wants mid social" over-penalised low-social places (library, forest) for mildly social profiles. |
| `w_social` | 1.0 → 0.8 | Social misfit cost was ~2× every other feature and dominated location choice. |
| `C[O, cognitive]` | 0.30 → 0.45 | O expressed almost solely via exploration; scholar-type openness (Gale) had no channel, losing library to market. |
| `w_cognitive` | 0.6 → 0.7 | Companion to the previous row; cognitive fit slightly more consequential. |

### Data changes (user-approved)

- `personalities.json`: Halsin `extraversion` −0.2 → −0.5. His old profile
  (A+.7, N−.6, E−.2) computed to a market-lover; the canonical solitary druid is
  more introverted. Fixes F6 (market → forest, gap was 2× and untunable by tables).
- `world.json`: library action `help_reader` → `discuss`
  (social .6, cognitive .6, cooperation .5; no strong `helping`). User judged
  help_reader a poor action option; its helping=0.8 also let the linear
  relational bonus dominate every mildly agreeable profile's library choice.

### Criterion revision

- F1 library-action sub-check: argmax ∈ {read, research} →
  `P(read)+P(research) ≥ P(discuss)`. Gale's action distribution is a near-tie
  (0.30/0.34/0.36); an argmax requirement over-reads a 0.02 gap. The mass
  comparison captures the intended "mostly there for the books" (0.64 vs 0.36).

### Equation-change scout (rejected)

Tested replacing the squared intensity cost `w·(o−μ)²` with `w·|o−μ|` to
rebalance intensity vs relational scale (in-memory only). Result: 20/25 —
uniformly sharper distributions broke B2/B4/C3/F2, and Gale's library
distribution barely moved (abs amplifies the social penalty and the cognitive
advantage equally). Conclusion: keep the squared form; the perceived "scale
imbalance" was not the root cause of F1/F6.

### End state

Scorecard 25/25; pytest 16/16. Known risks to re-check in later rounds: C2
novelty ratio for O+1 sits near its 0.87 upper bound; B1 openness TVD sits at
the 0.15 lower bound before the C[O,cognitive] increase (now comfortable).

## Round 2 — 2026-07-02 · conflict semantics + Halsin/Karlach fixes (25/25 kept)

Trigger: user review of character behaviour (Karlach too tame at the arena;
Halsin's E questioned).

### Data changes (world.json, user-approved)

| value | from → to | reason |
|---|---|---|
| `spar.conflict` | 0.5 → 0.2 | `conflict` semantics tightened to "hostility toward an unwilling target"; sparring is consensual training. |
| `arena fight.conflict` | 0.9 → 0.45 | Sanctioned sport with willing opponents; distinct from `brawl` (0.9, unprovoked). Lets agreeable-but-chaotic profiles enjoy legitimate violence. |
| `arena fight.social` | 0.3 → 0.5 | Fighting before a crowd is performative; 0.3 over-penalised it for social profiles. |

### Profile changes (personalities.json, user-approved)

- Halsin `openness` 0.6 → 0.8 (E stays −0.5). Raising E was tested (−0.35, −0.2,
  with and without O boosts): the market overtakes the forest in every variant,
  so his nature-seeking must be carried by O (exploration channel) with E kept
  low. Reading: E here encodes preferred crowd level of places, not warmth
  (his warmth is A+0.7). Margin now forest 0.29 vs market 0.25.

### Effects

- Karlach @ arena: fight 0.15 → 0.27 (spectate 0.42, bet 0.31 — near-even mix).
- Astarion tops `fight` at the arena (0.36; low A makes conflict a bonus);
  Laezel mostly spectates (0.53; high-C risk aversion) — both judged in-character.
- Correction note: an earlier in-conversation estimate ("fight ≈ 0.30 after the
  conflict fix") was not an actual measurement; measured value was 0.20 before
  the social bump, 0.27 after.

### End state

Scorecard 25/25; pytest 16/16.

## Round 3 — 2026-07-02 · Laezel blood-seeker fix (25/25 kept)

Trigger: user correction — "养鸡妹" (githyanki girl) = **Laezel**, not Karlach
(nickname misread in round 1); she craves combat and must fight at the arena,
where she was mostly spectating (0.53 vs fight 0.24).

Diagnosis: high C's stimulation (−0.2) and risk (−0.3) aversion suppress all
combat actions; the only "loves legitimate fighting" channel is low-A × conflict.

### Changes (user-approved)

| change | from → to | reason |
|---|---|---|
| `arena fight.structure` (world) | 0.3 → 0.5 | A sanctioned duel is a rule-bound activity. |
| `W_A[A, conflict]` (weights, also feeds `W_rel`) | −0.7 → −0.9 | Strengthen the low-A combat-drive channel. |
| Laezel `conscientiousness` | 0.8 → 0.7 | Slightly less stim/risk damping; still clearly disciplined. |
| Laezel `agreeableness` | −0.7 → −0.9 | Canon-level harshness; feeds the conflict channel. |
| F2 criterion | + "at the arena, top action = fight" | Locks the intended behaviour. |

Grid evidence: milder combos left spectate on top; stronger generic levers
(A→risk −0.2, C-row softening) broke B1/C3/F5.

### Effects and known limitation

- Laezel @ arena: fight 0.41 > spectate 0.39 (was 0.24/0.53); yard spar 0.44
  (blood-seeker), coach 0.10 (still lowest).
- **Known limitation:** her arena *location* probability stays ~0.07 (rank 6).
  The v1 location schema has no combat/opposition feature, so a high-C warrior's
  combat-venue seeking is inexpressible at the location level; all tested
  location-level levers broke other checks. Candidate future fix: a location
  schema revision (phase-1 material), recorded here rather than forced.
- Observation (corrected): Astarion's top tavern action has been `brawl` since
  round 1 (0.404 under conflict −0.7; the round-3 −0.9 change only nudged it to
  0.420 — the earlier "it changed in round 3" attribution was wrong). Cause:
  low-A conflict bonus (+0.44) plus his high stimulation appetite; the schema
  cannot distinguish covert hostility (his style) from open aggression.
- **N-lever test (failed, kept N=0.4):** raising his neuroticism 0.4→0.6 only
  moved brawl 0.42→0.39. Reason: N's risk-avoidance shift is diluted by N's own
  temperature (T = exp(1.5N) flattens all differences as N rises) — the two N
  channels partially cancel for per-option avoidance effects. Note this when
  analysing the N channel in RQ1. Decision: accept brawl-top (chat+drink 0.58
  remains the majority; the F4 mass criterion guards the boundary); the
  covert/open-hostility distinction joins the schema-revision candidate list
  alongside the location-level combat feature.

### End state

Scorecard 25/25; pytest 16/16.

## Round 4 — 2026-07-02 · equation v1.2 + location `conflict` feature (25/25)

Trigger: trajectory review — the universal satiation penalty exiled Laezel from
her routine (training_yard → library/chapel visits, "转职成法师"), and the base
ranking had no channel for combat-venue seeking. User-approved package:

### Design changes (equation v1.2 + schema)

| change | detail | reason |
|---|---|---|
| C-modulated satiation | `rho = lambda_R·(1 − kappa_C·C)`, `kappa_C = 0.5` [PROVISIONAL] | Satiation rate is a personality trait: high C tolerates routine, low C bores fast. Universal penalty was the v1.1 assumption; pilot falsified it. Alternatives considered and rejected: probability-threshold discounts / hard cutoffs in the scorer (candidate-set dependent, discontinuous, and personality-external — bad for RQ1 attribution). |
| Location schema 8 → 9 | + `conflict` ("hosts combat / open opposition"); `mu_conflict = 0.10 − 0.35·A − 0.20·N` | Combat-venue seeking was inexpressible at the location level (round-3 known limitation). Reuses the action-side `conflict` slot in the 12-dim model vector — learned-model interface unchanged. Low A (harsh) and low N (fearless) raise the ideal; separates Laezel (both low → seeks arenas) from Astarion (low A but anxious → avoids them). |
| `min_p` selection fuse | `DecisionController(min_p=...)`, default 0 | Game-layer truncation of the low tail before sampling; the research object `P_rule` stays continuous. |
| celebration buff | `active: true → false` | A permanently-active festival kept the tavern at social/stim 1.0, distorting all social ideals. |

### Data/table/profile adjustments (user-approved)

- world.json: per-location `conflict` values (arena .8, enemy_camp .9,
  training_yard .4, tavern .15, market/forest .05, library/chapel 0);
  tavern `cognitive` 0.1 → 0.2 (tall tales, games — restored Karlach's
  tavern-vs-market tie-break after the buff removal).
- weights: `C[O, exploration]` 0.50 → 0.55 (B1 openness TVD had hit its 0.15
  floor after the world changes); bilinear fallback `W_L` gained a draft
  conflict column.
- profiles: Karlach `openness` 0.3 → 0.2 (offsets the O-boost pulling her to
  the market; she is not the curious type).

### Criterion revision

- C3 split: neutral profile keeps share ≤ 0.60 / distinct ≥ 4 (degeneracy
  guard); named profiles loosened to ≤ 0.75 / ≥ 3 — personality-driven
  concentration is now intended behaviour (Laezel 66% training_yard).

### Effects

- Laezel 50-round trajectory: training_yard 66%, market 26%, arena 8%,
  **zero library/chapel visits** (was the complaint). Static ranking:
  yard .37, market .15, library/chapel .12, tavern .09, arena .08 — tavern rose
  from .04 and the memory system can no longer reach library/chapel (gap >
  max satiation swing), though statically they remain slightly above tavern.
- Astarion: P(arena) 0.13 rank 5 (combat venues now repel the anxious).
- All six trajectories in character; scorecard 25/25; pytest 18/18.
- Note: `docs/v11_tables.png` / `equation_v1.1_tables_example.md` predate the
  conflict column; `weights.py` is the source of truth for current values.

### End state

Scorecard 25/25; pytest 18/18.
