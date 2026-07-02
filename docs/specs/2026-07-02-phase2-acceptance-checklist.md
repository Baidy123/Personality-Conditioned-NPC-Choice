# Phase-2 Tuning Acceptance Checklist (v1.1 scorer)

Status: FROZEN 2026-07-02 (user-approved; §E decisions resolved, §F added).
Numeric bounds may still be revised during tuning, but only with a logged reason;
every table/λ change is evaluated against this checklist. Purpose: make "the behaviour looks right" checkable, so
tuning can run as an automated loop with documented decisions
(CLAUDE.md: documented pilot refinement).

Scope: the hand-authored scorer only (`base_form = ideal_point`), demo world
(`data/world.json`, sparse test data — see note in §E3). Checks run on
distributions (no sampling noise) except §C3, which simulates trajectories with a
fixed seed. "Sweep" = trait value in {−1, −0.5, 0, +0.5, +1}, all other traits 0,
empty memory unless stated.

Tuning rules of engagement (agreed 2026-07-02):
- scalar knobs (`lambda_R/O/C/N`, `tau_0`, `recency_decay`) may be grid-searched
  automatically against this checklist;
- table values (`b`, `C`, `w`, `W_rel`) change only via proposed edits with
  written reasons, approved by the user (the scorer must remain hand-authored);
- every change is logged: what, why, before/after check results.

## A. Direction checks (pass/fail)

Location checks use the memory-free, temperature-free base distribution
`P_base = softmax(base / tau_0)` so trait-preference direction is not confounded
by the N temperature. `SOCIAL = {tavern, market, arena}`,
`QUIET = {library, chapel, forest}`, `STRUCT = {library, chapel, training_yard}`.

- **A1 (E):** `P_base(SOCIAL)` strictly increases over the E sweep; top choice at
  E=−1 is in QUIET and at E=+1 is in SOCIAL.
- **A2 (O):** `P_base(forest)` strictly increases over the O sweep.
- **A3 (C):** `P_base(STRUCT)` strictly increases over the C sweep;
  `P_base(arena)` strictly decreases.
- **A4 (N, preference):** on `P_base`, `P_base(arena)` decreases and
  `P_base(chapel)` increases over the N sweep (risk avoidance / privacy seeking).
- **A5 (N, temperature):** on `P_rule`, entropy strictly increases over the N
  sweep (bidirectional temperature).
- **A6 (A, actions):** at the tavern action set, `P(chat)` increases and
  `P(brawl)` decreases over the A sweep; analogous direction at `training_yard`
  (`P(coach)` up, `P(spar)` down).

## B. Magnitude checks (tuning targets)

- **B1 (no dead trait):** for each trait at its home level (O/C/E/N: location;
  A: action), the total variation distance between the sweep endpoints'
  distributions is **≥ 0.15**.
- **B2 (no caricature):** for every single-trait profile in the sweep grid,
  `max P_rule ≤ 0.50` (empty memory).
- **B3 (neutral profile is non-degenerate):** for the all-zero personality,
  `max P_rule ≤ 0.25` and `min P_rule ≥ 0.03` over the 7 locations.
- **B4 (N range is visible but not absolute):**
  `entropy(P_rule | N=+1) − entropy(P_rule | N=−1) ≥ 0.4 nat`, and at N=−1
  `max P_rule ≤ 0.70` (stable ≠ deterministic).

## C. Memory-behaviour checks

- **C1 (satiation):** neutral personality; push `tavern` once into `H_L`; then
  `P_rule(tavern)` drops to **30%–80%** of its empty-memory value. (Too small =
  rep penalty invisible; too large = hard ban.)
- **C2 (routine/novelty magnitude):** with `tavern` in `H_L`, `P_rule(market)`
  changes vs empty memory by a factor in **[1.15, 2.0]** for C=+1 (up) and
  **[0.5, 0.87]** for O=+1 (down).
- **C3 (no behavioural collapse):** 50-round controller trajectory (sample mode,
  seed 42, `selection_temperature = 0.1`) for every profile in
  `data/personalities.json` plus the neutral profile: no location accounts for
  **> 60%** of visits; at least **4 distinct** locations are visited.
- **C4 (action-level satiation):** stay at the tavern for consecutive decisions;
  after choosing `chat` once, `P_rule(chat)` on the next same-location decision
  drops to 30%–80% of its previous value (mirror of C1).

## D. Structural regressions (already automated)

The 16 pytest checks in `code/tests/` must stay green after every change
(equivariance, empty-buffer behaviour, bilinear fallback, γ directions,
bidirectional N, ideal-point signature).

## E. Judgment items (resolved 2026-07-02)

- **E1 — N=−1 loves the arena. DECISION: keep `C[N, risk] = −0.35`.** After
  densifying `world.json` the single-trait N=−1 profile tops the market, with the
  arena in the second tier; trait combinations modulate the effect further
  (calm+introvert → forest; calm+conscientious → arena suppressed). Single-trait
  extremes are diagnostic profiles; combined-profile plausibility is checked in §F.
- **E2 — Agreeableness at the location level. DECISION: keep the weak link**
  (`C^L[A, social] = +0.15`); A expresses mainly through the relational action
  features so that A and E remain distinguishable to human observers. A6 covers
  A at its home level.
- **E3 — sparse world.json zeros. DECISION: densified 2026-07-02.** All intensity
  features now carry authored values (0.05/0.1 = near-absent); a literal 0 no
  longer occurs accidentally. Existing values were left untouched.

## F. Combined-profile plausibility checks (BG3-inspired dev profiles)

Profiles live in `data/personalities.json`. **Dev/tuning use only:** the human
study must NOT use these names — participants who recognise the characters would
infer personality from the name rather than from behaviour. Study materials use
anonymised profiles (values may be reused, names replaced).

Each check uses the empty-memory location distribution `P_rule` (and the stated
action set where given). "top" = argmax.

- **F1 Gale** (O+.9 C+.5 E+.1 A+.4 N+.3): top = `library`; `P(tavern) ≥ 0.08`
  (a solid second tier — he does like a good tavern story, just less than books);
  at the library, `P(read) + P(research) ≥ P(discuss)` (bookish mass beats
  socialising; an argmax check was too strict for a near-tie distribution —
  revised 2026-07-02 after the library action set was reworked, see
  docs/tuning_log.md).
- **F2 Laezel** (O−.3 C+.8 E−.2 A−.7 N−.7): top = `training_yard`; at the
  training yard `P(coach)` is the lowest of the three actions.
- **F3 Shadowheart** (O+.1 C+.4 E−.5 A+.1 N+.6): top = `chapel`; at the chapel,
  top action ∈ {pray, meditate}.
- **F4 Astarion** (O+.4 C−.5 E+.6 A−.7 N+.4): top ∈ {tavern, market};
  `P(arena) ≥ P(chapel)`; at the tavern `P(chat) + P(drink) ≥ P(brawl)` but
  `P(brawl)` is not negligible (≥ 0.15) — scheming, not saintly.
- **F5 Karlach** (O+.3 C−.4 E+.9 A+.6 N−.5): top ∈ {tavern, arena,
  training_yard}; at the training yard top action ∈ {spar, coach} (friendly
  fighter: sparring yes, brawling no); at the tavern `P(brawl)` is the lowest.
- **F6 Halsin** (O+.8 C+.1 E−.5 A+.7 N−.6; revised round 1–2, see
  docs/tuning_log.md): top = `forest`; at the forest, top action ∈ {explore,
  forage}. Note on E−.5: in this schema E encodes the preferred crowd/stimulation
  level of places, not interpersonal warmth — Halsin's warmth lives in A+.7.

## Runner

`examples/acceptance_check.py` evaluates A1–A6, B1–B4, C1–C4, and F1–F6 and
prints a pass/fail scorecard with measured values, so each tuning round produces
a comparable report. Section D is covered by `python -m pytest tests`.
