# RQ1 automated experiments — design record

Status: design approved in conversation 2026-07-03; implemented in
`experiments/rq1/`. The perceptual side of RQ1 (human personality recovery)
is out of scope here and comes from the Study-3 human evaluation.

RQ1: *to what extent does the representation and decision structure produce
consistent and distinguishable personality-conditioned patterns of location
and action choice?* Each keyword maps to one analysis:

| keyword | analysis | script |
|---|---|---|
| personality-conditioned | E1 single-trait sensitivity | `run_e1.py` |
| distinguishable | E2 profile distinguishability | `run_e2.py` |
| consistent patterns | E3 trajectory-level patterns | `run_e3.py` |
| decision structure (bounded context) | E4 memory ablation | `run_e4.py` |

## Matched cases (`gen_cases.py` → `data/rq1_cases/`)

A matched case fixes the world variant, the memory state, and the decision
level; only personality varies. Profiles: 5 traits × 9-point sweeps (45),
300 uniform random OCEAN vectors (seed 20260703), the 6 BG3-inspired dev
profiles (illustration only, never statistics), and the neutral profile.
Contexts: 9 location contexts (6 memory states on the full world + 3
candidate-set variants: enemy_camp unlocked, tavern celebration active,
market locked) and 24 action contexts (4 action-memory states × 6 locations).

## Conventions

- Distributions: `P_rule` is the research object; E1 additionally reports
  `P_base` so the N temperature channel can be separated from the preference
  channels.
- Distances: TVD for sweep endpoints (matches the acceptance checklist B1),
  JSD (nats) for profile pairs, Euclidean for personality space.
- Statistics: Spearman rho on distance pairs; Mantel permutation test
  (999 perms) because pairwise distances are not independent.
- Trajectories: 50 rounds × 10 seeds, sample mode,
  `selection_temperature = 0.1` — identical to acceptance check C3.
- `action_repeat_rate` is defined only over consecutive same-location cycles
  (the only case where `H_A` persists); it is what separates the
  `location_only` and `full` ablation conditions.
- Figures use the Okabe-Ito colourblind-safe palette with fixed
  entity-to-colour assignment (`common.py`); all outputs land in
  `results/rq1/` as PNG + CSV.

## Headline numbers (first run, 2026-07-03, scorer v1.2 provisional values)

- E1 endpoint TVD (location, P_rule): E 0.51 > N 0.50 > C 0.28 > O 0.16 >
  A 0.10. A expresses at the action level instead (0.23) — by design
  (checklist §E2: keeps A and E distinguishable); O is the weakest live
  channel; N's location TVD is dominated by its temperature channel
  (P_base 0.36 vs P_rule 0.50).
- E2: Spearman rho = 0.45, Mantel p = 0.001 (300 profiles, 44 850 pairs,
  33 contexts) — personality distance and behavioural distance correlate;
  distinguishability holds beyond the single-trait sweeps.
- E3: trait extremes concentrate trajectories (any |trait| = 1 lowers visit
  entropy); C+1 is the routine channel (max share 0.56, the only profile
  with a visible location-repeat rate); A barely moves trajectory shape
  (its home level is actions).
- E4: without memory, sharpened sampling collapses every profile onto its
  favourite location (repeat rate 0.27-0.95); the memory term restores
  variety and does so through the designed channels — C+1 keeps repeat 0.19
  (routine tolerance, kappa_C) while every other profile drops to ~0.
