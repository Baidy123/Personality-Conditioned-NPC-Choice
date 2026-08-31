# NPC Policy — core representation & hand-authored scorer

Initial scaffold for the MSc dissertation *"Personality-conditioned discrete NPC
choice with bounded recent-choice context"*. This package implements the core of
`project_flow.md` §1–§2: the representation, the bounded recent-choice memory, the
hand-authored scorer (with its full equations), the controller that drives the
nested location → action decision, and the world loader / local-event resolver that
keeps authored content (and a future game engine) out of the policy code.

## Quick start

```bash
cd code
pip install -r requirements.txt
python -m examples.demo
```

## Running the experiments

Every command runs from `code/`; run `python -m pytest -q` first to confirm
the environment (expected: all green).

**RQ1 — behavioural-expression analyses** (outputs → `results/rq1/`):

```powershell
python -m experiments.rq1.gen_cases   # matched cases (profiles x contexts x world variants)
python -m experiments.rq1.run_e1      # E1 trait sensitivity curves
python -m experiments.rq1.run_e2      # E2 profile-distance correlation
python -m experiments.rq1.run_e3      # E3 trajectory patterns (named + sweep profiles)
python -m experiments.rq1.run_e4      # E4 memory/context ablations
```

Run `gen_cases` first; E1–E4 are independent of each other after that.

**RQ2 — policy learning** (outputs → `results/rq2/`, `results/rq2b/`):

2A, controlled supervision from the hand-authored scorer. Run the four steps in
order; add `--smoke` to every one of them first for a fast end-to-end check.

```powershell
python -m experiments.rq2.gen_controlled   # dataset (train / val / G1-G6 splits)
python -m experiments.rq2.train            # 130 runs, resumable
python -m experiments.rq2.run_2a           # metrics, main table, figures
python -m experiments.rq2.run_e_diag       # E1-E4 structural diagnostic
```

`train` takes `--device cuda` and `--only <run-prefix>` for a single run.

2B, independent supervision from the externally labelled dataset in
`data/rq2_independent/`:

```powershell
python -m experiments.rq2.import_independent   # raw batches -> reviewed splits
python -m experiments.rq2.run_label_probe      # label health check, run before training
python -m experiments.rq2.train_2b
python -m experiments.rq2.run_2b
```

**Game modes (playback generation + live demo):** see `configs/README.md` —
the change-X-edit-Y map, both run commands, and model/checkpoint selection.

## What is implemented (verified)

1. Given a personality, a candidate set, and the relevant memory, produce a
   **choice distribution** over candidates (for a location or an action decision).
2. **Nested decision:** the controller runs the full location → action cycle,
   maintaining both memory buffers and enforcing the reset rule automatically.
3. **Per-level parameters:** action coefficients (λ, temperature) can be tuned
   independently of location ones.
4. **Relation features:** exact repetition (`rep`, by id) is separate from semantic
   similarity (`sim`, by features).
5. **Transparency:** `scorer.trace(...)` exposes every intermediate quantity
   (`base / P_base / mu / gamma / rep / sim / nov / T_N / P_rule`) for the RQ1 analysis.
6. **Data formats:** JSON read/write for both decision-case types (carrying `level`).
7. **Selection modes:** `argmax` (reproducible) or `sample` (draw from the distribution).

## Module map

```
code/npc_policy/
  schema.py         # feature schemas + OCEAN axes; per-level indexing & 12-dim padding
  representation.py  # Option, Personality, RecentBuffer
  weights.py        # v1.1 tables b/C/w (+ W_rel for actions) + fallback W_L / W_A
  config.py         # ScorerConfig: tunables; LevelParams for location vs action
  relations.py      # rep / sim / nov from a buffer
  scorer.py         # HandAuthoredScorer: base score + reweighting -> distribution
  controller.py     # DecisionController: owns H_L/H_A, nested choice + reset rule
  world.py          # load world.json + local-event resolution (game-layer boundary)
  cases.py          # ControlledCase / IndependentCase + (de)serialisation
data/world.json        # authored locations, action sets, unlocked flags, local events
data/personalities.json # named NPC OCEAN profiles
data/rq1_cases/        # generated matched cases (profiles + contexts + world variants)
examples/demo.py    # end-to-end demo + self-checks
experiments/rq1/    # RQ1 automated analyses (gen_cases + run_e1..e4 -> results/rq1)
```

Dependency direction: `schema` → `representation` / `weights` → `relations` /
`scorer` → `controller` / `world` / `cases` → `demo`.

## Representation (`project_flow.md` §1)

- **Personality:** OCEAN vector in `[-1, 1]` (`Personality`).
- **Location schema (9, v1.2):** `social, stimulation, structure, cognitive, physical,
  risk, exploration, privacy, conflict` (`conflict` = the place hosts combat /
  open opposition; added in tuning round 4).
- **Action schema (11):** the shared first seven, then `cooperation, helping,
  conflict, control`.
- **Unified 12-dim model vector** (`MODEL_TAGS`): location fills `conflict` natively
  (v1.2) and pads the three remaining action-only fields with zeros, action pads
  `privacy` with zero. Interface padding only — used
  by future learned models, **not** by the scorer. Produced on demand by
  `Option.to_padded12()`; an `Option` stores its **native** vector (8 or 11).
- **Option:** carries `id` (identity, for exact repetition) + native `features`
  (semantics, for similarity) + `level` (`location` / `action`).
- **Memory buffers (`RecentBuffer`, FIFO):** `H_L` recent locations (length `K_L`);
  `H_A` recent actions at the *current* location (length `K_A`), reset on a location
  change. Both store full `Option`s (id + features). Provisional `K_L = K_A = 3`.

## One decision cycle (controller view)

```
(1) choose_location(personality, available_locations)
    read H_L -> relations.py computes rep/sim/nov for each location
    -> scorer: ideal-point base (b/C/w) -> reweight q (gamma*fam - lambda_R*rep) -> P_rule  (level="location")
    -> pick a location L_t from P_rule
    -> if L_t != previous location: clear H_A      # the reset rule, automatic
    -> push L_t into H_L

(2) choose_action(personality, actions_at(L_t))
    read H_A (empty right after a location change) -> rep/sim/nov for each action
    -> scorer: same structure with the action tables (+ W_rel)      (level="action")
    -> pick an action -> push it into H_A

-> wait for the next checkpoint, back to (1)
```

Location and action share the **same equation structure**; they differ only in
which tables are used, which memory is read, and which per-level coefficients apply.

## World content & local events (`world.py`)

Content lives in `data/*.json`, not in code. `load_world` / `load_personalities`
read it into `Option` / `Personality` objects, so changing the world means editing
JSON only. A live game engine is just another source feeding the same policy inputs
`(personality, candidates, memory, level)` — the same door.

Each location (see `location_schema_figure.html`, Listing 1) carries base `features`,
its action set, a game-layer `unlocked` flag, and a set of **local events**. A local
event has `active`, a `buff` (temporary feature deltas added while active, location
features only), and `force_npc` (a scripted override). `unlocked`, the events, and
`force_npc` stay in the game layer — the model never sees them.

`World.resolve()` produces the current location candidate set:

1. drop locations whose `unlocked` is false;
2. for each remaining location, add every **active** event's `buff` to the base
   features, clamped to `[0, 1]`, giving the **effective** features;
3. the scorer receives only these effective options.

Global events (the upper-layer switches that toggle `unlocked` / `active`, e.g.
"war won" unlocking a camp or activating a tavern celebration) are handled at the
game-engine layer; in the JSON these flags are set directly.

## Scorer equations (`scorer.py`, from `project_flow.md` §2)

For decision level `d ∈ {L, A}`, candidate `o_i` in its native schema, relation
features from the relevant buffer (`fam` = recency-weighted similarity, stored as
`sim`; `nov` is a learned-model input feature only):

```
# base_form = ideal_point (default)
mu_f(p)  = clip(b_f + Σ_t C[t,f]·p_t, 0, 1)        # intensity features
base_i   = − Σ_f w_f·(o_i[f] − mu_f(p))²
           + Σ_t Σ_f p_t·W_rel[t,f]·o_i[f]         # action relational features only

# base_form = bilinear (debugging fallback)
base_i   = p^T W^d o_i^d

# memory + temperature (shared)
gamma    = lambda_C·C − lambda_O·O + lambda_Nf·N   # v1.3: N clings to the familiar
rho      = lambda_R·(1 − kappa_C·C)     # v1.2: routine-tolerant satiation
q_i      = base_i / tau_0 + gamma·fam_i − rho·rep_i
T_N      = exp(lambda_N·N)
P_rule   = softmax(q / T_N)
```

When the relevant buffer is empty, `rep = fam = 0`, so `P_rule = softmax(base/tau_0 / T_N)`
(the first action after a location change uses the unadjusted base distribution;
only the temperature applies).

## Status of values

Everything marked `PROVISIONAL` in `config.py` or `weights.py` is a starting value,
not a decided one (`project_flow.md` §9):

- the coefficients `lambda_R / kappa_C / lambda_O / lambda_C / lambda_N / lambda_Nf`, base temperature
  `tau_0`, `recency_decay`, buffer lengths, and the `base_form` switch live in
  `config.py` (location and action each get their own `LevelParams`, defaulting to
  equal values);
- the v1.1 tables `b / C / w` (per level) and `W_rel` live in `weights.py`, alongside
  the fallback `W_L` / `W_A`. All are hand-authored provisional directions to be
  tuned there; `python -m examples.make_tables_figures` renders the current values
  to `docs/v12_tables.png` / `docs/v12_ideal_levels_demo.png`.

These values must be examined empirically (RQ1), not assumed correct.

## Status & next steps (maps to `project_flow.md` §10)

Done:

- [x] Representation: location (8) / action (11) schemas, 12-dim padding, `Option`,
      `Personality`, FIFO buffers (§1).
- [x] Hand-authored scorer with the v1.1 equations (ideal-point base + bilinear
      fallback, gamma-familiarity memory term, bidirectional N temperature) and
      `rep / sim / nov` relation features.
- [x] `DecisionController`: nested location → action cycle, buffer ownership, the
      action-buffer reset rule (§5).
- [x] World loader + local-event resolution (`unlocked`, active-event buffs); content
      authored in `data/*.json` (§5b).
- [x] Decision-case formats with JSON (de)serialisation (`cases.py`), native vectors.

- [x] Matched-case generator (§10 step 2) and the automated RQ1 analyses
      (§10 step 3): trait sensitivity, profile distinguishability, trajectory
      patterns, memory ablations (`experiments/rq1/`).
- [x] Learned-policy layer (simple / nonlinear / agnostic under the shared
      12-dim interface) and the 2A controlled pipeline: dataset generation,
      130 training runs (S0 + G1–G6 + ablations + data sizes), metrics, and
      the E1–E4 structural diagnostic (`experiments/rq2/`).
- [x] 2B independent pipeline: externally labelled dataset with import/review
      gates, label probe, training, and evaluation (`results/rq2b/`).
- [x] RQ3 tooling: sequence pipeline + Unity evaluation environment
      (playback player and live demo with the local decision service;
      `experiments/rq3/`, `unity/dissertation`, `configs/README.md`).

Next:
- [ ] Human-study protocol (with supervisor), ethics, stimulus recording,
      and the study itself — the sole evidence source for RQ3.
- [ ] Study 3 comparative automated evaluation write-up on the independent
      test cases.
- [ ] Dissertation writing.

Global-event switches (toggling `unlocked` / `active`) are deferred to game-engine
integration; `force_npc` is stored but unused (game layer, excluded from data per §5c).
