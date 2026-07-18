# RQ3 stimulus-generation and Unity interface design

Date: 2026-07-17. Status: approved in conversation; implementation pending.

## Purpose

Provide the pipeline that turns trained policies into human-study stimuli:
a unified policy interface so one decision controller can drive all four
policy variants, a replayable behaviour-sequence file format that a Unity
evaluation world consumes, and a batch generation script with a manifest.
Python generates behaviour; Unity only presents it.

## Context decisions (from design discussion, 2026-07-17)

- **Stimulus form.** Human-study stimuli are recorded video clips of a 2D
  top-down Unity scene with a fixed camera. Actions are conveyed by simple
  character animation plus an on-screen text label. An interactive build
  additionally exists for thesis/viva demonstration; whether participants
  also get an interactive block (operating the NPC world themselves before
  judging personality) is an **open Study-3 decision** — if adopted, note
  that remote participants have no local Python, so live inference would
  need a hosted service or an in-engine port (C# scorer + ONNX/Sentis
  models), and self-directed interaction adds per-participant stimulus
  variance that the recorded-clip measure deliberately avoids. Recorded
  clips remain the primary measure in any case.
- **Playback timing.** Sequence files carry no time information. The Unity
  player advances one step per operator input ("continue" button) or on a
  fixed auto-advance interval used while recording, so matched clips share
  an identical rhythm. The NPC loops its current action animation between
  steps.
- **Static worlds for stimuli.** Every recorded sequence uses a fixed world
  state (no events firing mid-sequence). Behavioural differences within a
  matched set are therefore attributable to personality and policy alone.
  Event buttons exist only in the live demo mode below.
- **Pre-generation, not live inference, for stimuli.** Sequences are
  generated offline and archived. Rationale: text-level quality control
  before any recording effort; batch generation over all conditions; the
  files are the citable stimulus archive (world + seed + file reproduce a
  clip exactly); recording sessions then need only Unity running.
- **No cherry-picking.** Generation conditions (world, personality, cycle
  count, seed) are fixed identically across the four policy variants; each
  policy's output is used as generated. Any exclusion (e.g. a degenerate
  sequence) follows a rule written down in advance, applied uniformly to
  all variants, with counts reported.

## 1. Scope and code layout

Implemented now (Python):

```
code/npc_policy/policies.py             new — policy adapter layer
code/npc_policy/controller.py           small change — accept any policy
code/experiments/rq3/gen_sequences.py   new — batch sequence generation
code/data/rq3_sequences/                output: sequence files + manifest.csv
```

Specified now, implemented later (Unity): the playback player and, last of
all, the live demo mode. Scorer, learned-model, world, and rq1/rq2
experiment code are unchanged.

## 2. Unified policy interface

A *policy* is any object with

```python
distribution(personality, options, buffer, level) -> np.ndarray  # probs over options
```

- `HandAuthoredScorer` gains a small `distribution()` that calls the
  existing `trace()` and returns `P_rule`.
- `LearnedPolicyAdapter(checkpoint_path)` serves all three learned
  variants (simple / nonlinear / agnostic). The checkpoint payload already
  stores the architecture (`{"model": ..., "state_dict": ...}`, written by
  `experiments/rq2/train.py`); the adapter loads it, reuses the rq2
  featurisation code to build model inputs from
  `(personality, options, buffer, level)`, runs a forward pass, and
  returns the probabilities. Checkpoints live under
  `code/results/rq2/models/` (2A) and `code/results/rq2b/models/` (2B);
  which file to use per variant is a generation-config entry.
- `DecisionController` takes a `policy` argument and calls
  `policy.distribution(...)` where it now calls `scorer.trace(...)`. When
  the policy is the hand-authored scorer the full `ScoreTrace` is still
  recorded on `Decision` (rq1/rq2 callers keep working unchanged); for
  learned policies only the final distribution is recorded and
  `Decision.trace` is absent.

Buffer semantics (H_L, H_A, the action-buffer reset on location change)
stay inside the controller and are identical for every policy.

## 3. Sequence file format (the only Unity input for playback)

One JSON file per sequence:

```json
{
  "meta": {
    "sequence_id": "S03",
    "policy": "nonlinear_2b",
    "checkpoint": "IND__nonlinear__wd0.001__s2.pt",
    "personality": {"name": "high_E", "ocean": {"O": 0, "C": 0, "E": 1, "A": 0, "N": 0}},
    "world": "base_world.json",
    "seed": 42,
    "n_cycles": 10,
    "generated_at": "2026-07-20T14:00:00"
  },
  "steps": [
    {"cycle": 1, "location": "tavern", "action": "chat",
     "moved": true,
     "location_probs": {"tavern": 0.61, "...": 0.0},
     "action_probs": {"chat": 0.48, "...": 0.0}},
    {"cycle": 2, "location": "tavern", "action": "drink", "moved": false, "...": {}}
  ]
}
```

- `moved` tells the player whether to play a walk-to segment before the
  action (false when the NPC stays at the same location).
- `location_probs` / `action_probs` are research archive fields (full
  per-step distributions); the Unity player ignores them.
- `checkpoint` is empty for the hand-authored scorer.
- Files are permanent research artefacts — never auto-deleted. Total
  volume is trivial (KBs per file).
- Participant-facing videos use neutral ids (V01.mp4, ...); the mapping
  from video id to (sequence_id, policy, personality) is stored separately
  from anything participants see, preserving blinding.

## 4. Batch generation script

```
python -m experiments.rq3.gen_sequences --config <config.json> [--preview]
```

The config lists: personality profiles, policy variants (with checkpoint
paths for learned ones), scenario world files, cycles per sequence, and
seeds. The script crosses these, runs the controller in `sample` mode with
the given seed per sequence, and writes one JSON file per combination plus
`manifest.csv` (one row per sequence: id, condition fields, file name,
generation time). `--preview` prints each sequence as text for pre-recording
quality control. Re-running with the same config and seeds reproduces
byte-identical step lists.

Usability additions (2026-07-18): the default config is
`configs/rq3_sequences.json` (`--config` optional); personality entries may
be name-only references into `personalities_file`; an optional `unity_dir`
mirrors the written `S*.json` into the Unity playback folder with stale-file
cleanup, removing the manual copy step.

## 5. Unity-side contract (specified only; implemented later)

- **Playback player.** Loads a sequence JSON; renders a fixed-camera 2D
  top-down scene with the world's locations at fixed positions; per step:
  if `moved`, walk the NPC to `location`, then play the `action` animation
  in a loop with a text label. Advances on the continue button or on a
  fixed-interval auto-advance toggle (used for recording).
  Scene decisions (2026-07-18, implemented in `unity/dissertation`, plan
  `docs/plans/2026-07-18-unity-playback-player.md`): on-screen info is
  location name plates + NPC action label only (no step counter; condition
  fields are never parsed by the player); NPC moves in a straight line;
  the auto-advance toggle was removed (2026-07-18, user decision) — stepping
  is manual (Continue/Space); if recordings need uniform pacing, a scripted
  pacer can be added at recording time. Recording mode = control bar hidden
  (H key); the recording tool (Unity Recorder vs OBS) is deferred. The
  player has three NPC slots (A/B/C, distinct colours, per-slot dropdown;
  an NPC-count selector shows 1–3 slot rows, default 1; a playback/live
  mode switch is deferred until the live mode exists)
  stepped in lockstep by one Continue — demo material only: sequences are
  independently generated, NPCs do not perceive each other, and single-NPC
  clips remain the planned study stimulus. Perspective may be
  straight top-down or Stardew-style 3/4 view — a tileset choice; the
  sequence format is perspective-agnostic.
- **Live demo mode** (design finalised 2026-07-18; brainstorm decisions:
  demo events live in `data/world.json` as inactive entries; NPCs display
  their configured personality name; forced behaviour enters the buffers
  and is marked as an override). A local Python HTTP service owns the
  mutable world state and one `DecisionController` per NPC; Unity is a
  pure frontend. NPC count is server-side configuration — Unity renders
  however many the service reports.

  **Server config** (`configs/live_demo.json`; run configs moved to
  `configs/` with a change-X-edit-Y README, 2026-07-18):

  ```json
  { "world": "data/world.json",
    "personalities": "data/personalities.json",
    "port": 8973,
    "mode": "sample", "selection_temperature": 1.0, "seed": 42,
    "npcs": [
      {"name": "Karlach",     "policy": "scorer"},
      {"name": "Shadowheart", "policy": "scorer"},
      {"name": "Laezel",      "policy": "scorer"} ] }
  ```

  `name` is both the display name and the lookup key into
  `personalities.json`; an inline `"ocean": {...}` overrides the lookup.
  Learned policies use `{"policy": "<label>", "checkpoint": "<path>"}`.
  Each NPC gets its own controller and a deterministic rng seeded
  `(seed, slot)`, so a session replays identically after `/reset`.

  **Endpoints** (all JSON; arrays rather than maps so Unity's JsonUtility
  can parse them; the key is `overridden` because `override` is a C#
  keyword):

  - `GET /state` →
    `{world, step_count, locations: [{id, unlocked, events: [{name,
    active, force_npc}]}], npcs: [{slot, name, policy, ocean: [O,C,E,A,N],
    location}]}`. Unity builds the event-controller panel from this —
    adding an event to `world.json` adds a button with no code change.
  - `POST /step` (empty body) → one decision cycle for every NPC:
    `{step_count, steps: [{slot, name, location, action, moved,
    overridden, location_probs: [{id, p}], action_probs: [{id, p}]}]}`.
    If any *active* event carries `force_npc: "location/action"`, every
    NPC's step is that pair with `overridden: true`, empty prob arrays
    (no policy call is made — the counterfactual distribution is not
    computed), and buffers updated through the controller's forced-commit
    path. With several active force events, the first in location-then-
    event order wins. A forced target that is locked is a 400.
  - `POST /event` — either `{location, event, active}` (toggle a local
    event) or `{location, unlocked}` (lock/unlock). Returns the updated
    `/state` payload. Consistent with the checkpoint principle, changes
    never interrupt a current behaviour; they alter what the NPCs see at
    the next `/step`.
  - `POST /npc` (added 2026-07-18) — runtime personality editing, the live
    mode's second control surface next to the event panel. Two payload
    forms: `{slot, ocean: {trait: value, ...}, reset_buffers?}` replaces an
    existing NPC's personality (buffers are **kept** by default — "same
    history, different character"; `reset_buffers: true` gives a blank
    slate), and `{name, ocean?, policy?, checkpoint?}` appends a new NPC
    (inline `ocean`, or a `personalities.json` name lookup). Returns the
    updated `/state`. Runtime edits and additions are in-memory only:
    `/reset` restores the startup config roster. Like every world change,
    a personality change takes effect at the next `/step`.
  - `POST /policy` (added 2026-07-18) — `{policy: label}` switches EVERY
    NPC to one policy from the catalog (the config's `"policies"` shelf
    of labelled checkpoints, plus the always-present `scorer`; per-NPC
    ad-hoc checkpoints join the catalog under their label). Buffers and
    last locations are kept — same history, different decision-maker —
    and the switch shows at the next `/step`. `/state` reports
    `policies` (catalog labels) and `active_policy` (one label, or
    `"mixed"` for a heterogeneous roster). Unity renders this as a
    Policy dropdown on the live bar.
  - `POST /reset` — authored world state restored (reload from file),
    controllers cleared, rngs re-seeded, runtime NPC edits/additions
    discarded. Returns the fresh `/state`.

  Errors: 400 with `{"error": msg}` for bad requests, 404 for unknown
  paths. The server start-up log prints the NPC roster with full OCEAN
  vectors (the values deliberately stay off the Unity screen — name
  labels only, so the frame stays clean; the presenter narrates traits).

  **Demo events** (added 2026-07-18, all `active: false`, so `resolve()`
  ignores them and every existing research artefact is unaffected):
  tavern `celebration` (pre-existing), library `guest_lecture`, chapel
  `holy_day`, market `festival`, forest `storm` (raises risk, *lowers*
  exploration), arena `grand_tournament` with `force_npc:
  "arena/spectate"`. `enemy_camp` is exercised through its unlock toggle.

  **Unity side.** A Mode dropdown (Playback / Live) toggles two UI groups
  and controller components in the same scene. Live mode: Connect first
  auto-starts the brain as a hidden child process (`BrainLauncher`: dev
  default `python -m experiments.rq3.live_server` from `code/`; the
  packaged build points the same component at the bundled PyInstaller
  exe), retries `/state` until the server answers, kills the process on
  quit, and mirrors the brain's console into the Unity Console with a
  `[brain]` prefix. A manually started server on the same port also
  works — the launcher's duplicate simply fails to bind and the retry
  loop reaches the existing one. Connect then fetches
  `/state`, spawns one dot per reported NPC from an inactive prototype
  (distinct colours, personality-name label under the dot, action label
  above on the same staggered height planes as playback), and builds the
  right-side event panel (per location: an unlock toggle plus one button
  per authored event; force events are visually marked) and the left-side
  personality panel (NPC selector, five OCEAN sliders in [−1, 1], Apply →
  `/npc` edit; an Add NPC control with a name field appends a new scorer
  NPC via `/npc` add). Continue/Space posts `/step` and animates the
  returned steps — walk if `moved`, then loop the action; overridden
  steps show the action label with a "(forced)" suffix. R posts `/reset`.
  Request failures land in the status line, not exceptions.
- **Global event controller** (the live mode's control panel). Unity
  calls *state* at startup and builds the panel automatically from the
  world data: per location, one on/off button per authored event, plus an
  unlock/lock toggle — adding an event to `world.json` adds a button with
  no code change. Pressing a button sends
  `{location, event_name, active}` (or an unlock change) to *event*;
  Python mutates world state. Consistent with the project's checkpoint
  principle, a change never interrupts the current behaviour: it alters
  the candidate set and effective features the NPC sees at its next
  *step* call, which is where the reaction becomes visible. `force_npc`
  events are exposed on the same panel; a forced behaviour is marked as
  an override and is never counted as an autonomous choice.
- **Distribution (decided 2026-07-17): single desktop executable.** The
  interactive build ships as one released application with the Python
  service packaged inside (e.g. PyInstaller-built subprocess that Unity
  starts and stops); no browser/WebGL route. Caveat to revisit with the
  evaluation protocol: if the interactive block is given to remotely
  recruited participants, they must download and run an executable —
  state this in the ethics application and expect some drop-off.

## 6. Validation and testing

Export-time validation (generation fails loudly rather than writing a bad
file): every `location` / `action` id must exist in the world file used;
each probability vector sums to 1 within tolerance; step count matches
`n_cycles`.

Tests:

- adapter consistency — `LearnedPolicyAdapter.distribution()` equals the
  rq2 `predict_distribution()` output on identical cases, for all three
  architectures;
- controller regression — the hand-authored scorer driven through the new
  `policy` argument reproduces the old scorer-argument behaviour exactly;
- pipeline round-trip — a small config generates files that re-load, pass
  schema validation, and regenerate identically under the same seeds.

## Out of scope

Unity project implementation, recording protocol details, participant
counterbalancing tables, and the live-demo endpoint payloads. The exact
personality profiles, worlds, cycle counts, and clip counts for the human
study are Study-3 design decisions made when the study materials are
finalised.
