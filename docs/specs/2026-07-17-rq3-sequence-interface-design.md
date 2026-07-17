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

## 5. Unity-side contract (specified only; implemented later)

- **Playback player.** Loads a sequence JSON; renders a fixed-camera 2D
  top-down scene with the world's locations at fixed positions; per step:
  if `moved`, walk the NPC to `location`, then play the `action` animation
  in a loop with a text label. Advances on the continue button or on a
  fixed-interval auto-advance toggle (used for recording).
- **Live demo mode** (lowest priority, after all data collection). A local
  Python HTTP service owns `World` + `DecisionController` and exposes
  three endpoints: *step* (compute and return the next decision), *event*
  (toggle a world event / unlock flag), *reset*. Unity renders, and its
  event buttons call *event*. Endpoint payloads are defined at
  implementation time; nothing in items 1–4 depends on this mode.

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
