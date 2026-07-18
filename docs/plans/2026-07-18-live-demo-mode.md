# Live demo mode — implementation plan (2026-07-18)

Implements the live-mode contract in
`docs/specs/2026-07-17-rq3-sequence-interface-design.md` (section 5, design
finalised 2026-07-18). Python side first (TDD, runnable without Unity), then
the Unity client.

Brainstorm decisions (2026-07-18, user):

1. Demo events go into `data/world.json`, all `active: false` — inert for
   every research pipeline.
2. Live NPCs show their personality name on screen; OCEAN numbers stay off
   the frame (server start-up log + presenter narration instead).
3. `force_npc` behaviour **enters the buffers** and is marked `overridden`.

## Task 1 — demo events in `data/world.json`

Add inactive events: library `guest_lecture` {social +0.25, stimulation
+0.15}; chapel `holy_day` {social +0.2, structure +0.1}; market `festival`
{social +0.15, stimulation +0.2, exploration +0.1}; forest `storm` {risk
+0.3, exploration −0.2, stimulation +0.1}; arena `grand_tournament`
{social +0.2, stimulation +0.2} with `force_npc: "arena/spectate"`.
Verification: `load_world` round-trip + full existing pytest suite green
(no research output may change — events are inactive).

## Task 2 — `DecisionController.commit_forced` (TDD)

```python
def commit_forced(self, location: Option, action: Option) -> None:
    """Record an externally scripted, non-autonomous choice pair.

    Applies the same structural buffer rules as autonomous choice; the
    caller is responsible for marking the step as an override."""
```

Same rules as `choose_location` + `choose_action`: clear `H_A` if the
location changed, push `H_L`, update last-location, push the action to
`H_A`. Tests (in `tests/test_live_session.py` alongside the session tests):
buffers after `commit_forced` equal buffers after the equivalent autonomous
pair; a following autonomous action choice sees the forced history.

## Task 3 — `experiments/rq3/live_session.py` (TDD)

`LiveSession` — all logic, no HTTP:

- `LiveSession.from_config(path)` — parses the config (spec format),
  resolves personalities (name lookup in `personalities.json`, inline
  `ocean` override, unknown name → `ValueError`), builds one
  `DecisionController` per NPC (`scorer` → `HandAuthoredScorer`, else
  `LearnedPolicyAdapter(checkpoint)`), rng per NPC =
  `np.random.default_rng([seed, slot])`.
- World mutability without touching core classes: `entries[loc] =
  dataclasses.replace(entry, ...)` (the entries dict is mutable; the
  frozen dataclasses are swapped, never mutated). `set_event(loc, name,
  active)`, `set_unlocked(loc, unlocked)`; unknown ids → `ValueError`.
- `step()` → list of per-NPC step dicts exactly as the spec's `/step`
  payload. Force detection: first active event with `force_npc` in
  location-then-event order; parse `"loc/act"`, locked target or unknown
  ids → `ValueError`; forced steps skip the policy call, use
  `commit_forced`, return empty prob arrays, `overridden: true`.
  Autonomous steps: `choose_location` / `choose_action` over
  `world.resolve()`, `moved` vs the controller's previous location id.
- `state()` → the `/state` payload dict.
- `reset()` — world reloaded from file, controllers rebuilt, rngs
  re-seeded; a stepped-then-reset session reproduces a fresh session's
  steps exactly.

Test list (`tests/test_live_session.py`), written before the code:

1. config parsing: roster size, personality lookup, inline ocean,
   unknown personality name raises, missing checkpoint raises;
2. `state()` shape: locations with events/unlocked, npcs with slot/name/
   ocean array/location "" before first step, `step_count`;
3. `step()`: per-NPC entries, `moved` flags across two steps, prob arrays
   sum ≈ 1 and cover the candidate set, `step_count` increments;
4. determinism: same config file → identical step sequences; reset →
   identical replay;
5. events: activating `celebration` changes tavern's effective features
   in the candidates (verify via world), unlocking `enemy_camp` makes it
   appear in `location_probs`; deactivating restores;
6. force: activate `grand_tournament` → all NPCs step to arena/spectate,
   `overridden` true, empty probs, buffers contain the forced pair
   (equivalence with an autonomous arena/spectate pair); deactivate →
   autonomous behaviour resumes from that history; locked force target
   raises;
7. `commit_forced` unit equivalence (Task 2's tests live here).

## Task 4 — `experiments/rq3/live_server.py` (TDD)

Stdlib `http.server.HTTPServer` (single-threaded — requests from Unity are
sequential; no new dependency), one `LiveSession`:

- routes per spec; `POST /event` accepts either payload form and returns
  the updated state; `ValueError` from the session → 400 `{"error": msg}`;
  unknown path → 404; malformed JSON body → 400.
- `main()`: argparse `--config` (required), `--port` (overrides config),
  start-up log listing each NPC with its OCEAN vector.
- `make_server(config_path, port=0)` factory so tests can bind an
  ephemeral port.

Tests (`tests/test_live_server.py`): server in a daemon thread on port 0;
`urllib` calls covering every endpoint, the two `/event` payload forms,
error paths (unknown location → 400, bad JSON → 400, unknown path → 404),
and a state→step→event→step→reset scenario asserting the world round-trip.

Config: `experiments/rq3/live_config_demo.json` — Karlach / Shadowheart /
Laezel, all `scorer`, port 8973, seed 42.

## Task 5 — Unity live client

New files under `Assets/Scripts/Live/` (own folder, same
`Dissertation.Playback` assembly? No — new asmdef `Dissertation.Live`
referencing `Dissertation.Playback` + `UnityEngine.UI` to keep layering
explicit):

- `LiveDtos.cs` — `[Serializable]` DTOs mirroring the payloads
  (`LiveState`, `LiveLocation`, `LiveEvent`, `LiveNpc`, `StepResponse`,
  `StepEntry` with `overridden`, `ProbEntry`; request DTOs for `/event`).
- `LiveClient.cs` — thin coroutine wrappers over `UnityWebRequest`:
  `Get(path, onOk, onErr)`, `Post(path, body, onOk, onErr)`; base URL
  `http://127.0.0.1:8973`, overridable in the inspector.
- `LiveController.cs` (MonoBehaviour) — Connect button → `/state` →
  destroy + respawn NPC dots from an inactive prototype (colour palette,
  name label under the dot, action label above using
  `LocationLayout.LabelPlanes[slot % 3]`), build the event panel; Step
  (shared hotkey Space when live mode active) → `/step` → walk/perform
  each entry, `"(forced)"` suffix on overridden steps; R → `/reset`;
  all request errors → status line. Step is refused while any live NPC
  is walking (same all-or-nothing rule as playback).
- `ModeSwitcher.cs` — Mode dropdown (Playback / Live): enables one
  controller component + UI group, disables the other, hides the other
  mode's NPCs.
- `PlaybackSceneBuilder` — adds: Mode dropdown at the bar's far left;
  a `LiveGroup` (Connect button + its own status use; reuses Continue/
  Restart? No — live group gets its own Step/Reset buttons so listener
  wiring stays trivial); right-side event-panel container (vertical
  layout, hidden in playback mode); inactive `LiveNpcPrototype` with
  `NpcAgent` + labels; wires `LiveController` + `ModeSwitcher`.

Event panel rows: per location a header text + `[unlock]` toggle button +
one button per event showing `name: ON/OFF` (force events prefixed `!`).
Buttons are `DefaultControls.CreateButton` with text swapped on state.

## Task 6 — verification & wrap-up

Full pytest suite; commits per task; independent review agent over the new
Python and C#; user runbook (start server, Unity steps, expected
behaviours: event effects appear at the next step, tournament forces
spectate with `(forced)` label, unlock adds enemy_camp to the world and
the panel shows live state after every `/event`).

## Acceptance

1. `python -m experiments.rq3.live_server --config experiments/rq3/live_config_demo.json`
   serves all four endpoints; tests green without Unity.
2. In Unity: switch to Live, Connect shows three named NPCs; Step moves
   them per their personalities; toggling `festival` visibly shifts
   Karlach toward market over subsequent steps; activating
   `grand_tournament` forces everyone to arena/spectate with the
   `(forced)` label; deactivating releases them; unlock lets Laezel
   discover `enemy_camp`; Reset restores the authored world and replays
   identically.
3. Playback mode is byte-identical in behaviour to before this round.
