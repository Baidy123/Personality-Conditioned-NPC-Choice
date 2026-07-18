# configs/ — the files you edit day to day

Every command below runs from `code/`. This folder is the front door: the
run configs live here, and this table says where everything else is.

## Change X → edit Y

| I want to change...                              | Edit this file                          |
|--------------------------------------------------|-----------------------------------------|
| which sequences get generated for Unity playback (characters, policies, steps, seeds) | `configs/rq3_sequences.json` |
| the live demo (roster, sampling temperature, port, seed) | `configs/live_demo.json` |
| the **game** world: locations, actions, feature values, events | `configs/game_world.json` — edit freely |
| the **game** character roster: names + OCEAN vectors | `configs/game_personalities.json` — edit freely |
| the map layout / colours in Unity                | `unity/.../Assets/Scripts/Playback/LocationLayout.cs`, then re-run *Dissertation → Build Playback Scene* |

**The research/game split (decided 2026-07-18).** `data/world.json` and
`data/personalities.json` are the RQ1/RQ2 research sources — frozen; changing
them means a documented tuning round (`docs/tuning_log.md`). The
`configs/game_*.json` files started as exact copies and are what BOTH game
modes (playback generation and live demo) actually read — edit them freely.
If a future tuning round changes the research files and you want the game to
follow, re-copy or merge by hand; the two pairs do not sync automatically.

**One Unity caveat:** if you add a location to `game_world.json`, also add its
position/colour to `LocationLayout.cs` and rebuild the scene, or the players
will refuse the unknown id.

## Commands

```bash
# generate playback sequences + auto-sync into Unity (then just open Unity)
python -m experiments.rq3.gen_sequences --preview

# start the live-demo brain by hand (normally unnecessary: Unity's Connect
# button auto-starts it)
python -m experiments.rq3.live_server
```

Both take `--config <file>` to point at a different config.

## Config field notes

`rq3_sequences.json`
- `personalities`: `{"name": "Karlach"}` looks the vector up in
  `personalities_file`; add an inline `"ocean": {...}` to override or to
  define a one-off profile.
- `unity_dir`: after generating, `S*.json` are mirrored here (with stale-file
  cleanup) so the Unity playback dropdown always matches the last run. Remove
  the key to skip syncing.
- `out_dir` keeps the archival copies + `manifest.csv` (one row per sequence:
  which policy/personality/seed/temperature produced which file).
- `selection_temperature`: same knob as the live demo (0.1 sharp/in-character,
  1.0 raw); recorded per sequence in meta and manifest. A stimulus-design
  decision for the human study — discuss before changing for formal material.

## Choosing the model (policy) an NPC runs on

Both modes use the same rule: **no `checkpoint` field = the hand-authored
scorer; a `checkpoint` field = that trained model.** The `policy` / `name`
label is display text only — the checkpoint path decides the model.

Live demo — per NPC in `live_demo.json`:

```json
{"name": "Karlach",  "policy": "scorer"},
{"name": "Astarion", "policy": "nonlinear_2a",
 "checkpoint": "results/rq2/models/S0__nonlinear__s0.pt"}
```

Playback generation — per policy in `rq3_sequences.json` (every personality
is crossed with every policy):

```json
"policies": [
  {"name": "scorer"},
  {"name": "nonlinear_2b",
   "checkpoint": "results/rq2b/models/IND__nonlinear__wd0.001__s0.pt"}
]
```

Checkpoint shelf (seeds s0–s4 exist for each):

| family | main picks | note |
|---|---|---|
| 2A students (imitate the scorer) | `results/rq2/models/S0__nonlinear__s0.pt`, `S0__simple__s0.pt` | closest to scorer behaviour |
| 2A controls / variants | `S0__agnostic_*`, `G1..G6__*`, `S0__*__abl_*`, `S0__*__n*` | agnostic ignores personality (sanity check: identical for everyone); G/abl/n are research splits |
| 2B students (independent labels) | `results/rq2b/models/IND__nonlinear__wd0.001__s0.pt`, `IND__simple__s0.pt` | hard-label trained → near-deterministic; repetitive trails, and `selection_temperature` 0.1 sharpens that further |
| 2B controls | `IND__agnostic_nonlinear__wd0.001__s0.pt` | |

`live_demo.json`
- `npcs`: any number; `{"name": ...}` looks up the roster, inline
  `"ocean"` overrides, learned policies use
  `{"policy": "<label>", "checkpoint": "results/..."}` as above.
- `selection_temperature`: 0.1 = sharp, in-character behaviour (the value the
  tuning-log personas were validated at); 1.0 = raw sampling, much noisier.
- The full HTTP interface (endpoints, payloads) is documented at the top of
  `experiments/rq3/live_server.py`.
