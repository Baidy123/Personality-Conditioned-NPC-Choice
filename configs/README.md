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

`live_demo.json`
- `npcs`: any number; `{"name": ...}` looks up the roster, inline
  `"ocean"` overrides, learned policies use
  `{"policy": "<label>", "checkpoint": "results/..."}`.
- `selection_temperature`: 0.1 = sharp, in-character behaviour (the value the
  tuning-log personas were validated at); 1.0 = raw sampling, much noisier.
- The full HTTP interface (endpoints, payloads) is documented at the top of
  `experiments/rq3/live_server.py`.
