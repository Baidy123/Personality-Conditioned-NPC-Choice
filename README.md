# Personality-Conditioned NPC Choice

Code for an MSc dissertation on personality-conditioned discrete NPC choice with
bounded recent-choice context. An NPC picks a **location**, then an **action** at
that location, conditioned on an OCEAN personality vector, the options currently
available, and a short memory of what it recently did.

Four policies share one interface: a hand-authored scorer, a personality-agnostic
control, and simple / nonlinear learned models. A Unity project replays the
generated behaviour sequences and runs a live demo.

## Install

Python 3.11+. Every command below runs from `code/`.

```bash
pip install -r requirements.txt
python -m pytest -q          # 181 tests, all should pass
python -m examples.demo 5    # 5 decision cycles per NPC
```

`torch` is only needed for the learned models (`npc_policy/learned.py`, all of
`experiments/rq2`); `matplotlib` only for the figure scripts. The hand-authored
scorer runs on numpy alone.

## Using a policy in your own code

```python
import numpy as np
from npc_policy import DecisionController, HandAuthoredScorer, load_world, load_personalities

world = load_world("data/world.json")
npcs  = load_personalities("data/personalities.json")

ctrl = DecisionController(
    HandAuthoredScorer(), mode="sample",
    rng=np.random.default_rng(42), selection_temperature=0.1,
)

karlach = npcs["Karlach"]
for _ in range(10):
    loc = ctrl.choose_location(karlach, world.resolve())
    act = ctrl.choose_action(karlach, world.actions_at(loc.option.id))
    print(loc.option.id, "->", act.option.id)
```

`world.resolve()` returns the locations that are currently unlocked, with active
event buffs already applied. The controller owns both memory buffers and clears
the action buffer automatically when the location changes. `mode="argmax"` gives
reproducible choices; `mode="sample"` draws from the distribution.

Need the distribution rather than a choice? `scorer.distribution(...)`. Need the
intermediate quantities (base score, mu, gamma, rep/sim/nov, temperature)?
`scorer.trace(...)`.

## Running the experiments

**RQ1 — behavioural expression** -> `results/rq1/`

```bash
python -m experiments.rq1.gen_cases   # matched cases; run this first
python -m experiments.rq1.run_e1      # trait-sensitivity curves
python -m experiments.rq1.run_e2      # profile-distance correlation
python -m experiments.rq1.run_e3      # trajectory patterns
python -m experiments.rq1.run_e4      # memory-context ablations
```

E1-E4 are independent of each other once `gen_cases` has run.

**RQ2 / 2A — learning from the hand-authored scorer** -> `results/rq2/`

Run the four in order. Pass `--smoke` to all of them first for a fast
end-to-end check before committing to the full matrix.

```bash
python -m experiments.rq2.gen_controlled   # dataset: train / val / G1-G6 splits
python -m experiments.rq2.train            # 130 runs; resumable, hours on CPU
python -m experiments.rq2.run_2a           # metrics, main table, figures
python -m experiments.rq2.run_e_diag       # E1-E4 diagnostic on the students
```

`train` takes `--device cuda` and `--only <run-prefix>` to repeat a single run.

**RQ2 / 2B — learning from the independent labels** -> `results/rq2b/`

```bash
python -m experiments.rq2.import_independent   # raw batches -> reviewed splits
python -m experiments.rq2.run_label_probe      # label health check; run before training
python -m experiments.rq2.train_2b
python -m experiments.rq2.run_2b
```

`run_label_consistency export|score` measures annotator test-retest agreement,
the practical ceiling on any 2B score.

## Game modes (Unity playback + live demo)

See **`configs/README.md`** — it is the front door for anything you would want to
change: which sequences get generated, the world and character rosters, how to
pick which trained model an NPC runs on, the artwork, and both run commands.

## Layout

```
npc_policy/          the policy package
  schema.py            feature schemas + OCEAN axes, per-level indexing, 12-dim padding
  representation.py    Option, Personality, RecentBuffer
  weights.py           hand-authored tables b / C / w / W_rel (+ bilinear fallback)
  config.py            ScorerConfig; LevelParams holds location vs action coefficients
  relations.py         rep / sim / nov computed against a buffer
  scorer.py            HandAuthoredScorer; its docstring carries the full equations
  learned.py           simple / nonlinear / agnostic torch models
  features.py          the learned-model input layout, numpy reference
  policies.py          one calling convention shared by all four policies
  controller.py        DecisionController: nested choice, buffer ownership, reset rule
  world.py             world loading + local-event resolution
  cases.py             decision-case formats and JSON (de)serialisation

configs/             run configs for the game modes — start at configs/README.md
data/                world.json, personalities.json, generated cases, 2B dataset
experiments/         rq1/ rq2/ — the analyses above; rq3/ drives the Unity modes
examples/            demo.py, acceptance_check.py, table/figure renderers
results/             experiment outputs, trained checkpoints
tests/               pytest suite
unity/dissertation/  Unity project: playback player + live demo
```

Dependency direction: `schema` -> `representation` / `weights` -> `relations` /
`scorer` -> `controller` / `world` / `cases`.

## Where the numbers come from

Everything marked `PROVISIONAL` in `config.py` and `weights.py` is a starting
value, tuned by hand and examined empirically in RQ1, not a validated constant.
Change coefficients in `config.py`, tables in `weights.py`;
`python -m examples.make_tables_figures` renders the current values.

World content lives in `data/*.json`, not in code, so changing the world means
editing JSON only.
