# RQ2 Study 2A pipeline — implementation design record

Status: design approved 2026-07-10. Scope: the four `experiments/rq2/` modules,
their tests, and a Chinese runbook. The model layer
(`npc_policy/features.py`, `npc_policy/learned.py`) was implemented 2026-07-09.
Research-level decisions (model forms, dataset mixture, G-splits, training
protocol, metrics) live in `2026-07-09-rq2-models-and-2a-experiments.md` and are
not restated; this record covers how the pipeline is built and operated.

## 0. Execution model

The pipeline is four sequential CLI commands. **The user runs them in his own
terminal** (locally or on a server); Claude writes the code and the runbook and
analyses the result files afterwards. No Claude-supervised experiment runs.

```text
python -m experiments.rq2.gen_controlled   # 1. dataset          (~minutes)
python -m experiments.rq2.train            # 2. all training runs (~hours, resumable)
python -m experiments.rq2.run_2a           # 3. metrics + figures (~minutes)
python -m experiments.rq2.run_e_diag       # 4. E1–E4 diagnostic  (~tens of minutes)
```

Every script accepts `--smoke`: a small-scale end-to-end pass (separate output
directories `data/rq2_controlled_smoke/`, `results/rq2_smoke/`) run once to
verify the pipeline before the full run. The runbook
(`docs/rq2_runbook.md`, Chinese) documents each command, expected duration,
expected output files, success checks, and a "run on a server / GPU"
section (clone/upload, CPU vs CUDA torch install commands, `--device`,
`nohup`, copying results back). The user has opted to run training on a GPU
server; the same commands work unchanged on a local CPU.

## 1. Module layout

```text
experiments/rq2/__init__.py
experiments/rq2/common.py          paths, run-matrix definition, dataset I/O, eval helpers
experiments/rq2/gen_controlled.py  synthetic sampler + rollout collector + split manifests
experiments/rq2/train.py           resumable training runs
experiments/rq2/run_2a.py          metrics tables + figures
experiments/rq2/run_e_diag.py      student adapter + E1–E4 overlay on frozen RQ1 cases
tests/test_rq2_pipeline.py         acceptance tests (§6)
docs/rq2_runbook.md                Chinese operating manual
```

Figure/CSV conventions follow `experiments/rq1/common.py` (Okabe-Ito palette,
fixed entity colours, `write_csv`). Teacher = v1.3 scorer with `DEFAULT_CONFIG`,
frozen; `meta.json` records a SHA-256 hash of the serialised config and the
generation seed (base `20260709`, per the research spec).

## 2. Dataset generation (`gen_controlled.py`)

**Master pool, filtered splits.** One generation pass writes a master pool of
~140k labelled cases (`ControlledCase` JSONL, ids `syn-…`/`roll-…`), mixture
synthetic:rollout ≈ 7:3 per the research spec §5. Each G-split is then a
*filter* over the pool (rules from research spec §6), not a fresh generation:

- every split (including S0) subsamples its filtered pool to exactly
  **100k train / 5k val** — data volume is matched across splits;
- split membership is stored as manifests (`splits.json`: split id → case-id
  lists), so the pool file is written once and never duplicated;
- filters are computed from case *content* (personality, candidate features,
  buffers, world tag), not from stored labels, and are pinned by tests.

**Targeted test sets.** Each split's held-out set (≥5k) is generated directly
under the split's test condition (e.g. G1: personalities sampled inside
`O > 0.5 ∧ C < −0.5`; G3: location sets of size 2 and 8; G5: buffers with three
consecutive repeats), labelled by the same frozen teacher.

**Split-specific notes.**

- Synthetic cases sample options from the base world (plus σ = 0.1 perturbed
  variants); rollout trajectories cover all four RQ1 world variants
  (`data/rq1_cases/worlds/`), 50 rounds each, sample mode. G4's train filter
  therefore only removes non-base rollout cases.
- G6 (`arena` left out): rollout location cases always contain `arena`, so the
  G6 train pool adds extra rollouts on an arena-locked world variant to
  preserve the mixture.

Smoke mode: ~2k pool, ~300 per test set, same code paths.

## 3. Training runs (`train.py`)

**Run matrix (130 runs).** One run = split × model × seed, seeds 0–4:

| group | runs |
|---|---|
| S0 main: simple, nonlinear, agnostic-simple, agnostic-nonlinear | 20 |
| G1–G6: simple, nonlinear | 60 |
| context ablations (no-context, location-only) × simple, nonlinear | 20 |
| data-size curve N ∈ {1k, 5k, 20k} × simple, nonlinear | 30 |

Reductions vs a full crossing (recorded design decisions): the agnostic
controls train **on S0 only** — they answer "what is personality information
worth", which the G-splits do not ask; the 100k point of the data-size curve
reuses the S0 main runs. Ablation runs retrain on transformed copies of the S0
data (relations zeroed in the *data*: both levels for no-context, action level
only for location-only), matching the "retrained, not input-zeroed" rule.

**Protocol** (research spec §7): Adam lr 1e-3, batch 256 (`PolicyBatch`
minibatches), KL loss, early stopping on val KL with patience 10,
max 150 epochs; dropout 0.1 / weight decay 1e-4 on the nonlinear model.

**Device.** `train.py` takes `--device {auto,cpu,cuda}` (default `auto`:
cuda if available). Training dtype follows the device — float64 on CPU,
float32 on CUDA (consumer GPUs run float64 at a fraction of float32 speed).
Saved weights are always cast back to float64, so evaluation
(`run_2a.py`, `run_e_diag.py`) and the numpy-parity tests stay CPU/float64
regardless of where training ran. The run-result JSON records the device and
dtype used.

**Resumability.** Each run has a deterministic id (`S0__simple__s0`); on
completion it writes `results/rq2/runs/<id>.json` (final val KL, epochs,
wall-time, config) and `results/rq2/models/<id>.pt`. On start-up `train.py`
skips ids whose result file exists — interrupting and re-running the same
command continues where it left off.

## 4. Metrics (`run_2a.py`)

Evaluates every trained model on its split's test set **and** the S0 test set
(no-general-degradation check), per decision type. Outputs to `results/rq2/`:

1. `main_table.csv` — KL(teacher‖student), JSD, top-1 agreement; mean ± std
   over seeds, per split × model × decision type.
2. `gap_by_split.png` — simple-vs-nonlinear KL gap across S0 + G1–G6.
3. `data_size_curve.png` — KL vs N ∈ {1k, 5k, 20k, 100k}, both families.
4. `diagnostics.csv` — ‖w[c_L ⊙ o]‖ for each simple run (expected ≈ 0);
   agnostic-vs-personality gap on S0; ablation deltas.

## 5. Structural diagnostic (`run_e_diag.py`)

A thin `StudentTraceAdapter` (in `experiments/rq2/`) makes a trained student
duck-type the scorer interface used by the RQ1 pipeline:
`trace(personality, candidates, buffer, level)` computes relations from the
buffer, calls `predict_distribution`, and returns a ScoreTrace-compatible
object whose `P_rule` is the student distribution. Action calls need the
selected location's context, which the scorer interface does not carry, so the
adapter holds a `current_location` field set by the diagnostic's own trajectory
runner (a local copy of `rq1.common.run_trajectory` with one extra line after
each location decision); `experiments/rq1/` itself is not modified.

Runs the S0-trained simple and nonlinear students (all 5 seeds; figures show
seed means) on the frozen RQ1 matched cases (`data/rq1_cases/`): E1 sweep
overlays (teacher vs student, per trait — the pre-registered N-temperature
check), E2 Spearman/Mantel, E3/E4 trajectory statistics under the RQ1 protocol.
Outputs to `results/rq2/e_diag/`.

## 6. Test plan (acceptance = all green)

1. **Label correctness**: sampled pool cases' `target_distribution` equals the
   frozen teacher recomputed on the case inputs (tolerance 1e-12).
2. **Split filters**: each G-split's train manifest contains no violating case
   (e.g. no G1 personality inside the excluded region; no `arena` anywhere in
   G6 train candidates or buffers); each test set satisfies its condition.
3. **Training loop**: on a small bilinear-teacher dataset with `lambda_N = 0`,
   the simple model's train KL approaches 0 (representable per the 2026-07-09
   certificate); early stopping triggers on a plateaued val curve; a re-invoked
   `train.py` skips completed run ids.
4. **Metrics**: KL/JSD/top-1 pinned by hand-computed miniature examples.
5. **Adapter**: `StudentTraceAdapter.trace` matches a direct
   `predict_distribution` call on identical inputs (both levels, empty and
   non-empty buffers); the local trajectory runner clears the action buffer on
   location change (mirrors `DecisionController` semantics).
6. **Smoke invariant**: `--smoke` exercises the same code paths as the full
   run (only sizes, epochs, and output directories differ).

## 7. Out of scope this round

- Actually running the full experiments (the user runs them from the runbook;
  analysis of `results/rq2/` is a later conversation).
- 2B generation/review protocol (separate spec, TBD).
- Study 3 automated comparison and any scorer tuning (teacher frozen).
