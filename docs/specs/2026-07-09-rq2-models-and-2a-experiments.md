# RQ2 learned policies and Study 2A experiments — design record

Status: design approved in conversation 2026-07-09 (model forms, agnostic-control
definition, data mixture, dataset sizes); recorded in `project_flow.md` §4a/§4c/§4d/§6b
and `CLAUDE.md`. Implementation pending. Study 2B runs the same architectures on the
independent dataset; its generation/review protocol is a separate spec (TBD).

RQ2: *how well can simple and nonlinear learned policies acquire
personality-conditioned choice relationships, generalize to unseen combinations of
personalities, options, and recent-choice contexts, and how do they compare with
hand-authored and personality-agnostic policies on independent test cases?*

Study 2A is the controlled half: the v1.3 scorer (`npc_policy/scorer.py`) is the
teacher, its full `P_rule` distribution is the soft label, and ground truth is known
by construction.

## 1. Shared interface

All learned policies implement the same call as the scorer: per-candidate scalar
scores followed by one softmax over the candidate set. Inputs per candidate
(all already available from `cases.py`):

```text
p        (5,)   OCEAN in [-1, 1]
o_i      (12,)  padded candidate features
rel_i    (3,)   rep_i, sim_i, nov_i        (all zero when the buffer is empty)
d               decision_type ∈ {location, action}
c_L      (12,)  padded selected-location features (zeros for location cases)
```

Per-candidate scoring gives variable set sizes and order equivariance for free and
matches the teacher's structure (the candidate set enters only through the softmax
normaliser). No set encoder.

## 2. Simple policy [form SET]

```text
phi_i   = concat( o_i, o_i², rel_i,
                  vec(p ⊗ o_i), vec(p ⊗ rel_i),
                  c_L ⊙ o_i )                  # 12+12+3+60+15+12 = 114
score_i = w_d^T phi_i                          # w_location, w_action: 2 × 114 = 228 params
```

- `phi_i` contains only candidate-varying terms (softmax cancels case constants).
- `o_i²` makes the unclipped ideal-point base score representable
  (`−w_f(o_f − μ_f(p))²` expands into `[o², o, p⊗o]` terms). Not representable:
  the `[0,1]` clip on `μ`, the `(Cp)²` personality-quadratic terms, and the
  N temperature (personality-dependent multiplicative rescaling). These are the
  predicted sources of the simple-versus-nonlinear gap.
- `sim_i + nov_i` encodes buffer non-emptiness; all three relation features stay.
- `c_L ⊙ o_i` should learn ≈ 0 under controlled supervision (the teacher's action
  score ignores location features); report the learned norm as a diagnostic.
- Optimiser: full-batch L-BFGS or Adam; the model is convex in `w` given the
  KL objective, so seeds matter little — still run the standard 5 seeds for
  protocol uniformity.

## 3. Nonlinear policy [form SET; sizes provisional]

```text
e_p     = MLP_p(p)                        # 5 → 64 → 32
e_o_i   = MLP_o([o_i, rel_i])             # 15 → 64 → 32
e_c     = MLP_c([onehot(d), c_L])         # 14 → 16
score_i = MLP_head([e_p, e_o_i, e_p ⊙ e_o_i, e_c])   # 112 → 64 → 1
```

GELU activations; ~13k parameters. Dropout 0.1 and weight decay 1e-4 as defaults
(matter mainly for 2B; harmless at 2A data volumes). Framework: PyTorch, CPU.

## 4. Personality-agnostic control [SET]

Same two architectures, same data, `p` zeroed at train and test time. Plus an
untrained floor: candidate-frequency baseline (uniform over candidates, since
candidate identity varies; equivalently the empirical marginal where ids repeat).
The trained controls isolate the value of personality information at matched
capacity and data.

## 5. Controlled dataset [sizes PROVISIONAL]

Mixture synthetic:rollout ≈ 7:3. Targets: 100k train / 5k val / ≥5k per test split.
Labels: full `P_rule` from the v1.3 scorer with `DEFAULT_CONFIG` (frozen for the
whole study; record the config hash in the dataset file). Format: `ControlledCase`
(`cases.py`), JSON lines.

**Synthetic sampler** (coverage): `p ~ U[-1,1]^5`; location candidate sets of size
2–8 drawn from a pool of real options (`data/world.json`) plus perturbed variants
(feature-wise Gaussian noise, σ = 0.1, clipped to [0,1], padding conventions
respected); buffers of length 0–`K_L` filled from the same pool; relations computed
by `relations.py`. Action cases: pick a location, use its (possibly perturbed)
action set, action buffer per the same-location rule.

**Rollout collector** (realism): `DecisionController` in sample mode over the world
variants in `data/rq1_cases/worlds/`, 50 rounds per trajectory, personalities drawn
uniformly; each decision's inputs + `trace.P_rule` become one case.

Seed conventions follow RQ1 (`20260709` base).

## 6. Structured generalisation splits

Each split filters the training pool, trains fresh models, and evaluates on its
held-out set plus the iid set S0 (to confirm no general degradation).

| id | split | construction |
|---|---|---|
| S0 | iid held-out | random case-level split |
| G1 | unseen personalities | exclude the region `O > 0.5 ∧ C < −0.5` from all training personalities; test only inside it |
| G2 | unseen feature combinations | exclude candidates with `risk > 0.6 ∧ privacy > 0.6` from training; test cases contain ≥ 1 such candidate |
| G3 | unseen set sizes | train on location sets of 3–6; test on 2 and 8 |
| G4 | unseen event-induced changes | train on the base world only; test on buffed/locked variants (`celebration`, `war_camp`, `market_locked`) |
| G5 | unseen history patterns | train with ≤ 2 consecutive same-location repeats in buffers; test with 3 |
| G6 | leave-one-family-out | exclude `arena` and its actions from training; test cases include it |

Threshold values in G1/G2/G5 are provisional; fix them before generation and do not
tune them afterwards.

## 7. Training protocol

- Loss: `KL(P_rule ‖ P_model)` averaged over cases.
- Adam, lr 1e-3, batch 256 cases (ragged sets flattened, segment softmax),
  early stopping on val KL (patience 10 epochs).
- 5 seeds per configuration; report mean ± std.
- Data-size curves: N ∈ {1k, 5k, 20k, 100k} on the S0 configuration.
- Context ablations (retrained, not input-zeroed): (a) no recent-choice context
  (relations zeroed in the *data*), (b) location relations only, (c) full.

## 8. Metrics and analyses

Primary, reported per decision type and per split, teacher as reference:

- mean KL(teacher ‖ student) in nats; JSD as the symmetric check;
- top-1 agreement with the teacher's argmax.

Secondary:

- data-size curves (KL vs N, both families, one figure);
- simple-vs-nonlinear gap per split (the G-splits say *where* capacity matters);
- `‖w[c_L ⊙ o]‖` diagnostic for the simple model.

**Structural diagnostic (E1–E4 on the student):** run the RQ1 pipeline
(`experiments/rq1/`) with the trained S0 students as drop-in policies on the
frozen matched cases (`data/rq1_cases/`); overlay teacher and student E1 sweep
curves, compare E2 Spearman ρ / Mantel, and E3/E4 trajectory statistics under the
RQ1 protocol (50 rounds × 10 seeds, sharpened sampling). Pre-registered
expectation: the simple model fails to track the N temperature channel (E1 N-sweep
on `P_rule` concentration); the nonlinear model follows it.

Figures: Okabe-Ito palette and fixed entity-colour assignment via
`experiments/rq1/common.py` conventions; outputs to `results/rq2/` as PNG + CSV.

## 9. Planned module layout

```text
npc_policy/features.py        phi construction + padding for learned inputs
npc_policy/learned.py         SimplePolicy, NonlinearPolicy, AgnosticWrapper (torch)
experiments/rq2/gen_controlled.py   synthetic sampler + rollout collector + splits
experiments/rq2/train.py            training loop, early stopping, seeds
experiments/rq2/run_2a.py           metrics, curves, split table
experiments/rq2/run_e_diag.py       E1–E4 structural diagnostic on students
```

## 10. Out of scope here

- 2B generation prompt, review protocol, acceptance rules, label form (separate
  spec; provisional scale ~1,200 generated → ~800 accepted is already recorded).
- Study 3 automated comparison (needs the 2B test set; reuses these policies
  unchanged through the shared interface).
- Any tuning of scorer tables or λ coefficients (the teacher is frozen for 2A).
