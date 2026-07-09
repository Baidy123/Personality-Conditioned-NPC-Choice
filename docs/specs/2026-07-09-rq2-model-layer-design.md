# RQ2 model layer — implementation design record

Status: design approved 2026-07-09; implemented same day (all §5 tests green — 49 passed, 0 warnings; as-executed deviations listed in `docs/plans/2026-07-09-rq2-model-layer.md` §Post-review amendments). Scope: **model layer only**
(`npc_policy/features.py`, `npc_policy/learned.py`, `tests/test_learned.py`).
Dataset generation, the training loop, 2A metrics, and the E1–E4 diagnostic are
later rounds. Research-level decisions (model forms, agnostic-control definition)
live in `2026-07-09-rq2-models-and-2a-experiments.md` and are not restated here.

## 1. Modules and boundaries

```text
npc_policy/features.py     numpy reference for the phi layout + input assembly
npc_policy/learned.py      torch: PolicyBatch, SimplePolicy, NonlinearPolicy,
                           AgnosticPolicy, UniformBaseline, KL loss
tests/test_learned.py      property tests (see §5)
```

- Only `learned.py` imports torch (CPU build, installed 2026-07-09); every other
  `npc_policy` module stays pure numpy.
- `features.py` (written 2026-07-09, reviewed as part of this design) is the
  single authority for the 114-dim `phi` order:
  `[o, o², rel, p ⊗ o, p ⊗ rel, c_L ⊙ o]`, trait-major bilinear blocks.
  `learned.py` rebuilds the same layout in torch; a test pins the two to equality.

## 2. Batch representation [decided: padded + mask]

A `PolicyBatch` holds one batch of B cases padded to the largest candidate count
M_max in the batch:

```text
p     (B, 5)    float   personality
d     (B,)      long    decision type: 0 = location, 1 = action
ctx   (B, 12)   float   padded selected-location context (zeros for location cases)
cand  (B, M, 12) float  padded candidate features (zero rows beyond each case's m_b)
rel   (B, M, 3) float   rep / sim / nov (zero rows likewise)
mask  (B, M)    bool    True at real candidates
```

Padding positions receive score −inf before the softmax, so they carry zero
probability and zero gradient. The flat + segment-index alternative was
considered and rejected: no practical memory benefit at ≤ 8 candidates and
~100k cases, and materially harder to debug.

Constructor `PolicyBatch.from_cases(inputs)` takes per-case dicts produced by a
helper `case_inputs(personality, candidates, decision_type, relations=None,
selected_location=None)` in `features.py` — the same raw materials the scorer
receives, so dataset rows and live decisions build batches identically.

## 3. Models

All models are `torch.nn.Module`s scoring each candidate independently, then
softmax within the mask — variable set sizes and order equivariance by
construction, matching the teacher's structure.

- **SimplePolicy**: parameter `w (2, 114)`; forward builds `phi` in torch and
  scores `(phi * w[d]).sum(-1)`. 228 parameters.
- **NonlinearPolicy** (research spec §3): towers `MLP_p 5→64→32`,
  `MLP_o 15→64→32`, `MLP_c 14→16`, head `112→64→1` over
  `[e_p, e_o, e_p ⊙ e_o, e_c]`; GELU; dropout 0.1 on hidden layers;
  ~13k parameters. Sizes provisional, set as constructor defaults.
- **AgnosticPolicy**: wraps either model; zeroes `p` in the batch before
  delegating. Used at train and test time (the trained control).
- **UniformBaseline**: untrained floor; uniform over the mask.

## 4. Interfaces

Two exits per model:

1. `forward(batch) → log-probabilities (B, M)` — training path; padding
   positions are −inf.
2. `predict_distribution(personality, candidates, decision_type, relations=None,
   selected_location=None) → np.ndarray (m,)` — single-case numpy path, parallel
   to `HandAuthoredScorer.distribution(...)`, so trained students drop into the
   RQ1 E1–E4 pipeline and Study 3 unchanged. Runs under `torch.no_grad()`, eval
   mode.

Loss: `kl_loss(log_q, target, mask) → scalar` — mean over cases of
`Σ_i target_i (log target_i − log q_i)` within the mask, `xlogy`-safe at
`target_i = 0`. Lives in `learned.py`; the training loop that consumes it is a
later round.

## 5. Test plan (acceptance = all green)

1. **phi parity**: torch phi equals `features.phi_matrix` to float64 tolerance
   on random inputs (both decision types, empty and non-empty relations).
2. **Permutation equivariance**: shuffling candidates permutes
   `predict_distribution` output without changing values (both models, random
   init).
3. **Batch/single consistency**: a case scored alone equals the same case inside
   a mixed-size batch (mask correctness, eval mode).
4. **Agnostic invariance**: `AgnosticPolicy` output is identical across
   different personalities on otherwise equal inputs.
5. **Probability validity**: probabilities sum to 1 within the mask and are
   exactly 0 outside it.
6. **Gradient smoke test**: NonlinearPolicy overfits 30 synthetic
   teacher-labelled cases (few hundred Adam steps) to a clearly lower KL;
   SimplePolicy reaches near-zero KL on a bilinear-teacher set with the
   N temperature disabled (`lambda_N = 0` — the temperature is on the
   not-representable list) and an overdetermined case count, so the pass
   genuinely certifies representability rather than interpolation.

Determinism: fixed torch/numpy seeds per test; CPU only.

## 6. Out of scope this round

- `experiments/rq2/` (generation, training, metrics, diagnostics).
- Any real training run or dataset file.
- Hyperparameter choices beyond constructor defaults (research spec marks them
  provisional).
