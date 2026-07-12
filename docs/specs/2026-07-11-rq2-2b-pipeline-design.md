# RQ2 Study 2B pipeline — design

Date: 2026-07-11. Status: approved design, pre-implementation.
Depends on: `docs/specs/2026-07-09-rq2-models-and-2a-experiments.md` (model forms),
`docs/specs/2026-07-10-rq2-2a-pipeline-design.md` (training loop, run/result conventions),
`project_flow.md` §3b (independent dataset requirements).

## 1. Goal and division of labour

Build the scorer-independent dataset and the 2B training/evaluation stage. Labels
come from a chat LLM prompted by the user; no API pipeline is built (decided
2026-07-11: the user generates and reviews data manually in a chat UI, code owns
formats, numbers, and training). The AI invents scenarios and choices **by name
only**; all numeric features are attached by code afterwards.

Three deliverables:

1. a **generation guide** the user pastes to a chat LLM (`docs/rq2b_generation_guide.md`);
2. an **import script** `experiments/rq2/import_independent.py` (validate → enrich →
   split → report);
3. **training and evaluation** `experiments/rq2/train_2b.py` / `run_2b.py`
   (cross-entropy on hard labels; comparative main table = Study 3 automated
   comparison).

Isolation rule (unchanged from `project_flow.md`): the scorer, its tables, its
coefficients, controlled labels, and controlled-model weights never touch 2B
training. The scorer appears only as an *evaluated system* on the finished test set.
`rep/sim/nov` are part of the shared representation (model inputs), not scorer
knowledge; they are computed by `npc_policy.relations` with `DEFAULT_CONFIG`
buffer/recency settings, identically for every policy.

## 2. Raw data format (what the AI outputs)

One JSON array per batch file, saved by the user under
`data/rq2_independent/raw/*.json`. Case object:

```json
{
  "personality": {"O": 0.7, "C": -0.4, "E": 0.6, "A": 0.1, "N": -0.2},
  "decision_type": "location",
  "recent_locations": ["market", "market", "tavern"],
  "candidates": ["tavern", "library", "arena"],
  "choice": "tavern",
  "reason": "gregarious and impulsive; returns to the lively room"
}
```

Action cases add `"selected_location"` and `"recent_actions_same_location"`
(list of action names at that location, empty after a location change), and their
`candidates`/`choice` are action names native to the selected location.
No numeric feature fields exist in the format; any present are ignored with a
warning. `reason` is retained for review only (never a model input or target).

Constraints the guide states and the validator enforces:

- names must come from the base-world lists (7 locations, native actions);
- trait values in [−1, 1]; five traits present;
- `recent_locations` length ≤ 3 (K_L), `recent_actions_same_location` ≤ 3 (K_A);
- action cases: newest `recent_locations` entry equals `selected_location`
  (2A invariant), recent actions native to that location;
- location candidates: 2–7 unique names; action candidates: the selected
  location's full native action set;
- `choice` must be one of `candidates`.

Diversity quotas (guide instructions, report-audited, not hard-rejected):
location/action cases ≈ 50/50; per trait, high (> 0.3) / mid / low (< −0.3)
each ≥ 20% across a batch; empty-history cases ≈ 20%.

The guide also instructs two separately requested batches for the structured
test groups (§4) and carries a one-line semantic description per location and
action (drafted at implementation, reviewed by the user in the guide file).

## 3. Import script

`python -m experiments.rq2.import_independent` rebuilds everything from `raw/`:

1. **Validate** every case against §2; failures go to `rejected.jsonl` with a
   reason code, accepted cases get a stable id (file + index). The user's own
   rejections are expressed by deleting cases from raw files or marking
   `"review_status": "rejected"` in place.
2. **Enrich**: look up feature vectors from `data/world.json`; compute
   `rep/sim/nov` via `npc_policy.relations` (empty history → relations `None`,
   matching 2A); build `IndependentCase` objects (`npc_policy/cases.py`) with
   provenance `source` (model name the user records once per raw file in a
   `"_meta"` header object) and `review_status = "accepted"`.
3. **Split** (seeded, deterministic): structured-test filters first —
   personality in the G1 region (O > 0.5 ∧ C < −0.5) or case touches the arena
   family → eligible only for the structured test groups (≈38 unseen-personality
   + ≈37 arena, target 75); remaining accepted cases split 550 train / 100 val /
   75 iid test, scaled proportionally when fewer are available. Exact-duplicate
   cases are dropped (keep first).
4. **Report** `report.txt`: acceptance counts, rejection reasons, quota
   coverage per trait band and decision type, split sizes, and per-group
   shortfalls phrased as "request N more cases of X".

Outputs under `data/rq2_independent/`: `cases.jsonl`, `rejected.jsonl`,
`splits.json`, `meta.json` (counts, world hash, import version), `report.txt`.

## 4. Test-set structure

150 isolated test cases (decided 2026-07-11, option B):

| group | n | content |
|---|---|---|
| iid | 75 | random holdout from the general pool |
| unseen personality | ≈38 | G1 region O > 0.5 ∧ C < −0.5 (same region as 2A G1) |
| unseen family | ≈37 | cases involving `arena` or its actions (same family as 2A G6) |

Train/val never contain G1-region personalities or arena-family content (the
import script enforces this; it does not rely on the AI following instructions).
Test cases are excluded from `report.txt` details (aggregate counts only) to
keep prompt iteration blind to test content.

## 5. Training

`train_2b.py` reuses the 2A loop (`run_all`, resume-by-file, atomic writes,
device selection) with:

- **loss**: masked cross-entropy `-log q[target_choice]` (new ~10-line function
  beside `kl_loss`; targets are indices, built from `IndependentCase.target_choice`);
- **models**: simple, nonlinear, agnostic_simple, agnostic_nonlinear × seeds 0–4
  (20 runs), fresh `build_model` instances (no controlled weights);
- **hyperparameters**: Adam lr 1e-3, batch 64, max 500 epochs, patience 30
  (validation NLL); nonlinear weight decay selected on val from
  {1e-4, 1e-3, 1e-2} (3 extra runs per seed for the two nonlinear variants);
  simple stays at 0 [PROVISIONAL numbers, forms fixed];
- **case → inputs**: a 2B variant of `case_to_inputs` reading `target_choice`
  (index) instead of `target_distribution`; `read_pool` generalised to take the
  case class.

Results under `results/rq2b/runs/` and `models/` with the 2A naming scheme.

## 6. Evaluation (includes Study 3 automated comparison)

`run_2b.py` evaluates on the 150 test cases:

- **systems**: uniform baseline, agnostic simple/nonlinear, hand-authored scorer
  (argmax of its distribution for top-1; its distribution for NLL), trained
  simple, trained nonlinear;
- **metrics**: top-1 accuracy and mean NLL of the labelled choice;
- **breakdowns**: overall × test group (iid / unseen-personality / arena) ×
  decision type; learned models aggregated over seeds (mean ± sd);
- **outputs**: `results/rq2b/main_table.csv` (the dissertation's 2B core table),
  `group_bars.png`, `diagnostics.csv` (per-case records for error analysis).

## 7. Testing

`tests/test_rq2b_pipeline.py`: validator accept/reject cases for every §2 rule;
enrichment parity against `npc_policy.features`/`relations` on a handcrafted
case; split isolation property (no G1-region/arena content in train/val);
CE loss vs a manual computation; end-to-end smoke on a checked-in 12-case raw
fixture through import → train (2 epochs) → eval.

## 8. Decisions resolved here vs still open

Resolved (2026-07-11): manual chat-based generation route; names-only raw
format; hard single-choice label (confirms the v1 provisional form); 75/75 test
structure reusing the G1 region and arena family; import-enforced isolation;
CE training reusing the 2A loop; scorer evaluated as a system on the same table.

Still open (do not block implementation): final accepted-case count (target
~800, actual depends on review yield — splits scale proportionally); nonlinear
weight-decay grid values; whether `reason` strings are quoted in the
dissertation (kept in data either way).

## 9. Amendment (2026-07-12)

- **Scale raised**: general-pool split targets are now 1200 train / 150 val /
  75 iid test (was 550/100/75; the old numbers were a review-throughput guess,
  and the user judged them too small). Structured test targets unchanged.
  Proportional scaling below target still applies.
- **Data-size curve added**: simple and nonlinear (weight decay pinned to
  1e-3) additionally train at 150/400/800 cases × 5 seeds (run matrix 40 → 70
  runs); `run_2b.py` writes `data_size_curve.csv/.png` with the fixed scorer
  as a reference line. Purpose: decide with evidence whether further scaling
  (or the shelved training-only location expansion) is worth its review cost.
- **Review protocol stratified**: the 150 test cases are reviewed one by one;
  train/val batches get outlier deletion plus a ~20% spot check. Label noise
  in training is tolerable; test labels are load-bearing.

## 10. Amendment (2026-07-12b): test-only held-out location

The unseen-family test group no longer uses arena. A test-only location —
`infirmary` (tend / assist / vigil; a helping-heavy family absent from the
deployment world) — is authored in `data/rq2b_test_world.json` and merged into
the 2B world at import; `data/world.json` is untouched, so RQ1/2A hashes are
unaffected.

Rationale (user decision): excluding arena cost the learned models one sixth of
the deployment world's training coverage, handicapping exactly the property the
comparison is about (data-driven coverage). Now training keeps all six
deployment locations; family-level extrapolation is tested on a location that
is genuinely unseen by every trained model. The scorer needs no training, so
the comparison stays fair. Trade-off accepted: the direct 2A-G6 alignment
(same held-out family in both studies) becomes a qualitative comparison.

Mechanics: split group `test_arena` renamed `test_family`; the isolation
filter applies the same family rule (selected location, candidates, or any
recent-locations entry) to `infirmary`; General batches now allow arena and
forbid infirmary; the third batch type is "Held-out location batch".
