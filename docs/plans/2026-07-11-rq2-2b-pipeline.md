# RQ2 Study 2B Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the scorer-independent (2B) data pipeline — generation guide, import/validate/enrich/split script, hard-label training reusing the 2A loop, and the comparative evaluation table.

**Architecture:** The AI (driven manually by the user in a chat UI) emits names-only JSON batches; `import_independent.py` validates them against the base world, attaches feature vectors and rep/sim/nov, and produces `cases.jsonl` + splits with enforced test isolation. Training reuses `train.run_all`/`train_one` unchanged except two new keyword parameters, because cross-entropy on a hard label equals `kl_loss` with a one-hot target. Evaluation runs every system (uniform, scorer, trained models) on the same 150 test cases.

**Tech Stack:** Python 3.11, numpy, torch (CPU float64), pytest; existing `npc_policy` package and `experiments/rq2` conventions (atomic writes, resume-by-file, pool JSONL with `gen` tags).

**Spec:** `docs/specs/2026-07-11-rq2-2b-pipeline-design.md`. Working dir for all commands: `code/`.

**File map:**

| File | Role |
|---|---|
| `docs/rq2b_generation_guide.md` | Create — pasted to the chat LLM by the user |
| `npc_policy/…` | Untouched |
| `experiments/rq2/common.py` | Modify — `read_pool(case_cls=…)`, `RunSpec.tag`, extract `case_input_dict` |
| `experiments/rq2/train.py` | Modify — `train_one(to_inputs=…, weight_decay=…)` |
| `experiments/rq2/independent.py` | Create — parse / validate / enrich / to-inputs / constants (library) |
| `experiments/rq2/import_independent.py` | Create — CLI: split + report + outputs |
| `experiments/rq2/train_2b.py` | Create — 40-run matrix, loader, drift check |
| `experiments/rq2/run_2b.py` | Create — comparative evaluation, main table, figure |
| `tests/test_rq2b_pipeline.py` | Create — all 2B tests |
| `docs/rq2_runbook.md` | Modify — append the 2B runbook section (Chinese) |

---

### Task 1: Generation guide document

**Files:**
- Create: `docs/rq2b_generation_guide.md`

No code. The guide is a deliverable the user pastes to a chat LLM.

- [ ] **Step 1: Write the guide**

Create `docs/rq2b_generation_guide.md` with exactly this content:

````markdown
# NPC decision-case generation guide (Study 2B)

You are generating decision cases for a fantasy-village NPC simulation. Each case
describes an NPC (a personality), a situation (where it has been recently, what it
can do now), and your judgement of what this NPC would choose. Judge as a
thoughtful human observer: stay in character, weigh comfort, boredom, and
curiosity against the personality. There is no "correct" answer; give the choice
this specific NPC would most plausibly make.

## Output format

Reply with a JSON array only — no prose around it. First element is a metadata
header; every other element is one case:

```json
[
  {"_meta": {"source": "<your model name and version>"}},
  {
    "personality": {"O": 0.7, "C": -0.4, "E": 0.6, "A": 0.1, "N": -0.2},
    "decision_type": "location",
    "recent_locations": ["market", "market", "tavern"],
    "candidates": ["tavern", "library", "arena"],
    "choice": "tavern",
    "reason": "one short sentence"
  },
  {
    "personality": {"O": -0.8, "C": 0.9, "E": -0.5, "A": 0.3, "N": 0.4},
    "decision_type": "action",
    "selected_location": "library",
    "recent_locations": ["chapel", "library"],
    "recent_actions_same_location": ["read"],
    "candidates": ["read", "research", "discuss"],
    "choice": "read",
    "reason": "one short sentence"
  }
]
```

## Personality

Five traits, each a number in [-1, 1] (0 = average):
O openness (curiosity, novelty seeking), C conscientiousness (order, routine),
E extraversion (sociability, stimulation), A agreeableness (warmth, cooperation),
N neuroticism (anxiety, volatility). You invent the personality for each case.

## World vocabulary (use these names only; never invent new ones)

Locations:
- `tavern` — a lively drinking house: noisy, social, little privacy, the odd brawl
- `library` — quiet halls of books: solitary study, order, deep thought
- `chapel` — a calm sanctuary: ritual, reflection, quiet confession
- `market` — a bustling trade square: haggling, browsing, deal-making
- `forest` — wild ground beyond the walls: roaming, foraging, solitude
- `arena` — a fighting ground: combat bouts, training, spectacle, betting

Actions per location (an action case's `candidates` must be this full list):
- tavern: `chat` (friendly talk), `drink` (loosen up alone or in company),
  `brawl` (start or join a fist-fight)
- library: `read` (quiet reading), `research` (dig into a problem),
  `discuss` (debate ideas with another scholar)
- chapel: `pray` (formal ritual), `meditate` (silent stillness),
  `confess` (unburden to the priest)
- market: `haggle` (push for a better price), `browse` (wander the stalls),
  `trade` (do business with a partner)
- forest: `explore` (push into unknown ground), `forage` (gather food),
  `rest` (settle down and recover)
- arena: `fight` (real bout), `spar` (practice bout), `drill` (disciplined
  training), `coach` (train someone else), `spectate` (watch the bouts),
  `bet` (wager on outcomes)

## Rules

1. `decision_type` is `"location"` or `"action"`; aim for half of each per batch.
2. Location cases: `candidates` = 2–6 distinct location names;
   `recent_locations` = 0–3 names, oldest first (the NPC's last visits);
   no `selected_location`, no `recent_actions_same_location`.
3. Action cases: `selected_location` = where the NPC is; `candidates` = that
   location's full action list; the last entry of `recent_locations` must equal
   `selected_location`; `recent_actions_same_location` = 0–3 action names of
   that same location (what it just did there — empty if it only just arrived).
4. `choice` must be one of `candidates`. Numbers only in `personality`; never
   output feature values or probabilities.
5. Vary personalities: within a batch, each trait should appear high (> 0.3),
   middling, and low (< -0.3) in at least ~20% of cases each. About 20% of
   cases should have empty history. Vary candidate subsets and history patterns
   (repeats, alternations, fresh arrivals).
6. Consider the history: an NPC that has repeated one place or act may be bored
   of it (or, if it loves routine, comforted by it) — let the personality decide.

## Batch types (the requester tells you which one)

- **General batch**: rules above, plus: never use `arena` (or its actions) and
  never give personalities with O > 0.5 together with C < -0.5.
- **Personality batch (test)**: every case's personality has O > 0.5 AND
  C < -0.5; no arena content.
- **Arena batch (test)**: every case involves `arena` — as an action case at
  arena, or a location case with `arena` among the candidates.

Produce the number of cases the requester asks for (default 50).
````

- [ ] **Step 2: Commit**

```bash
git add docs/rq2b_generation_guide.md
git commit -m "rq2 2b: generation guide for chat-LLM case authoring"
```

---

### Task 2: `common.py` — generalised pool reader, `RunSpec.tag`, input-dict extraction

**Files:**
- Modify: `experiments/rq2/common.py`
- Test: `tests/test_rq2b_pipeline.py` (new file, first tests)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rq2b_pipeline.py`:

```python
"""Study 2B pipeline tests. Plan: docs/plans/2026-07-11-rq2-2b-pipeline.md."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.rq2.common import RunSpec, case_input_dict, read_pool, write_pool
from npc_policy import DEFAULT_CONFIG, IndependentCase, Option, RecentBuffer, compute_relations
from npc_policy.representation import Personality

CODE = Path(__file__).resolve().parents[1]


def _mini_independent_case() -> IndependentCase:
    cands = [Option.location("tavern", social=0.9), Option.location("library", cognitive=0.9)]
    return IndependentCase(
        personality=np.array([0.1, 0.2, 0.3, -0.1, 0.0]),
        decision_type="location",
        candidates=cands,
        recent_locations=[Option.location("market", social=0.5)],
        candidate_history_features=compute_relations(
            cands, _buf([Option.location("market", social=0.5)], 3)),
        target_choice=1,
        source="test", review_status="accepted",
    )


def _buf(options, maxlen):
    b = RecentBuffer(maxlen=maxlen)
    for o in options:
        b.push(o)
    return b


class TestPoolRoundTrip:
    def test_independent_case_round_trip(self, tmp_path):
        case = _mini_independent_case()
        path = tmp_path / "pool.jsonl"
        write_pool(path, [(case, {"id": "c0", "group": "train"})])
        [(back, tags)] = read_pool(path, case_cls=IndependentCase)
        assert tags == {"id": "c0", "group": "train"}
        assert back.target_choice == 1
        assert back.source == "test"
        assert [o.id for o in back.candidates] == ["tavern", "library"]

    def test_default_case_cls_unchanged(self, tmp_path):
        # 2A call sites pass no case_cls and still get ControlledCase
        from npc_policy import ControlledCase
        c = ControlledCase(
            personality=np.zeros(5), decision_type="location",
            candidates=[Option.location("tavern", social=0.9)],
            target_distribution=np.array([1.0]),
        )
        path = tmp_path / "pool.jsonl"
        write_pool(path, [(c, {"id": "x"})])
        [(back, _)] = read_pool(path)
        assert isinstance(back, ControlledCase)


class TestRunSpecTag:
    def test_tag_in_run_id(self):
        assert RunSpec("IND", "nonlinear", 3, tag="wd0.001").run_id == \
            "IND__nonlinear__wd0.001__s3"

    def test_empty_tag_keeps_2a_ids(self):
        assert RunSpec("S0", "simple", 0).run_id == "S0__simple__s0"
        assert RunSpec("G1", "simple", 0, ablation="no_context").run_id == \
            "G1__simple__abl_no_context__s0"


class TestCaseInputDict:
    def test_matches_case_to_inputs_minus_target(self):
        from experiments.rq2.common import case_to_inputs
        from npc_policy import ControlledCase
        cands = [Option.location("tavern", social=0.9),
                 Option.location("library", cognitive=0.9)]
        c = ControlledCase(
            personality=np.array([0.5, 0.0, 0.0, 0.0, 0.0]),
            decision_type="location", candidates=cands,
            target_distribution=np.array([0.7, 0.3]),
        )
        d_full = case_to_inputs(c)
        d_input = case_input_dict(c)
        assert "target" not in d_input
        for k in ("p", "d", "ctx", "cand", "rel"):
            np.testing.assert_array_equal(d_full[k], d_input[k])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rq2b_pipeline.py -q`
Expected: FAIL — `ImportError: cannot import name 'case_input_dict'` (and `read_pool` lacks `case_cls`, `RunSpec` lacks `tag`).

- [ ] **Step 3: Implement in `common.py`**

Three edits.

(a) `read_pool` (replace the existing function):

```python
def read_pool(path: Path, case_cls=ControlledCase) -> list[tuple]:
    """Read a JSONL pool; buffer lists are stored oldest→newest.

    ``case_cls`` is any class with ``from_dict`` (``ControlledCase`` default,
    ``IndependentCase`` for the 2B pool). A record without ``"gen"`` tags is
    corruption and raises ``KeyError``.
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            tags = d.pop("gen")
            records.append((case_cls.from_dict(d), tags))
    return records
```

(b) `RunSpec` — add a `tag` field after `n_train` and weave it into `run_id`:

```python
@dataclass(frozen=True)
class RunSpec:
    """One training run: split × model × seed (+ ablation / data-size / tag)."""

    split: str
    model: str
    seed: int
    ablation: str = "full"
    n_train: int | None = None      # None → the split's full train manifest
    tag: str = ""                   # free-form variant marker (2B weight-decay grid)

    @property
    def run_id(self) -> str:
        parts = [self.split, self.model]
        if self.ablation != "full":
            parts.append(f"abl_{self.ablation}")
        if self.n_train is not None:
            parts.append(f"n{self.n_train}")
        if self.tag:
            parts.append(self.tag)
        parts.append(f"s{self.seed}")
        return "__".join(parts)
```

(c) Split `case_to_inputs`: extract everything before the target line into
`case_input_dict(case, ablation="full")` (same body, same validation, returns
`d` without target); `case_to_inputs` becomes:

```python
def case_to_inputs(case: ControlledCase, ablation: str = "full") -> dict:
    """``ControlledCase`` → inputs dict plus the soft ``"target"`` (2A)."""
    d = case_input_dict(case, ablation)
    d["target"] = np.array(case.target_distribution, dtype=float)   # copy: no aliasing
    return d
```

`case_input_dict` keeps the original docstring (input assembly + invariants) and
the full validation block, ending with `return case_inputs(...)`'s dict (no
`d["target"]` line).

- [ ] **Step 4: Run the new tests and the 2A regression suite**

Run: `python -m pytest tests/test_rq2b_pipeline.py tests/test_rq2_pipeline.py tests/test_learned.py -q`
Expected: all PASS (2A suite proves the refactor changed nothing).

- [ ] **Step 5: Commit**

```bash
git add experiments/rq2/common.py tests/test_rq2b_pipeline.py
git commit -m "rq2 2b: pool reader case_cls, RunSpec.tag, case_input_dict extraction"
```

---

### Task 3: `train.py` — injectable input builder and weight decay

**Files:**
- Modify: `experiments/rq2/train.py:68-90`
- Test: `tests/test_rq2b_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rq2b_pipeline.py`:

```python
class TestTrainOneInjection:
    def test_to_inputs_and_weight_decay_params(self):
        """train_one accepts to_inputs and weight_decay and trains on one-hot targets."""
        from experiments.rq2.train import train_one

        def onehot_inputs(case, ablation="full"):
            d = case_input_dict(case, ablation)
            t = np.zeros(len(case.candidates))
            t[case.target_choice] = 1.0
            d["target"] = t
            return d

        cases = [_mini_independent_case() for _ in range(8)]
        result, state = train_one(
            RunSpec("IND", "simple", 0), cases, cases[:4],
            torch.device("cpu"), to_inputs=onehot_inputs, weight_decay=0.0,
            max_epochs=2, batch_size=4,
        )
        assert result["epochs_run"] <= 2
        assert np.isfinite(result["best_val_kl"])   # one-hot KL == NLL of the choice
        assert any(k == "w" for k in state)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rq2b_pipeline.py::TestTrainOneInjection -q`
Expected: FAIL — `train_one() got an unexpected keyword argument 'to_inputs'`.

- [ ] **Step 3: Modify `train_one`**

In `experiments/rq2/train.py`, change the signature and two lines:

```python
def train_one(
    spec: RunSpec,
    train_cases: list[ControlledCase],
    val_cases: list[ControlledCase],
    device: torch.device,
    *,
    lr: float = 1e-3,
    batch_size: int = 256,
    max_epochs: int = 150,
    patience: int = 10,
    to_inputs=case_to_inputs,
    weight_decay: float | None = None,   # None → 2A rule (1e-4 nonlinear, else 0)
) -> tuple[dict, dict]:
```

Replace the two input-assembly lines:

```python
    inputs = [to_inputs(c, spec.ablation) for c in train_cases]
    val_inputs = [to_inputs(c, spec.ablation) for c in val_cases]
```

Replace the weight-decay line:

```python
    if weight_decay is None:
        weight_decay = 1e-4 if "nonlinear" in spec.model else 0.0
```

(`opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)`
stays.) Hard-label note for the docstring: a one-hot target makes `kl_loss`
compute exactly `-log q[choice]` (cross-entropy), so 2B needs no new loss.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_rq2b_pipeline.py tests/test_rq2_pipeline.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/rq2/train.py tests/test_rq2b_pipeline.py
git commit -m "rq2 2b: train_one accepts to_inputs and weight_decay"
```

---

### Task 4: `independent.py` — parse and validate raw AI batches

**Files:**
- Create: `experiments/rq2/independent.py`
- Test: `tests/test_rq2b_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rq2b_pipeline.py`:

```python
def _raw_case(**over) -> dict:
    base = {
        "personality": {"O": 0.2, "C": -0.1, "E": 0.5, "A": 0.0, "N": -0.3},
        "decision_type": "location",
        "recent_locations": ["market", "tavern"],
        "candidates": ["tavern", "library", "forest"],
        "choice": "library",
        "reason": "quiet after the bustle",
    }
    base.update(over)
    return base


def _raw_action_case(**over) -> dict:
    base = {
        "personality": {"O": -0.5, "C": 0.8, "E": -0.2, "A": 0.4, "N": 0.1},
        "decision_type": "action",
        "selected_location": "library",
        "recent_locations": ["chapel", "library"],
        "recent_actions_same_location": ["read"],
        "candidates": ["read", "research", "discuss"],
        "choice": "read",
    }
    base.update(over)
    return base


class TestValidate:
    def _validate(self, case):
        from experiments.rq2.independent import load_base_world, validate_case
        return validate_case(case, load_base_world())

    def test_valid_location_case(self):
        assert self._validate(_raw_case()) is None

    def test_valid_action_case(self):
        assert self._validate(_raw_action_case()) is None

    @pytest.mark.parametrize("case,reason", [
        (_raw_case(candidates=["tavern", "netbar"]), "unknown_location"),
        (_raw_case(recent_locations=["enemy_camp"]), "unknown_location"),
        (_raw_case(personality={"O": 1.5, "C": 0, "E": 0, "A": 0, "N": 0}), "trait_range"),
        (_raw_case(personality={"O": 0.1}), "traits_missing"),
        (_raw_case(recent_locations=["market"] * 4), "history_too_long"),
        (_raw_case(candidates=["tavern"]), "candidate_count"),
        (_raw_case(candidates=["tavern", "tavern", "library"]), "duplicate_candidates"),
        (_raw_case(choice="arena"), "choice_not_in_candidates"),
        (_raw_case(decision_type="teleport"), "bad_decision_type"),
        (_raw_case(selected_location="tavern"), "location_case_has_selected"),
        (_raw_action_case(candidates=["read", "research"]), "not_full_action_set"),
        (_raw_action_case(recent_locations=["chapel"]), "selected_location_mismatch"),
        (_raw_action_case(recent_actions_same_location=["pray"]), "action_not_native"),
        (_raw_action_case(selected_location=None), "missing_selected_location"),
    ])
    def test_rejections(self, case, reason):
        assert self._validate(case) == reason

    def test_empty_recent_locations_ok_for_action(self):
        # enrich later auto-fills [selected_location]; validation accepts it
        assert self._validate(_raw_action_case(recent_locations=[])) is None


class TestParseRawFile:
    def test_meta_header_and_user_rejection(self, tmp_path):
        from experiments.rq2.independent import parse_raw_file
        payload = [
            {"_meta": {"source": "gpt-5.5"}},
            _raw_case(),
            {**_raw_case(choice="tavern"), "review_status": "rejected"},
        ]
        f = tmp_path / "batch_01.json"
        f.write_text(json.dumps(payload), encoding="utf-8")
        source, cases, user_rejected = parse_raw_file(f)
        assert source == "gpt-5.5"
        assert len(cases) == 1 and len(user_rejected) == 1

    def test_missing_meta_defaults_unknown(self, tmp_path):
        from experiments.rq2.independent import parse_raw_file
        f = tmp_path / "b.json"
        f.write_text(json.dumps([_raw_case()]), encoding="utf-8")
        source, cases, _ = parse_raw_file(f)
        assert source == "unknown" and len(cases) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rq2b_pipeline.py::TestValidate -q`
Expected: FAIL — `ModuleNotFoundError: experiments.rq2.independent`.

- [ ] **Step 3: Create `experiments/rq2/independent.py`**

```python
"""Study 2B independent-dataset library: parse, validate, enrich, model inputs.

Design: ``docs/specs/2026-07-11-rq2-2b-pipeline-design.md``. Raw batches are
names-only JSON authored by a chat LLM (guide: ``docs/rq2b_generation_guide.md``);
this module owns every number: feature vectors come from ``data/world.json`` and
rep/sim/nov from ``npc_policy.relations``. The scorer plays no part here.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from npc_policy import DEFAULT_CONFIG, IndependentCase, RecentBuffer, World, compute_relations, load_world
from npc_policy.representation import Personality

from .common import CODE, DATA, case_input_dict

IND_DATA = DATA / "rq2_independent"
IND_RESULTS = CODE / "results" / "rq2b"
IMPORT_SEED = 20260711

TRAITS = ("O", "C", "E", "A", "N")
_TRAIT_KW = {"O": "openness", "C": "conscientiousness", "E": "extraversion",
             "A": "agreeableness", "N": "neuroticism"}

# split targets (spec §4); general pool scales proportionally to these
GENERAL_TARGETS = {"train": 550, "val": 100, "test_iid": 75}
STRUCT_TARGETS = {"test_pers": 38, "test_arena": 37}
TEST_GROUPS = ("test_iid", "test_pers", "test_arena")

WD_GRID = (1e-4, 1e-3, 1e-2)        # nonlinear-family sweep, chosen on val NLL


@lru_cache(maxsize=1)
def load_base_world() -> World:
    return load_world(DATA / "world.json")


# ------------------------------------------------------------------ validate --
def validate_case(raw: dict, world: World) -> str | None:
    """Reason code for a rule violation (spec §2), or ``None`` if valid."""
    p = raw.get("personality")
    if not isinstance(p, dict) or set(p) != set(TRAITS):
        return "traits_missing"
    if any(not isinstance(v, (int, float)) or not -1.0 <= v <= 1.0
           for v in p.values()):
        return "trait_range"

    dt = raw.get("decision_type")
    if dt not in ("location", "action"):
        return "bad_decision_type"

    unlocked = {o.id for o in world.resolve()}
    recent = raw.get("recent_locations") or []
    if any(name not in unlocked for name in recent):
        return "unknown_location"
    if len(recent) > DEFAULT_CONFIG.K_L:
        return "history_too_long"

    cands = raw.get("candidates") or []
    if len(set(cands)) != len(cands):
        return "duplicate_candidates"

    if dt == "location":
        if raw.get("selected_location") is not None:
            return "location_case_has_selected"
        if raw.get("recent_actions_same_location"):
            return "location_case_has_actions"
        if not 2 <= len(cands) <= len(unlocked):
            return "candidate_count"
        if any(name not in unlocked for name in cands):
            return "unknown_location"
    else:
        sel = raw.get("selected_location")
        if sel not in unlocked:
            return "missing_selected_location"
        if recent and recent[-1] != sel:
            return "selected_location_mismatch"
        native = {a.id for a in world.actions_at(sel)}
        if set(cands) != native:
            return "not_full_action_set"
        acts = raw.get("recent_actions_same_location") or []
        if len(acts) > DEFAULT_CONFIG.K_A:
            return "history_too_long"
        if any(a not in native for a in acts):
            return "action_not_native"

    if raw.get("choice") not in cands:
        return "choice_not_in_candidates"
    return None


# --------------------------------------------------------------------- parse --
def parse_raw_file(path: Path) -> tuple[str, list[dict], list[dict]]:
    """One raw batch file → (source, candidate cases, user-rejected cases).

    The optional ``{"_meta": {"source": …}}`` header names the labelling model;
    elements carrying ``"review_status": "rejected"`` are the user's manual
    rejections (kept for the audit trail, never imported).
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path.name}: expected a JSON array")
    source = "unknown"
    cases, user_rejected = [], []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"{path.name}: non-object array element")
        if "_meta" in item:
            source = str(item["_meta"].get("source", "unknown"))
        elif item.get("review_status") == "rejected":
            user_rejected.append(item)
        else:
            cases.append(item)
    return source, cases, user_rejected


# -------------------------------------------------------------------- enrich --
def _buffer(options, maxlen):
    buf = RecentBuffer(maxlen=maxlen)
    for o in options:                       # oldest → newest (pool convention)
        buf.push(o)
    return buf


def enrich_case(raw: dict, world: World, source: str) -> IndependentCase:
    """Names → numbers: features from the world, relations from the formula.

    Assumes ``validate_case`` returned ``None``. For an action case with empty
    ``recent_locations`` the selected location is auto-filled as the sole entry
    (2A invariant: newest recent location == selected location).
    """
    p = Personality.from_traits(
        **{_TRAIT_KW[k]: float(v) for k, v in raw["personality"].items()})
    recent_names = list(raw.get("recent_locations") or [])
    if raw["decision_type"] == "location":
        cands = [world.effective_location(n) for n in raw["candidates"]]
        history = [world.effective_location(n) for n in recent_names]
        rel = (compute_relations(cands, _buffer(history, DEFAULT_CONFIG.K_L))
               if history else None)
        return IndependentCase(
            personality=p.vector, decision_type="location", candidates=cands,
            recent_locations=history, candidate_history_features=rel,
            target_choice=raw["candidates"].index(raw["choice"]),
            source=source, review_status="accepted", rationale=raw.get("reason"),
        )
    sel = raw["selected_location"]
    if not recent_names:
        recent_names = [sel]
    by_id = {a.id: a for a in world.actions_at(sel)}
    cands = [by_id[n] for n in raw["candidates"]]
    acts = [by_id[n] for n in (raw.get("recent_actions_same_location") or [])]
    rel = (compute_relations(cands, _buffer(acts, DEFAULT_CONFIG.K_A))
           if acts else None)
    return IndependentCase(
        personality=p.vector, decision_type="action", candidates=cands,
        selected_location=sel,
        recent_locations=[world.effective_location(n) for n in recent_names],
        recent_actions_same_location=acts, candidate_history_features=rel,
        target_choice=raw["candidates"].index(raw["choice"]),
        source=source, review_status="accepted", rationale=raw.get("reason"),
    )


# ------------------------------------------------------------- split filters --
def in_pers_region(case: IndependentCase) -> bool:
    """G1 region O > 0.5 ∧ C < −0.5 (same thresholds as 2A, spec §4)."""
    return case.personality[0] > 0.5 and case.personality[1] < -0.5


def touches_arena(case: IndependentCase) -> bool:
    if case.decision_type == "action":
        return case.selected_location == "arena"
    return any(o.id == "arena" for o in case.candidates + case.recent_locations)


def dedupe_key(raw: dict) -> str:
    """Canonical input+label identity for exact-duplicate dropping."""
    return json.dumps({
        "p": {k: round(float(v), 4) for k, v in sorted(raw["personality"].items())},
        "dt": raw["decision_type"],
        "sel": raw.get("selected_location"),
        "rl": raw.get("recent_locations") or [],
        "ra": raw.get("recent_actions_same_location") or [],
        "c": raw["candidates"],
        "y": raw["choice"],
    }, sort_keys=True)


# ------------------------------------------------------- model input assembly --
def independent_case_to_inputs(case: IndependentCase, ablation: str = "full") -> dict:
    """``IndependentCase`` → inputs dict plus a one-hot ``"target"``.

    With a one-hot target, ``kl_loss`` computes exactly ``-log q[choice]``
    (cross-entropy) — 2B trains through the unchanged 2A loop.
    """
    d = case_input_dict(case, ablation)
    t = np.zeros(len(case.candidates), dtype=float)
    t[case.target_choice] = 1.0
    d["target"] = t
    return d
```

Note: `case_input_dict` validates the action-case invariant (newest recent
location == selected location) and reads only fields shared by both case
classes, so it accepts `IndependentCase` unchanged.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_rq2b_pipeline.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/rq2/independent.py tests/test_rq2b_pipeline.py
git commit -m "rq2 2b: raw-batch parse and validate library"
```

---

### Task 5: enrichment parity tests

**Files:**
- Test: `tests/test_rq2b_pipeline.py`

Enrichment code already exists (Task 4); this task pins its numbers against the
`npc_policy` reference implementations.

- [ ] **Step 1: Write the tests**

Append to `tests/test_rq2b_pipeline.py`:

```python
class TestEnrich:
    def test_location_features_match_world(self):
        from experiments.rq2.independent import enrich_case, load_base_world
        w = load_base_world()
        case = enrich_case(_raw_case(), w, source="m1")
        np.testing.assert_array_equal(
            case.candidates[0].features, w.effective_location("tavern").features)
        assert case.target_choice == 1          # "library" at index 1
        assert case.source == "m1" and case.review_status == "accepted"

    def test_relations_match_reference(self):
        from experiments.rq2.independent import enrich_case, load_base_world
        w = load_base_world()
        case = enrich_case(_raw_case(), w, source="m1")
        history = [w.effective_location("market"), w.effective_location("tavern")]
        ref = compute_relations(case.candidates, _buf(history, DEFAULT_CONFIG.K_L))
        np.testing.assert_allclose(case.candidate_history_features.rep, ref.rep)
        np.testing.assert_allclose(case.candidate_history_features.sim, ref.sim)

    def test_empty_history_gives_none_relations(self):
        from experiments.rq2.independent import enrich_case, load_base_world
        case = enrich_case(_raw_case(recent_locations=[]), load_base_world(), "m")
        assert case.candidate_history_features is None

    def test_action_case_autofills_recent_location(self):
        from experiments.rq2.independent import enrich_case, load_base_world
        case = enrich_case(_raw_action_case(recent_locations=[]),
                           load_base_world(), "m")
        assert [o.id for o in case.recent_locations] == ["library"]

    def test_enriched_case_feeds_model_inputs(self):
        from experiments.rq2.independent import (
            enrich_case, independent_case_to_inputs, load_base_world)
        case = enrich_case(_raw_action_case(), load_base_world(), "m")
        d = independent_case_to_inputs(case)
        assert d["target"].tolist() == [1.0, 0.0, 0.0]      # choice "read" = index 0
        assert d["cand"].shape == (3, 12) and d["d"] == 1
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_rq2b_pipeline.py::TestEnrich -q`
Expected: all PASS (if any fail, fix `enrich_case` — the reference is `npc_policy`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_rq2b_pipeline.py
git commit -m "rq2 2b: enrichment parity tests against npc_policy reference"
```

---

### Task 6: `import_independent.py` — split, report, CLI

**Files:**
- Create: `experiments/rq2/import_independent.py`
- Test: `tests/test_rq2b_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rq2b_pipeline.py`:

```python
def _write_raw_dir(tmp_path, n_general=40, n_pers=6, n_arena=6) -> Path:
    """Synthesise a small raw/ directory covering all three batch types."""
    import random
    rng = random.Random(0)
    locs = ["tavern", "library", "chapel", "market", "forest"]

    def general(i):
        pool = rng.sample(locs, 3)
        return _raw_case(
            personality={"O": rng.uniform(-1, 0.5), "C": rng.uniform(-0.5, 1),
                         "E": rng.uniform(-1, 1), "A": 0.0, "N": 0.0},
            recent_locations=rng.sample(locs, rng.randint(0, 3)),
            candidates=pool, choice=pool[i % 3])

    def pers(i):
        pool = rng.sample(locs, 3)
        return _raw_case(
            personality={"O": 0.8, "C": -0.8, "E": 0.1, "A": 0.0, "N": 0.0},
            candidates=pool, choice=pool[i % 3])

    def arena(i):
        pool = ["arena"] + rng.sample(locs, 2)
        return _raw_case(candidates=pool, choice="arena")

    raw = tmp_path / "raw"
    raw.mkdir()
    for name, gen, n in [("general.json", general, n_general),
                         ("pers.json", pers, n_pers),
                         ("arena.json", arena, n_arena)]:
        payload = [{"_meta": {"source": "test-llm"}}] + [gen(i) for i in range(n)]
        (raw / name).write_text(json.dumps(payload), encoding="utf-8")
    return raw


class TestImport:
    def test_end_to_end_import(self, tmp_path):
        from experiments.rq2.import_independent import run_import
        raw = _write_raw_dir(tmp_path)
        meta = run_import(raw_dir=raw, out_dir=tmp_path / "out")
        splits = json.loads((tmp_path / "out" / "splits.json").read_text(encoding="utf-8"))
        # proportional scaling: general pool of 40 → 550:100:75 ratios
        n = sum(len(splits[k]) for k in ("train", "val", "test_iid"))
        assert n == 40
        assert len(splits["test_iid"]) == round(40 * 75 / 725)
        assert len(splits["test_pers"]) == 6 and len(splits["test_arena"]) == 6
        assert (tmp_path / "out" / "report.txt").exists()
        assert meta["accepted"] == 52

    def test_isolation_no_structured_content_in_train_val(self, tmp_path):
        from experiments.rq2.import_independent import run_import
        from experiments.rq2.independent import in_pers_region, touches_arena
        from npc_policy import IndependentCase
        raw = _write_raw_dir(tmp_path)
        run_import(raw_dir=raw, out_dir=tmp_path / "out")
        pool = read_pool(tmp_path / "out" / "cases.jsonl", case_cls=IndependentCase)
        splits = json.loads((tmp_path / "out" / "splits.json").read_text(encoding="utf-8"))
        trainval = set(splits["train"]) | set(splits["val"])
        for case, tags in pool:
            if tags["id"] in trainval:
                assert not in_pers_region(case) and not touches_arena(case)

    def test_rejects_recorded_and_duplicates_dropped(self, tmp_path):
        from experiments.rq2.import_independent import run_import
        raw = tmp_path / "raw"
        raw.mkdir()
        bad = _raw_case(choice="arena")                  # invalid: not a candidate
        dup = _raw_case()
        payload = [{"_meta": {"source": "m"}}, dup, dup, bad,
                   {**_raw_case(candidates=["tavern", "library"], choice="tavern"),
                    "review_status": "rejected"}]
        (raw / "b.json").write_text(json.dumps(payload), encoding="utf-8")
        meta = run_import(raw_dir=raw, out_dir=tmp_path / "out")
        rejected = [json.loads(l) for l in
                    (tmp_path / "out" / "rejected.jsonl").read_text(encoding="utf-8").splitlines()]
        reasons = sorted(r["reason"] for r in rejected)
        assert reasons == ["choice_not_in_candidates", "duplicate", "user_rejected"]
        assert meta["accepted"] == 1

    def test_deterministic_splits(self, tmp_path):
        from experiments.rq2.import_independent import run_import
        raw = _write_raw_dir(tmp_path)
        run_import(raw_dir=raw, out_dir=tmp_path / "o1")
        run_import(raw_dir=raw, out_dir=tmp_path / "o2")
        s1 = (tmp_path / "o1" / "splits.json").read_text(encoding="utf-8")
        s2 = (tmp_path / "o2" / "splits.json").read_text(encoding="utf-8")
        assert s1 == s2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rq2b_pipeline.py::TestImport -q`
Expected: FAIL — `ModuleNotFoundError: experiments.rq2.import_independent`.

- [ ] **Step 3: Create `experiments/rq2/import_independent.py`**

```python
"""Study 2B import: raw AI batches → validated, enriched, split dataset.

Rebuilds everything from ``data/rq2_independent/raw/*.json`` on every run
(idempotent; add files and re-run). Outputs ``cases.jsonl`` (pool format with
``gen`` tags), ``rejected.jsonl``, ``splits.json``, ``meta.json``, ``report.txt``.

Run from ``code/``:  python -m experiments.rq2.import_independent
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from npc_policy import IndependentCase

from .common import config_hash, write_pool
from .independent import (
    GENERAL_TARGETS,
    IMPORT_SEED,
    IND_DATA,
    STRUCT_TARGETS,
    TRAITS,
    dedupe_key,
    enrich_case,
    in_pers_region,
    load_base_world,
    parse_raw_file,
    touches_arena,
    validate_case,
)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _assign_splits(records: list[tuple[IndependentCase, dict]],
                   rng: np.random.Generator) -> dict[str, list[str]]:
    """Spec §4: structured filters first (isolation enforced here, not by the
    AI following instructions), then proportional general-pool splitting."""
    pers = [t["id"] for c, t in records if in_pers_region(c)]
    arena = [t["id"] for c, t in records
             if touches_arena(c) and not in_pers_region(c)]
    general = [t["id"] for c, t in records
               if not in_pers_region(c) and not touches_arena(c)]
    for ids in (pers, arena, general):
        rng.shuffle(ids)

    splits = {"test_pers": pers[: STRUCT_TARGETS["test_pers"]],
              "test_arena": arena[: STRUCT_TARGETS["test_arena"]]}
    # structured surplus is dropped (never train/val — isolation), recorded in meta
    splits["dropped_structured"] = (pers[STRUCT_TARGETS["test_pers"]:]
                                    + arena[STRUCT_TARGETS["test_arena"]:])

    total = sum(GENERAL_TARGETS.values())               # 725
    n = len(general)
    n_iid = round(n * GENERAL_TARGETS["test_iid"] / total)
    n_val = round(n * GENERAL_TARGETS["val"] / total)
    splits["test_iid"] = general[:n_iid]
    splits["val"] = general[n_iid: n_iid + n_val]
    splits["train"] = general[n_iid + n_val:]
    return splits


def _coverage(records: list[tuple[IndependentCase, dict]]) -> list[str]:
    lines = []
    dts = Counter(c.decision_type for c, _ in records)
    lines.append(f"decision types: {dict(dts)}")
    for i, name in enumerate(TRAITS):
        vals = np.array([c.personality[i] for c, _ in records])
        lines.append(f"trait {name}: high(>0.3) {int((vals > 0.3).sum())}, "
                     f"mid {int(((vals >= -0.3) & (vals <= 0.3)).sum())}, "
                     f"low(<-0.3) {int((vals < -0.3).sum())}")
    empty = sum(1 for c, _ in records if c.candidate_history_features is None)
    lines.append(f"empty-history cases: {empty} ({empty / max(len(records), 1):.0%})")
    return lines


def run_import(raw_dir: Path = IND_DATA / "raw",
               out_dir: Path = IND_DATA) -> dict:
    world = load_base_world()
    files = sorted(raw_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no raw batches in {raw_dir}")

    records: list[tuple[IndependentCase, dict]] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    for f in files:
        source, cases, user_rejected = parse_raw_file(f)
        for item in user_rejected:
            rejected.append({"file": f.name, "reason": "user_rejected", "case": item})
        for idx, raw in enumerate(cases):
            reason = validate_case(raw, world)
            if reason is None:
                key = dedupe_key(raw)
                if key in seen:
                    reason = "duplicate"
                seen.add(key)
            if reason is not None:
                rejected.append({"file": f.name, "reason": reason, "case": raw})
                continue
            case = enrich_case(raw, world, source)
            records.append((case, {"id": f"{f.stem}#{idx}", "source_file": f.name}))

    rng = np.random.default_rng(IMPORT_SEED)
    splits = _assign_splits(records, rng)
    group_of = {cid: g for g, ids in splits.items() for cid in ids}
    kept = []
    for case, tags in records:
        g = group_of.get(tags["id"])
        if g == "dropped_structured":
            rejected.append({"file": tags["source_file"], "reason": "structured_surplus",
                             "case": {"id": tags["id"]}})
            continue
        case.split = ("test" if g in ("test_iid", "test_pers", "test_arena") else g)
        kept.append((case, {**tags, "group": g}))

    out_dir.mkdir(parents=True, exist_ok=True)
    write_pool(out_dir / "cases.jsonl", kept)
    _atomic_write(out_dir / "rejected.jsonl",
                  "".join(json.dumps(r) + "\n" for r in rejected))
    public_splits = {k: v for k, v in splits.items() if k != "dropped_structured"}
    _atomic_write(out_dir / "splits.json",
                  json.dumps({"splits": public_splits}, indent=2))

    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": IMPORT_SEED,
        "config_hash": config_hash(),
        "world_hash": hashlib.sha256(
            (Path(__file__).resolve().parents[2] / "data" / "world.json")
            .read_bytes()).hexdigest(),
        "raw_files": [f.name for f in files],
        "accepted": len(kept),
        "rejected": len(rejected),
        "split_sizes": {k: len(v) for k, v in public_splits.items()},
    }
    _atomic_write(out_dir / "meta.json", json.dumps(meta, indent=2))

    # report: aggregates only — test-case details stay blind (spec §4)
    trainval = [(c, t) for c, t in kept if t["group"] in ("train", "val")]
    lines = [f"accepted {len(kept)} / raw {len(kept) + len(rejected)}",
             "rejections: " + json.dumps(Counter(r["reason"] for r in rejected)),
             "split sizes: " + json.dumps(meta["split_sizes"]), "",
             "-- coverage (train+val pool only) --", *_coverage(trainval), ""]
    for grp, target in [("test_pers", STRUCT_TARGETS["test_pers"]),
                        ("test_arena", STRUCT_TARGETS["test_arena"]),
                        ("test_iid", GENERAL_TARGETS["test_iid"])]:
        got = meta["split_sizes"].get(grp, 0)
        if got < target:
            kind = {"test_pers": "personality-batch", "test_arena": "arena-batch",
                    "test_iid": "general-batch"}[grp]
            lines.append(f"SHORTFALL {grp}: {got}/{target} — request "
                         f"{target - got} more {kind} cases")
    _atomic_write(out_dir / "report.txt", "\n".join(lines) + "\n")
    print(f"imported {len(kept)} cases → {out_dir}; see report.txt")
    return meta


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Import Study 2B raw AI batches")
    ap.add_argument("--raw", type=Path, default=IND_DATA / "raw")
    ap.add_argument("--out", type=Path, default=IND_DATA)
    args = ap.parse_args(argv)
    run_import(raw_dir=args.raw, out_dir=args.out)


if __name__ == "__main__":
    main()
```

Note: `case.split = …` works because `IndependentCase` is a plain (non-frozen)
dataclass with a `split` field.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_rq2b_pipeline.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/rq2/import_independent.py tests/test_rq2b_pipeline.py
git commit -m "rq2 2b: import script - validate, enrich, split, report"
```

---

### Task 7: `train_2b.py` — 40-run matrix over the imported dataset

**Files:**
- Create: `experiments/rq2/train_2b.py`
- Test: `tests/test_rq2b_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rq2b_pipeline.py`:

```python
class TestTrain2B:
    def test_run_matrix_shape(self):
        from experiments.rq2.train_2b import run_matrix_2b
        specs = run_matrix_2b()
        assert len(specs) == 40                     # 2×5 simple + 2×5×3 nonlinear
        assert len({s.run_id for s in specs}) == 40
        assert all(s.split == "IND" for s in specs)

    def test_wd_of_spec(self):
        from experiments.rq2.train_2b import run_matrix_2b, wd_of
        for s in run_matrix_2b():
            if "nonlinear" in s.model:
                assert wd_of(s) in (1e-4, 1e-3, 1e-2) and s.tag.startswith("wd")
            else:
                assert wd_of(s) == 0.0 and s.tag == ""

    def test_end_to_end_smoke(self, tmp_path):
        """import → train 2 runs, 2 epochs → resumable artefacts on disk."""
        from experiments.rq2.import_independent import run_import
        from experiments.rq2.train_2b import train_main
        raw = _write_raw_dir(tmp_path, n_general=30, n_pers=4, n_arena=4)
        run_import(raw_dir=raw, out_dir=tmp_path / "data")
        train_main(["--data", str(tmp_path / "data"),
                    "--results", str(tmp_path / "res"),
                    "--only", "IND__simple", "--max-epochs", "2"])
        runs = list((tmp_path / "res" / "runs").glob("*.json"))
        assert len(runs) == 5                       # simple × 5 seeds
        meta = json.loads(runs[0].read_text(encoding="utf-8"))
        assert np.isfinite(meta["best_val_kl"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rq2b_pipeline.py::TestTrain2B -q`
Expected: FAIL — `ModuleNotFoundError: experiments.rq2.train_2b`.

- [ ] **Step 3: Create `experiments/rq2/train_2b.py`**

```python
"""Study 2B training — hard labels through the unchanged 2A loop.

One-hot targets make ``kl_loss`` equal cross-entropy, so ``train.train_one``
runs as-is; ``best_val_kl`` in the run results IS the validation NLL. The
nonlinear families sweep weight decay (chosen on val NLL at evaluation time);
the simple families train without weight decay.

Run from ``code/``:  python -m experiments.rq2.train_2b [--device auto|cpu|cuda]
                                                        [--only PREFIX]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from npc_policy import IndependentCase

from .common import SEEDS, RunSpec, config_hash, read_pool
from .independent import IND_DATA, IND_RESULTS, WD_GRID, independent_case_to_inputs
from .train import pick_device, run_all, train_one

SIMPLE_MODELS = ("simple", "agnostic_simple")
NONLINEAR_MODELS = ("nonlinear", "agnostic_nonlinear")


def run_matrix_2b() -> list[RunSpec]:
    runs = [RunSpec("IND", m, s) for m in SIMPLE_MODELS for s in SEEDS]        # 10
    runs += [RunSpec("IND", m, s, tag=f"wd{wd:g}")
             for m in NONLINEAR_MODELS for wd in WD_GRID for s in SEEDS]       # 30
    return runs


def wd_of(spec: RunSpec) -> float:
    return float(spec.tag[2:]) if spec.tag.startswith("wd") else 0.0


def train_2b_one(spec, train_cases, val_cases, device, **kw):
    return train_one(spec, train_cases, val_cases, device,
                     to_inputs=independent_case_to_inputs,
                     weight_decay=wd_of(spec), **kw)


def make_loader_2b(data_dir: Path):
    state: dict = {}

    def load_cases(spec: RunSpec):
        if "by_id" not in state:
            state["by_id"] = {t["id"]: c for c, t in
                              read_pool(data_dir / "cases.jsonl",
                                        case_cls=IndependentCase)}
            state["splits"] = json.loads(
                (data_dir / "splits.json").read_text(encoding="utf-8"))["splits"]
        by_id, splits = state["by_id"], state["splits"]
        return ([by_id[i] for i in splits["train"]],
                [by_id[i] for i in splits["val"]])

    return load_cases


def train_main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train all Study 2B runs (resumable)")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--data", type=Path, default=IND_DATA)
    ap.add_argument("--results", type=Path, default=IND_RESULTS)
    ap.add_argument("--max-epochs", type=int, default=500)
    args = ap.parse_args(argv)
    if not (args.data / "cases.jsonl").exists():
        raise SystemExit(f"dataset missing: {args.data} - run import_independent first")
    meta = json.loads((args.data / "meta.json").read_text(encoding="utf-8"))
    if config_hash() != meta["config_hash"]:
        raise SystemExit("relation config drifted since import; re-run "
                         "import_independent or restore the config")
    specs = run_matrix_2b()
    if args.only:
        specs = [s for s in specs if s.run_id.startswith(args.only)]
    run_all(specs, make_loader_2b(args.data), args.results, pick_device(args.device),
            train_fn=train_2b_one, batch_size=64,
            max_epochs=args.max_epochs, patience=30)
    print(f"done: {args.results / 'runs'}")


if __name__ == "__main__":
    train_main()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_rq2b_pipeline.py -q`
Expected: all PASS (smoke test trains 5 tiny runs on CPU, ~seconds).

- [ ] **Step 5: Commit**

```bash
git add experiments/rq2/train_2b.py tests/test_rq2b_pipeline.py
git commit -m "rq2 2b: training over the independent dataset (40-run matrix)"
```

---

### Task 8: `run_2b.py` — comparative evaluation (Study 3 automated table)

**Files:**
- Create: `experiments/rq2/run_2b.py`
- Test: `tests/test_rq2b_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rq2b_pipeline.py`:

```python
class TestRun2B:
    def test_scorer_and_uniform_rows(self, tmp_path):
        """Full mini-pipeline: import → train simple → evaluate all systems."""
        import csv
        from experiments.rq2.import_independent import run_import
        from experiments.rq2.run_2b import eval_main
        from experiments.rq2.train_2b import train_main
        raw = _write_raw_dir(tmp_path, n_general=30, n_pers=4, n_arena=4)
        run_import(raw_dir=raw, out_dir=tmp_path / "data")
        train_main(["--data", str(tmp_path / "data"),
                    "--results", str(tmp_path / "res"),
                    "--only", "IND__simple", "--max-epochs", "2"])
        eval_main(["--data", str(tmp_path / "data"),
                   "--results", str(tmp_path / "res")])
        with open(tmp_path / "res" / "main_table.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        systems = {r["system"] for r in rows}
        assert {"uniform", "scorer", "simple"} <= systems
        groups = {r["group"] for r in rows}
        assert {"all", "test_iid", "test_pers", "test_arena"} <= groups
        for r in rows:
            assert 0.0 <= float(r["top1_mean"]) <= 1.0
            assert float(r["nll_mean"]) >= 0.0
        assert (tmp_path / "res" / "group_bars.png").exists()

    def test_scorer_predictions_scored_like_models(self):
        """Scorer NLL/top-1 computed from its distribution on the labelled choice."""
        from experiments.rq2.run_2b import scorer_probs
        from experiments.rq2.independent import enrich_case, load_base_world
        case = enrich_case(_raw_case(), load_base_world(), "m")
        q = scorer_probs(case)
        assert q.shape == (3,) and abs(q.sum() - 1.0) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rq2b_pipeline.py::TestRun2B -q`
Expected: FAIL — `ModuleNotFoundError: experiments.rq2.run_2b`.

- [ ] **Step 3: Create `experiments/rq2/run_2b.py`**

```python
"""Study 2B / Study 3 automated comparison on the independent test set.

Every system predicts the same 150 test cases; metrics are top-1 accuracy and
NLL of the labelled choice. Learned families aggregate over seeds (mean ± sd);
per (family, seed) the nonlinear weight-decay variant with the best val NLL is
selected. The hand-authored scorer appears here strictly as an evaluated
system — it took no part in labels or training.

Run from ``code/``:  python -m experiments.rq2.run_2b
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.rq1.common import setup_style, write_csv
from npc_policy import HandAuthoredScorer, IndependentCase
from npc_policy.learned import PolicyBatch, UniformBaseline
from npc_policy.representation import Personality

from .common import SEEDS, read_pool
from .independent import IND_DATA, IND_RESULTS, TEST_GROUPS, independent_case_to_inputs
from .run_2a import load_student
from .train_2b import NONLINEAR_MODELS, SIMPLE_MODELS

SYSTEM_COLORS = {"uniform": "#CCCCCC", "agnostic_simple": "#999999",
                 "agnostic_nonlinear": "#CC79A7", "scorer": "#009E73",
                 "simple": "#0072B2", "nonlinear": "#E69F00"}
_TINY = np.finfo(float).tiny


def scorer_probs(case: IndependentCase) -> np.ndarray:
    scorer = HandAuthoredScorer()
    return scorer.distribution(
        Personality(np.array(case.personality, dtype=float)),
        case.candidates, relations=case.candidate_history_features,
        level=case.decision_type)


def model_probs(model: torch.nn.Module, cases: list[IndependentCase],
                chunk: int = 512) -> list[np.ndarray]:
    out = []
    model.eval()
    for lo in range(0, len(cases), chunk):
        part = cases[lo:lo + chunk]
        batch = PolicyBatch.from_cases(
            [independent_case_to_inputs(c) for c in part])
        with torch.no_grad():
            probs = model(batch).exp().numpy()
        out.extend(probs[i, : len(c.candidates)] for i, c in enumerate(part))
    return out


def case_metrics(q: np.ndarray, case: IndependentCase) -> dict:
    y = case.target_choice
    return {"top1": int(int(np.argmax(q)) == y),
            "nll": float(-np.log(max(q[y], _TINY))),
            "decision_type": case.decision_type}


def best_nonlinear_runs(results_dir: Path, family: str) -> list[str]:
    """Per seed, the weight-decay variant with the lowest val NLL."""
    best: dict[int, tuple[float, str]] = {}
    for f in (results_dir / "runs").glob(f"IND__{family}__wd*.json"):
        meta = json.loads(f.read_text(encoding="utf-8"))
        seed, val = meta["seed"], meta["best_val_kl"]
        if seed not in best or val < best[seed][0]:
            best[seed] = (val, meta["run_id"])
    return [run_id for _, run_id in sorted(best.values(), key=lambda t: t[1])]


def _rows_for(system: str, seed, per_case: list[dict],
              groups: dict[str, str], ids: list[str]) -> list[dict]:
    """Aggregate one prediction set to (group × decision_type) means."""
    rows = []
    for grp in ("all",) + TEST_GROUPS:
        sel = [m for m, cid in zip(per_case, ids)
               if grp == "all" or groups[cid] == grp]
        if not sel:
            continue
        for dt in ("location", "action", "all"):
            part = sel if dt == "all" else [m for m in sel if m["decision_type"] == dt]
            if not part:
                continue
            rows.append({"system": system, "seed": seed, "group": grp,
                         "decision_type": dt, "n_cases": len(part),
                         "top1": float(np.mean([m["top1"] for m in part])),
                         "nll": float(np.mean([m["nll"] for m in part]))})
    return rows


def _std(xs):
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def eval_main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Study 2B comparative evaluation")
    ap.add_argument("--data", type=Path, default=IND_DATA)
    ap.add_argument("--results", type=Path, default=IND_RESULTS)
    args = ap.parse_args(argv)
    setup_style()

    pool = read_pool(args.data / "cases.jsonl", case_cls=IndependentCase)
    by_id = {t["id"]: c for c, t in pool}
    groups = {t["id"]: t["group"] for _, t in pool}
    ids = [t["id"] for _, t in pool if t["group"] in TEST_GROUPS]
    cases = [by_id[i] for i in ids]
    if not cases:
        raise SystemExit("no test cases in the pool - run import_independent first")

    per_seed_rows: list[dict] = []
    # fixed systems (no seeds)
    uni = [case_metrics(q, c) for q, c in
           zip(model_probs(UniformBaseline(), cases), cases)]
    per_seed_rows += _rows_for("uniform", 0, uni, groups, ids)
    sco = [case_metrics(scorer_probs(c), c) for c in cases]
    per_seed_rows += _rows_for("scorer", 0, sco, groups, ids)
    # learned systems
    run_ids = {m: [f"IND__{m}__s{s}" for s in SEEDS] for m in SIMPLE_MODELS}
    run_ids |= {m: best_nonlinear_runs(args.results, m) for m in NONLINEAR_MODELS}
    for system, rids in run_ids.items():
        for rid in rids:
            if not (args.results / "runs" / f"{rid}.json").exists():
                continue
            model = load_student(args.results, rid)
            per = [case_metrics(q, c) for q, c in
                   zip(model_probs(model, cases), cases)]
            seed = json.loads((args.results / "runs" / f"{rid}.json")
                              .read_text(encoding="utf-8"))["seed"]
            per_seed_rows += _rows_for(system, seed, per, groups, ids)

    # aggregate over seeds
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in per_seed_rows:
        grouped[(r["system"], r["group"], r["decision_type"])].append(r)
    table = [{"system": sys_, "group": grp, "decision_type": dt,
              "n_cases": rs[0]["n_cases"], "n_seeds": len(rs),
              "top1_mean": float(np.mean([r["top1"] for r in rs])),
              "top1_std": _std([r["top1"] for r in rs]),
              "nll_mean": float(np.mean([r["nll"] for r in rs])),
              "nll_std": _std([r["nll"] for r in rs])}
             for (sys_, grp, dt), rs in sorted(grouped.items())]
    write_csv(args.results / "main_table.csv", list(table[0].keys()),
              [list(r.values()) for r in table])

    # per-case diagnostics for error analysis
    diag = [[sys_, r["group"], r["decision_type"], r["n_cases"],
             round(r["top1_mean"], 4), round(r["nll_mean"], 4)]
            for r in table for sys_ in [r["system"]]]
    write_csv(args.results / "diagnostics.csv",
              ["system", "group", "decision_type", "n_cases", "top1", "nll"], diag)

    # figure: top-1 per system × group
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plot_groups = ("all",) + TEST_GROUPS
    x = np.arange(len(plot_groups))
    systems = [s for s in SYSTEM_COLORS if any(r["system"] == s for r in table)]
    for i, sys_ in enumerate(systems):
        ys, es = [], []
        for grp in plot_groups:
            rows = [r for r in table if r["system"] == sys_ and r["group"] == grp
                    and r["decision_type"] == "all"]
            ys.append(rows[0]["top1_mean"] if rows else np.nan)
            es.append(rows[0]["top1_std"] if rows else 0.0)
        off = (i - (len(systems) - 1) / 2) * 0.8 / len(systems)
        ax.bar(x + off, ys, width=0.75 / len(systems), yerr=es,
               color=SYSTEM_COLORS[sys_], label=sys_, capsize=2)
    ax.set_xticks(x, plot_groups)
    ax.set_ylabel("top-1 accuracy")
    ax.set_title("2B — agreement with independent labels per test group")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(args.results / "group_bars.png", bbox_inches="tight")
    plt.close(fig)
    print(f"written: {args.results / 'main_table.csv'}, group_bars.png, diagnostics.csv")


if __name__ == "__main__":
    eval_main()
```

- [ ] **Step 4: Run the full 2B suite plus 2A regression**

Run: `python -m pytest tests/test_rq2b_pipeline.py tests/test_rq2_pipeline.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/rq2/run_2b.py tests/test_rq2b_pipeline.py
git commit -m "rq2 2b: comparative evaluation - main table, group bars"
```

---

### Task 9: runbook section and dev-log entry

**Files:**
- Modify: `docs/rq2_runbook.md` (append)
- Modify: `docs/dev_log.md` (append)

- [ ] **Step 1: Append the Chinese 2B runbook section to `docs/rq2_runbook.md`**

````markdown
## Study 2B 流程（独立数据集）

前置：2A 无需完成；2B 全程零 API 成本，本地分钟级。

1. **生成数据（手工）**：把 `docs/rq2b_generation_guide.md` 全文贴给
   GPT/Claude，按批索要（"General batch, 50 cases"）。三种批次都要：
   General ~1000 条、Personality batch ~60 条、Arena batch ~60 条。
   每批回复存成一个文件放进 `data/rq2_independent/raw/`（如 `general_01.json`）。
   审核：删掉离谱的 case，或给它加 `"review_status": "rejected"`。
2. **导入**：`python -m experiments.rq2.import_independent`
   看 `data/rq2_independent/report.txt`——有 SHORTFALL 行就按提示再要数据，
   重跑导入（全量重建，随时可重跑）。
3. **训练**：`python -m experiments.rq2.train_2b`
   40 个 run，CPU 上约几分钟；中断后重跑同命令自动续。
4. **评估**：`python -m experiments.rq2.run_2b`
   产出 `results/rq2b/main_table.csv`（论文 2B 主表）、`group_bars.png`、
   `diagnostics.csv`。

校验：任何时候 `python -m pytest tests/test_rq2b_pipeline.py -q` 全绿说明管线本身没问题。
````

- [ ] **Step 2: Append a dated entry to `docs/dev_log.md`** summarising: 2B pipeline implemented per `docs/specs/2026-07-11-rq2-2b-pipeline-design.md`; chat-manual generation route; one-hot-target reuse of the 2A loop; isolation enforced by the import script.

- [ ] **Step 3: Run the whole test suite once**

Run: `python -m pytest -q`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add docs/rq2_runbook.md docs/dev_log.md
git commit -m "rq2 2b: runbook section and dev log"
```

---

## Self-review notes

- **Spec coverage:** guide (spec §2 quotas/vocabulary → Task 1), import validate/enrich/split/report (§3 → Tasks 4–6), test structure 75/38/37 with import-enforced isolation (§4 → Task 6), training via one-hot CE with wd sweep (§5 → Tasks 3, 7), comparative evaluation incl. scorer + uniform with group × decision-type breakdown (§6 → Task 8), tests (§7 → every task, TDD). Provenance (`source`, `review_status`, `rejected.jsonl`) in Tasks 4/6.
- **Deviation from spec §7:** the "checked-in 12-case raw fixture" is realised as the in-test generator `_write_raw_dir` (self-contained, no binary fixture file); coverage is equivalent.
- **Type consistency:** `RunSpec.tag` (Task 2) ↔ `run_matrix_2b`/`wd_of` (Task 7); `case_input_dict` (Task 2) ↔ `independent_case_to_inputs` (Task 4); `read_pool(case_cls=…)` (Task 2) ↔ loaders (Tasks 6–8); `load_student` reused from `run_2a` unchanged.
- **Verify at implementation time** (interfaces read 2026-07-11, double-check when touching): `Personality.from_traits` kwargs, `RecentBuffer(maxlen=…).push`, `scorer.distribution(p, cands, relations=…, level=…)`, `IndependentCase` field names, `write_csv`/`setup_style` in `experiments/rq1/common.py`.
