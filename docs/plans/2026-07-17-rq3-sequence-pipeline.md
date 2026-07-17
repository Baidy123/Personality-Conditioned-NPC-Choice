# RQ3 Sequence Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let one `DecisionController` drive all four policy variants and batch-generate replayable behaviour-sequence JSON files (the Unity playback input) with a manifest and text preview.

**Architecture:** A `LearnedPolicyAdapter` in `npc_policy/policies.py` gives trained checkpoints the same `distribution(...)` shape the scorer already has; `DecisionController` gains a `policy` argument and passes selected-location context; `experiments/rq3/` crosses config entries into validated sequence files. Spec: `docs/specs/2026-07-17-rq3-sequence-interface-design.md`.

**Tech Stack:** Python 3.11, numpy, torch (already project deps). All commands run from `code/`.

**Existing interfaces this plan builds on (verified 2026-07-17):**

- `HandAuthoredScorer.trace(personality, candidates, buffer=None, relations=None, level="location") -> ScoreTrace` and `.distribution(...)` returning `trace.P_rule` (`npc_policy/scorer.py:154-212`).
- `predict_distribution(model, personality, candidates, decision_type, relations=None, selected_location=None) -> np.ndarray` (`npc_policy/learned.py:130`). Raises `ValueError` if an action case lacks `selected_location` (via `features.case_inputs`).
- Checkpoint payload format (written by `experiments/rq2/train.py:179`): `{"model": <arch name str>, "state_dict": ...}`; arch names `simple | nonlinear | agnostic_simple | agnostic_nonlinear` (see `experiments/rq2/common.py:195 build_model`).
- `compute_relations(candidates, buffer, config) -> Relations` (`npc_policy/relations.py:65`); empty buffer → all-zero relations.
- `World.resolve()`, `World.actions_at(id)`, `load_world(path)` (`npc_policy/world.py`); demo content at `data/world.json`.
- `DecisionController(scorer, config=..., mode=..., rng=..., ...)` — every existing caller passes the scorer **positionally**, and nothing outside the class reads `ctrl.scorer` or uses the `Decision.distribution` property, so renaming the parameter/attribute is safe (grep-verified).

## File Structure

```
npc_policy/policies.py              create — build_architecture(), LearnedPolicyAdapter
npc_policy/controller.py            modify — policy arg, Decision.distribution field, ctx pass-through
experiments/rq3/__init__.py         create — empty
experiments/rq3/common.py           create — SequenceSpec, generate_sequence, validate_sequence, format_preview
experiments/rq3/gen_sequences.py    create — CLI: config → files + manifest.csv (+ --preview)
experiments/rq3/config_smoke.json   create — scorer-only smoke config on data/world.json
tests/test_policies.py              create — adapter + controller tests
tests/test_rq3_pipeline.py          create — generation + CLI tests
```

---

### Task 1: LearnedPolicyAdapter (`npc_policy/policies.py`)

**Files:**
- Create: `npc_policy/policies.py`
- Test: `tests/test_policies.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_policies.py`:

```python
"""Tests for npc_policy.policies (unified policy layer) and the generalised
DecisionController (Task 2 adds the controller tests to this file)."""

import numpy as np
import pytest
import torch

from npc_policy.config import DEFAULT_CONFIG
from npc_policy.learned import NonlinearPolicy, predict_distribution
from npc_policy.policies import LearnedPolicyAdapter, build_architecture
from npc_policy.relations import compute_relations
from npc_policy.representation import Option, Personality, RecentBuffer

LOCS = [
    Option.location("tavern", social=0.9, stimulation=0.7),
    Option.location("library", cognitive=0.9, privacy=0.8),
    Option.location("forge", physical=0.8, structure=0.7),
]
ACTS = [
    Option.action("chat", social=0.9),
    Option.action("drink", stimulation=0.6),
]
P = Personality.from_traits(extraversion=0.8, openness=0.4)


def save_checkpoint(tmp_path, name, seed=0):
    torch.manual_seed(seed)
    model = build_architecture(name)
    path = tmp_path / f"{name}.pt"
    torch.save({"model": name, "state_dict": model.state_dict()}, path)
    return path, model


def test_build_architecture_unknown_name():
    with pytest.raises(ValueError, match="unknown architecture"):
        build_architecture("transformer")


def test_adapter_all_architectures_valid_distribution(tmp_path):
    for name in ("simple", "nonlinear", "agnostic_simple", "agnostic_nonlinear"):
        path, _ = save_checkpoint(tmp_path, name)
        adapter = LearnedPolicyAdapter(path)
        dist = adapter.distribution(P, LOCS, level="location")
        assert dist.shape == (3,)
        assert abs(dist.sum() - 1.0) < 1e-9
        assert (dist >= 0).all()


def test_adapter_matches_predict_distribution_with_buffer(tmp_path):
    path, model = save_checkpoint(tmp_path, "nonlinear")
    adapter = LearnedPolicyAdapter(path)
    buf = RecentBuffer(maxlen=3)
    buf.push(LOCS[0])
    buf.push(LOCS[1])
    got = adapter.distribution(P, LOCS, buffer=buf, level="location")
    rel = compute_relations(LOCS, buf, DEFAULT_CONFIG)
    want = predict_distribution(model, P, LOCS, "location", relations=rel)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_adapter_empty_buffer_equals_no_relations(tmp_path):
    path, model = save_checkpoint(tmp_path, "nonlinear")
    adapter = LearnedPolicyAdapter(path)
    got = adapter.distribution(P, LOCS, buffer=RecentBuffer(maxlen=3), level="location")
    want = predict_distribution(model, P, LOCS, "location", relations=None)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_adapter_action_requires_selected_location(tmp_path):
    path, _ = save_checkpoint(tmp_path, "nonlinear")
    adapter = LearnedPolicyAdapter(path)
    with pytest.raises(ValueError):
        adapter.distribution(P, ACTS, level="action")   # no selected_location
    dist = adapter.distribution(P, ACTS, level="action", selected_location=LOCS[0])
    assert abs(dist.sum() - 1.0) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_policies.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'npc_policy.policies'`

- [ ] **Step 3: Write the implementation**

Create `npc_policy/policies.py`:

```python
"""Unified policy layer — one calling convention for all four policy variants.

A *policy* for sequence generation is any object exposing

    distribution(personality, candidates, buffer=None, level="location",
                 selected_location=None) -> np.ndarray   # probs over candidates

``DecisionController`` drives either a ``HandAuthoredScorer`` (special-cased so
the full ``ScoreTrace`` is still recorded) or any object with the shape above.
``LearnedPolicyAdapter`` gives trained RQ2 checkpoints that shape: it computes
relation features from the buffer exactly as the scorer does, then delegates to
``learned.predict_distribution``. Spec:
``docs/specs/2026-07-17-rq3-sequence-interface-design.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import DEFAULT_CONFIG, ScorerConfig
from .learned import AgnosticPolicy, NonlinearPolicy, SimplePolicy, predict_distribution
from .relations import compute_relations
from .representation import Option, Personality, RecentBuffer


def build_architecture(name: str) -> torch.nn.Module:
    """Fresh module for a checkpoint's ``payload["model"]`` name.

    Mirrors ``experiments.rq2.common.build_model`` without the seeding (the
    weights are immediately overwritten by ``load_state_dict``); duplicated here
    because ``npc_policy`` must not import from ``experiments``.
    """
    if name == "simple":
        return SimplePolicy()
    if name == "nonlinear":
        return NonlinearPolicy()
    if name == "agnostic_simple":
        return AgnosticPolicy(SimplePolicy())
    if name == "agnostic_nonlinear":
        return AgnosticPolicy(NonlinearPolicy())
    raise ValueError(f"unknown architecture {name!r}")


class LearnedPolicyAdapter:
    """A trained checkpoint behind the unified ``distribution`` convention."""

    def __init__(self, checkpoint_path: str | Path,
                 config: ScorerConfig = DEFAULT_CONFIG):
        self.checkpoint_path = Path(checkpoint_path)
        payload = torch.load(self.checkpoint_path, weights_only=False)
        self.architecture = payload["model"]
        self.model = build_architecture(self.architecture)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.config = config          # recency/similarity params for relations

    def distribution(
        self,
        personality: Personality,
        candidates: list[Option],
        buffer: RecentBuffer | None = None,
        level: str = "location",
        selected_location: Option | None = None,
    ) -> np.ndarray:
        relations = None
        if buffer is not None and not buffer.is_empty():
            relations = compute_relations(candidates, buffer, self.config)
        return predict_distribution(
            self.model, personality, candidates, level,
            relations=relations, selected_location=selected_location,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_policies.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add npc_policy/policies.py tests/test_policies.py
git commit -m "feat: LearnedPolicyAdapter — unified distribution() over RQ2 checkpoints"
```

---

### Task 2: Generalise `DecisionController` to any policy

**Files:**
- Modify: `npc_policy/controller.py` (docstring, `Decision`, `__init__`, both choose methods)
- Test: `tests/test_policies.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_policies.py`:

```python
from npc_policy.controller import DecisionController
from npc_policy.scorer import HandAuthoredScorer


def replay_scorer_by_hand(scorer, personality, worlds_cycles, seed):
    """Reference behaviour: drive scorer + buffers manually, mirroring the
    controller's documented rules, for regression comparison."""
    rng = np.random.default_rng(seed)
    H_L = RecentBuffer(maxlen=DEFAULT_CONFIG.K_L)
    H_A = RecentBuffer(maxlen=DEFAULT_CONFIG.K_A)
    last_loc = None
    picks = []
    for locs, acts_of in worlds_cycles:
        t = scorer.trace(personality, locs, buffer=H_L, level="location")
        i = int(rng.choice(len(locs), p=t.P_rule))
        loc = locs[i]
        if loc.id != last_loc:
            H_A.clear()
        H_L.push(loc)
        last_loc = loc.id
        acts = acts_of(loc.id)
        t2 = scorer.trace(personality, acts, buffer=H_A, level="action")
        j = int(rng.choice(len(acts), p=t2.P_rule))
        H_A.push(acts[j])
        picks.append((loc.id, acts[j].id))
    return picks


def test_controller_scorer_regression():
    """Scorer through the new policy-shaped controller reproduces the manual
    reference exactly (same rng consumption order, same buffer rules)."""
    scorer = HandAuthoredScorer()
    acts_of = lambda loc_id: ACTS
    cycles = [(LOCS, acts_of)] * 6
    want = replay_scorer_by_hand(scorer, P, cycles, seed=7)

    ctrl = DecisionController(scorer, mode="sample", rng=np.random.default_rng(7))
    got = []
    for locs, acts_fn in cycles:
        d_loc = ctrl.choose_location(P, locs)
        d_act = ctrl.choose_action(P, acts_fn(d_loc.option.id))
        assert d_loc.trace is not None                       # scorer keeps traces
        np.testing.assert_array_equal(d_loc.distribution, d_loc.trace.P_rule)
        got.append((d_loc.option.id, d_act.option.id))
    assert got == want


def test_controller_drives_learned_adapter(tmp_path):
    path, model = save_checkpoint(tmp_path, "nonlinear")
    ctrl = DecisionController(LearnedPolicyAdapter(path), mode="sample",
                              rng=np.random.default_rng(0))
    d_loc = ctrl.choose_location(P, LOCS)
    assert d_loc.trace is None
    assert abs(d_loc.distribution.sum() - 1.0) < 1e-9
    d_act = ctrl.choose_action(P, ACTS)        # would raise without ctx pass-through
    assert d_act.trace is None

    # first action after arriving: H_A empty -> equals context-only prediction
    want = predict_distribution(model, P, ACTS, "action",
                                relations=None, selected_location=d_loc.option)
    np.testing.assert_allclose(d_act.distribution, want, rtol=0, atol=1e-12)


def test_controller_action_context_follows_location_change(tmp_path):
    path, model = save_checkpoint(tmp_path, "nonlinear")
    ctrl = DecisionController(LearnedPolicyAdapter(path), mode="argmax")
    ctrl.choose_location(P, [LOCS[0]])          # forced tavern
    ctrl.choose_action(P, ACTS)
    ctrl.choose_location(P, [LOCS[1]])          # forced move -> library
    d = ctrl.choose_action(P, ACTS)
    # buffer was cleared by the move AND context switched to the new location
    want = predict_distribution(model, P, ACTS, "action",
                                relations=None, selected_location=LOCS[1])
    np.testing.assert_allclose(d.distribution, want, rtol=0, atol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_policies.py -v`
Expected: `test_controller_scorer_regression` already PASSES (it pins the current
scorer-driven behaviour — that is the regression baseline); the two adapter
tests FAIL with `AttributeError: 'LearnedPolicyAdapter' object has no attribute
'trace'` (the controller still hard-codes the scorer). Task 1 tests still PASS.

- [ ] **Step 3: Modify `npc_policy/controller.py`**

Replace the `Decision` dataclass (currently lines 32-46) with:

```python
@dataclass(frozen=True)
class Decision:
    """Outcome of one location or action choice.

    ``distribution`` is the policy's choice distribution over the candidate
    list. ``trace`` carries the scorer's full breakdown when the driving policy
    is the hand-authored scorer; learned policies provide no trace.
    """

    option: Option          # the chosen option
    index: int              # its index in the candidate list passed in
    distribution: np.ndarray
    trace: ScoreTrace | None = None
```

In `DecisionController.__init__`, rename the first parameter `scorer` → `policy` and the attribute `self.scorer` → `self.policy` (all callers pass it positionally — verified). Update the class docstring line to: `"""Holds ``H_t^L`` and ``H_t^A`` and runs the nested choice for one NPC, driven by any policy (see npc_policy.policies)."""`

Add a private dispatch method after `_select`:

```python
    # -- policy dispatch --------------------------------------------------------
    def _distribution(
        self,
        personality: Personality,
        candidates: list[Option],
        buffer: RecentBuffer,
        level: str,
        selected_location: Option | None,
    ) -> tuple[np.ndarray, ScoreTrace | None]:
        """Ask the policy for its choice distribution.

        The hand-authored scorer is special-cased so its full trace is kept
        (rq1/rq2 dataset generation records it); any other policy follows the
        unified ``distribution`` convention of ``npc_policy.policies``.
        """
        if isinstance(self.policy, HandAuthoredScorer):
            trace = self.policy.trace(personality, candidates, buffer=buffer, level=level)
            return trace.P_rule, trace
        dist = self.policy.distribution(
            personality, candidates, buffer=buffer, level=level,
            selected_location=selected_location,
        )
        return np.asarray(dist, dtype=float), None
```

Track the selected location `Option` (needed as action context): in `__init__`, next to `_last_location_id`, add:

```python
        self._last_location: Option | None = None
```

Rewrite the two choose methods to use the dispatch:

```python
    def choose_location(
        self, personality: Personality, locations: list[Option]
    ) -> Decision:
        """Choose a location using ``H_t^L``; then commit it and apply the
        action-buffer reset rule."""
        dist, trace = self._distribution(personality, locations, self.H_L,
                                         "location", None)
        idx = self._select(dist)
        chosen = locations[idx]

        # reset local action buffer BEFORE the next action choice if location changed
        if chosen.id != self._last_location_id:
            self.H_A.clear()
        # commit the location into the recent-location buffer
        self.H_L.push(chosen)
        self._last_location_id = chosen.id
        self._last_location = chosen
        return Decision(option=chosen, index=idx, distribution=dist, trace=trace)

    def choose_action(
        self, personality: Personality, actions: list[Option]
    ) -> Decision:
        """Choose an action from the current location's action set using ``H_t^A``
        (empty right after a location change), then record it. Learned policies
        additionally receive the selected location as context."""
        dist, trace = self._distribution(personality, actions, self.H_A,
                                         "action", self._last_location)
        idx = self._select(dist)
        chosen = actions[idx]
        self.H_A.push(chosen)
        return Decision(option=chosen, index=idx, distribution=dist, trace=trace)
```

In `reset()`, add `self._last_location = None` beside `self._last_location_id = None`. In `Decision`'s old body, delete the `@property distribution` (replaced by the field). Keep `mode`/`min_p`/`selection_temperature` logic untouched.

- [ ] **Step 4: Run the full suite (regression across rq1/rq2 users of the controller)**

Run: `python -m pytest tests/ -v`
Expected: all PASS (existing `test_rq2_pipeline.py` etc. exercise the controller via positional scorer).

- [ ] **Step 5: Commit**

```bash
git add npc_policy/controller.py tests/test_policies.py
git commit -m "feat: DecisionController drives any policy; Decision carries distribution + optional trace"
```

---

### Task 3: Sequence generation core (`experiments/rq3/common.py`)

**Files:**
- Create: `experiments/rq3/__init__.py` (empty file)
- Create: `experiments/rq3/common.py`
- Test: `tests/test_rq3_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rq3_pipeline.py`:

```python
"""RQ3 sequence pipeline: generation core + CLI (Task 4 appends CLI tests)."""

import json

import numpy as np
import pytest

from experiments.rq3.common import (
    SequenceSpec, format_preview, generate_sequence, validate_sequence,
)
from npc_policy.representation import Personality
from npc_policy.scorer import HandAuthoredScorer
from npc_policy.world import load_world

TINY_WORLD = {
    "locations": [
        {"id": "tavern",
         "features": {"social": 0.9, "stimulation": 0.7},
         "actions": [{"id": "chat", "features": {"social": 0.9}},
                     {"id": "drink", "features": {"stimulation": 0.6}}]},
        {"id": "library",
         "features": {"cognitive": 0.9, "privacy": 0.8},
         "actions": [{"id": "read", "features": {"cognitive": 0.9}},
                     {"id": "study", "features": {"cognitive": 0.8, "structure": 0.6}}]},
    ]
}


@pytest.fixture
def world(tmp_path):
    path = tmp_path / "world.json"
    path.write_text(json.dumps(TINY_WORLD), encoding="utf-8")
    return load_world(path)


def make_spec(seed=42, n_cycles=5):
    return SequenceSpec(
        sequence_id="S01", policy_name="scorer", checkpoint="",
        personality_name="high_E",
        personality=Personality.from_traits(extraversion=1.0),
        world_path="world.json", n_cycles=n_cycles, seed=seed,
    )


def test_sequence_structure_and_moved_flags(world):
    seq = generate_sequence(make_spec(), HandAuthoredScorer(), world)
    assert seq["meta"]["sequence_id"] == "S01"
    assert seq["meta"]["ocean"] == {"O": 0.0, "C": 0.0, "E": 1.0, "A": 0.0, "N": 0.0}
    steps = seq["steps"]
    assert [s["cycle"] for s in steps] == [1, 2, 3, 4, 5]
    assert steps[0]["moved"] is True                     # first step always walks
    for prev, cur in zip(steps, steps[1:]):
        assert cur["moved"] == (cur["location"] != prev["location"])
    for s in steps:
        assert abs(sum(s["location_probs"].values()) - 1.0) < 1e-6
        assert abs(sum(s["action_probs"].values()) - 1.0) < 1e-6
        assert s["action"] in s["action_probs"]


def test_same_seed_reproduces_exactly(world):
    a = generate_sequence(make_spec(seed=7), HandAuthoredScorer(), world)
    b = generate_sequence(make_spec(seed=7), HandAuthoredScorer(), world)
    assert a["steps"] == b["steps"]


def test_different_seed_may_differ_and_validates(world):
    seq = generate_sequence(make_spec(seed=8, n_cycles=8), HandAuthoredScorer(), world)
    validate_sequence(seq, world)                        # should not raise


def test_validate_rejects_unknown_ids_and_bad_probs(world):
    seq = generate_sequence(make_spec(), HandAuthoredScorer(), world)
    bad = json.loads(json.dumps(seq))                    # deep copy
    bad["steps"][2]["location"] = "arena"
    with pytest.raises(ValueError, match="arena"):
        validate_sequence(bad, world)

    bad2 = json.loads(json.dumps(seq))
    bad2["steps"][0]["action_probs"] = {"chat": 0.4, "drink": 0.4}
    with pytest.raises(ValueError, match="sum"):
        validate_sequence(bad2, world)


def test_preview_is_one_line_per_sequence(world):
    seq = generate_sequence(make_spec(), HandAuthoredScorer(), world)
    line = format_preview(seq)
    assert line.startswith("S01")
    assert "scorer" in line and "high_E" in line
    assert line.count("/") >= 5                          # loc/act per cycle
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rq3_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'experiments.rq3'`

- [ ] **Step 3: Write the implementation**

Create empty `experiments/rq3/__init__.py`, then `experiments/rq3/common.py`:

```python
"""RQ3 sequence-generation core: one (personality, policy, world, seed) spec ->
one replayable sequence dict (spec: docs/specs/2026-07-17-rq3-sequence-interface-design.md).

The sequence dict is exactly what is written to JSON for the Unity playback
player. ``location_probs`` / ``action_probs`` are research-archive fields the
player ignores. No time information is stored: playback pacing is a player
concern (continue button / auto-advance)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from npc_policy.controller import DecisionController
from npc_policy.representation import Personality
from npc_policy.world import World

OCEAN_KEYS = ("O", "C", "E", "A", "N")


@dataclass(frozen=True)
class SequenceSpec:
    """Generation conditions for one sequence (everything the manifest records)."""

    sequence_id: str
    policy_name: str
    checkpoint: str           # "" for the hand-authored scorer
    personality_name: str
    personality: Personality
    world_path: str
    n_cycles: int
    seed: int


def generate_sequence(spec: SequenceSpec, policy, world: World) -> dict:
    """Roll one NPC for ``n_cycles`` decision cycles and return the sequence dict.

    Same seed + same spec -> identical steps (the controller's rng is the only
    randomness; learned adapters run in eval mode)."""
    ctrl = DecisionController(policy, mode="sample",
                              rng=np.random.default_rng(spec.seed))
    steps: list[dict] = []
    prev: str | None = None
    for cycle in range(1, spec.n_cycles + 1):
        locs = world.resolve()
        d_loc = ctrl.choose_location(spec.personality, locs)
        acts = world.actions_at(d_loc.option.id)
        d_act = ctrl.choose_action(spec.personality, acts)
        steps.append({
            "cycle": cycle,
            "location": d_loc.option.id,
            "action": d_act.option.id,
            "moved": d_loc.option.id != prev,
            "location_probs": {o.id: float(p) for o, p in zip(locs, d_loc.distribution)},
            "action_probs": {o.id: float(p) for o, p in zip(acts, d_act.distribution)},
        })
        prev = d_loc.option.id
    return {
        "meta": {
            "sequence_id": spec.sequence_id,
            "policy": spec.policy_name,
            "checkpoint": spec.checkpoint,
            "personality_name": spec.personality_name,
            "ocean": dict(zip(OCEAN_KEYS, (float(v) for v in spec.personality.vector))),
            "world": spec.world_path,
            "seed": spec.seed,
            "n_cycles": spec.n_cycles,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "steps": steps,
    }


def validate_sequence(seq: dict, world: World) -> None:
    """Export gate: raise ValueError rather than let a bad file reach Unity."""
    sid = seq["meta"]["sequence_id"]
    steps = seq["steps"]
    if len(steps) != seq["meta"]["n_cycles"]:
        raise ValueError(f"{sid}: {len(steps)} steps != n_cycles {seq['meta']['n_cycles']}")
    for step in steps:
        loc = step["location"]
        if loc not in world.location_ids() or not world.entries[loc].unlocked:
            raise ValueError(f"{sid} cycle {step['cycle']}: unknown/locked location {loc!r}")
        action_ids = {a.id for a in world.actions_at(loc)}
        if step["action"] not in action_ids:
            raise ValueError(
                f"{sid} cycle {step['cycle']}: action {step['action']!r} not at {loc!r}")
        for key in ("location_probs", "action_probs"):
            total = sum(step[key].values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"{sid} cycle {step['cycle']}: {key} sum {total} != 1")


def format_preview(seq: dict) -> str:
    """One text line per sequence for pre-recording quality control."""
    m = seq["meta"]
    trail = " -> ".join(f"{s['location']}/{s['action']}" for s in seq["steps"])
    return (f"{m['sequence_id']} [{m['policy']} | {m['personality_name']} | "
            f"seed {m['seed']}]: {trail}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_rq3_pipeline.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/rq3/__init__.py experiments/rq3/common.py tests/test_rq3_pipeline.py
git commit -m "feat: RQ3 sequence generation core with export validation and preview"
```

---

### Task 4: Batch CLI (`experiments/rq3/gen_sequences.py`) + smoke config

**Files:**
- Create: `experiments/rq3/gen_sequences.py`
- Create: `experiments/rq3/config_smoke.json`
- Test: `tests/test_rq3_pipeline.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rq3_pipeline.py`:

```python
import csv

import torch

from experiments.rq3.gen_sequences import run_config
from npc_policy.policies import build_architecture


def write_config(tmp_path, world_path, policies):
    cfg = {
        "out_dir": str(tmp_path / "out"),
        "worlds": [str(world_path)],
        "personalities": [
            {"name": "high_E", "ocean": {"extraversion": 1.0}},
            {"name": "low_E", "ocean": {"extraversion": -1.0}},
        ],
        "policies": policies,
        "n_cycles": 4,
        "seeds": [42, 43],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path, cfg


def test_run_config_end_to_end(tmp_path, world):
    world_path = tmp_path / "world.json"          # written by the fixture
    torch.manual_seed(0)
    model = build_architecture("simple")
    ckpt = tmp_path / "simple.pt"
    torch.save({"model": "simple", "state_dict": model.state_dict()}, ckpt)

    cfg_path, cfg = write_config(
        tmp_path, world_path,
        [{"name": "scorer"}, {"name": "simple_2b", "checkpoint": str(ckpt)}],
    )
    n = run_config(cfg_path, preview=False)
    # 1 world x 2 personalities x 2 policies x 2 seeds
    assert n == 8

    out = tmp_path / "out"
    files = sorted(f.name for f in out.glob("S*.json"))
    assert files == [f"S{i:02d}.json" for i in range(1, 9)]

    with open(out / "manifest.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 8
    assert rows[0]["sequence_id"] == "S01"
    assert {r["policy"] for r in rows} == {"scorer", "simple_2b"}

    # every written file passes the export gate again on re-read
    for name in files:
        seq = json.loads((out / name).read_text(encoding="utf-8"))
        validate_sequence(seq, world)


def test_run_config_reproducible_steps(tmp_path, world):
    world_path = tmp_path / "world.json"
    cfg_path, _ = write_config(tmp_path, world_path, [{"name": "scorer"}])
    run_config(cfg_path, preview=False)
    first = {f.name: json.loads(f.read_text(encoding="utf-8"))["steps"]
             for f in (tmp_path / "out").glob("S*.json")}
    run_config(cfg_path, preview=False)               # overwrite in place
    second = {f.name: json.loads(f.read_text(encoding="utf-8"))["steps"]
              for f in (tmp_path / "out").glob("S*.json")}
    assert first == second
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rq3_pipeline.py -v`
Expected: the two new tests FAIL (`ImportError: cannot import name 'run_config'`); Task 3 tests still PASS.

- [ ] **Step 3: Write the implementation**

Create `experiments/rq3/gen_sequences.py`:

```python
"""Batch-generate RQ3 behaviour sequences from a config file.

    python -m experiments.rq3.gen_sequences --config experiments/rq3/config_smoke.json --preview

Run from ``code/``. Config JSON:

    { "out_dir": "data/rq3_sequences",
      "worlds": ["data/world.json"],
      "personalities": [ {"name": "high_E", "ocean": {"extraversion": 1.0}}, ... ],
      "policies":      [ {"name": "scorer"},
                         {"name": "nonlinear_2b",
                          "checkpoint": "results/rq2b/models/<file>.pt"}, ... ],
      "n_cycles": 10,
      "seeds": [42] }

Sequences are crossed in world -> personality -> policy -> seed order and
numbered S01, S02, ... deterministically; ``manifest.csv`` in ``out_dir`` is the
stimulus ledger (one row per sequence). Files are permanent research artefacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from npc_policy.policies import LearnedPolicyAdapter
from npc_policy.representation import Personality
from npc_policy.scorer import HandAuthoredScorer
from npc_policy.world import load_world

from .common import SequenceSpec, format_preview, generate_sequence, validate_sequence

MANIFEST_FIELDS = ["sequence_id", "file", "policy", "checkpoint",
                   "personality", "world", "seed", "n_cycles", "generated_at"]


def build_policy(entry: dict):
    """'scorer' -> hand-authored scorer; anything else needs a checkpoint path."""
    if entry["name"] == "scorer":
        return HandAuthoredScorer()
    if "checkpoint" not in entry:
        raise ValueError(f"policy {entry['name']!r} needs a 'checkpoint' path")
    return LearnedPolicyAdapter(entry["checkpoint"])


def run_config(config_path: str | Path, preview: bool = False) -> int:
    """Generate every configured sequence; returns the number written."""
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    policies = [(e, build_policy(e)) for e in cfg["policies"]]

    rows: list[dict] = []
    i = 0
    for world_path in cfg["worlds"]:
        world = load_world(world_path)
        for pers in cfg["personalities"]:
            personality = Personality.from_traits(**pers.get("ocean", {}))
            for entry, policy in policies:
                for seed in cfg["seeds"]:
                    i += 1
                    spec = SequenceSpec(
                        sequence_id=f"S{i:02d}",
                        policy_name=entry["name"],
                        checkpoint=entry.get("checkpoint", ""),
                        personality_name=pers["name"],
                        personality=personality,
                        world_path=str(world_path),
                        n_cycles=int(cfg["n_cycles"]),
                        seed=int(seed),
                    )
                    seq = generate_sequence(spec, policy, world)
                    validate_sequence(seq, world)
                    fname = f"{spec.sequence_id}.json"
                    (out_dir / fname).write_text(
                        json.dumps(seq, indent=2), encoding="utf-8")
                    m = seq["meta"]
                    rows.append({
                        "sequence_id": m["sequence_id"], "file": fname,
                        "policy": m["policy"], "checkpoint": m["checkpoint"],
                        "personality": m["personality_name"], "world": m["world"],
                        "seed": m["seed"], "n_cycles": m["n_cycles"],
                        "generated_at": m["generated_at"],
                    })
                    if preview:
                        print(format_preview(seq))

    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {i} sequences + manifest.csv to {out_dir}")
    return i


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--preview", action="store_true",
                    help="print each sequence as one text line for QC")
    args = ap.parse_args()
    run_config(args.config, preview=args.preview)


if __name__ == "__main__":
    main()
```

Create `experiments/rq3/config_smoke.json` (scorer-only so it never depends on a
particular training run; add learned entries with real checkpoint paths when
composing the actual study config):

```json
{
  "out_dir": "data/rq3_sequences/smoke",
  "worlds": ["data/world.json"],
  "personalities": [
    {"name": "high_E", "ocean": {"extraversion": 1.0}},
    {"name": "low_E_high_N", "ocean": {"extraversion": -1.0, "neuroticism": 1.0}}
  ],
  "policies": [{"name": "scorer"}],
  "n_cycles": 8,
  "seeds": [42]
}
```

- [ ] **Step 4: Run tests, then the smoke config for real**

Run: `python -m pytest tests/test_rq3_pipeline.py -v`
Expected: 7 PASS

Run: `python -m experiments.rq3.gen_sequences --config experiments/rq3/config_smoke.json --preview`
Expected: two preview lines (S01/S02: eight `loc/act` hops each) + `wrote 2 sequences + manifest.csv to data\rq3_sequences\smoke`

- [ ] **Step 5: Commit**

```bash
git add experiments/rq3/gen_sequences.py experiments/rq3/config_smoke.json tests/test_rq3_pipeline.py
git commit -m "feat: RQ3 batch sequence CLI with manifest and preview QC"
```

---

### Task 5: Full-suite verification

- [ ] **Step 1: Run everything**

Run: `python -m pytest tests/ -v`
Expected: all tests PASS (existing rq1/rq2 suites confirm no controller regression).

- [ ] **Step 2: Commit generated smoke artefacts** (they are the first concrete
example of the Unity input format and useful for the Unity work later)

```bash
git add data/rq3_sequences/smoke
git commit -m "data: smoke RQ3 sequences (scorer, 2 personalities, seed 42)"
```

## Out of scope (later rounds)

Unity playback player, live demo service (state/step/event/reset), desktop
packaging, and the real study config (checkpoint selection, personality set,
cycle counts) — all per the spec's Unity-side contract.
