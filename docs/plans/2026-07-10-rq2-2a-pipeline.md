# RQ2 Study 2A Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The four `experiments/rq2/` CLI modules (dataset generation, resumable training, metrics, E1–E4 diagnostic), their tests, and a Chinese runbook — per `docs/specs/2026-07-10-rq2-2a-pipeline-design.md`.

**Architecture:** One master pool of teacher-labelled cases with generation tags; G-splits are content-based filters over the pool plus targeted test sets. Training is a flat list of 130 `RunSpec`s, each writing its own result JSON + weights (resume = skip existing). Metrics and the structural diagnostic only read files written by earlier steps. The user runs the four commands himself; nothing here launches long jobs automatically.

**Tech Stack:** Python 3.11+, numpy, PyTorch (CPU float64 / CUDA float32), matplotlib (Agg), pytest. Reuses `npc_policy` (scorer/teacher, controller, cases, features, learned models) and `experiments/rq1/common.py` conventions.

**Working directory:** `C:\Users\76992\Desktop\MSc_dissertation\code` (the git repo). Run all commands from there.

**Key existing interfaces (do not modify these modules):**
- `npc_policy.scorer.HandAuthoredScorer.trace(personality, candidates, buffer=None, relations=None, level=...)` → `ScoreTrace` with `.P_rule`, `.relations`.
- `npc_policy.controller.DecisionController(scorer, config=..., mode="sample", rng=..., selection_temperature=...)` — `.choose_location/.choose_action` return `Decision(option, index, trace)`; buffers `H_L`/`H_A` with `.recent_to_old()` (newest→oldest).
- `npc_policy.cases.ControlledCase(personality, decision_type, candidates, selected_location, recent_locations, recent_actions_same_location, candidate_history_features, target_distribution)` + `to_dict/from_dict` (`from_dict` ignores extra JSON keys — generation tags ride in a `"gen"` key).
- `npc_policy.features.case_inputs(personality, candidates, decision_type, relations=None, selected_location=None)` → dict for `PolicyBatch.from_cases`; action cases REQUIRE `selected_location`, location cases must NOT have one.
- `npc_policy.learned`: `PolicyBatch` (frozen dataclass; `replace()` works), `SimplePolicy` (`.w` shape `(2, 114)`; the last 12 phi dims are the `c_L ⊙ o` block), `NonlinearPolicy`, `AgnosticPolicy(inner)`, `UniformBaseline`, `kl_loss(log_q, target, mask)`, `predict_distribution(model, ...)`.
- `npc_policy.relations.compute_relations(candidates, buffer, config)`; empty buffer → all-zero `Relations`.
- `experiments/rq1/common.py`: `TRAITS`, `TRAIT_COLORS`, `LOC_COLORS`, `setup_style`, `load_cases`, `world_for`, `personality_of`, `location_buffer`, `entropy`, `pairwise_jsd`, `spearman`, `mantel`, `trajectory_metrics`, `write_csv`, `TRAJ_TEMPERATURE=0.1`.
- `experiments/rq1/gen_cases.build_worlds()` — idempotent, writes `data/rq1_cases/worlds/*.json` from `data/world.json`.

**File structure:**

```text
experiments/rq2/__init__.py        (empty)
experiments/rq2/common.py          paths, config hash, pool I/O, case→input assembly,
                                   numpy metrics, RunSpec + run matrix, model factory
experiments/rq2/gen_controlled.py  SyntheticSampler, rollout collector, split filters,
                                   manifests, targeted test sets, CLI
experiments/rq2/train.py           train_one, run_all (resumable), device handling, CLI
experiments/rq2/run_2a.py          evaluation, main table, figures, diagnostics, CLI
experiments/rq2/run_e_diag.py      StudentTraceAdapter, student trajectories,
                                   E1 overlay / E2 correlation / E3–E4 stats, CLI
tests/test_rq2_pipeline.py         all acceptance tests (design §6)
docs/rq2_runbook.md                Chinese operating manual
```

Conventions: every test gets a fixed seed; tests write only to `tmp_path`; figures use Agg backend via `rq1.common.setup_style()`; commit after every task with the trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: `experiments/rq2/common.py` — shared utilities

**Files:**
- Create: `experiments/rq2/__init__.py` (empty file)
- Create: `experiments/rq2/common.py`
- Create: `tests/test_rq2_pipeline.py`

- [ ] **Step 1.1: Write the failing tests**

Create `tests/test_rq2_pipeline.py`:

```python
"""Acceptance tests for the Study 2A pipeline (design spec §6:
docs/specs/2026-07-10-rq2-2a-pipeline-design.md)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from npc_policy import (
    DEFAULT_CONFIG,
    ControlledCase,
    HandAuthoredScorer,
    Option,
    Personality,
    ScorerConfig,
)
from experiments.rq2 import common


def _loc(id_, **tags):
    return Option.location(id_, **tags)


def _act(id_, **tags):
    return Option.action(id_, **tags)


def _mk_case(decision_type="location", personality=None, n_cand=3,
             selected=None, recent_locs=(), recent_acts=(), target=None):
    """Minimal hand-built ControlledCase for unit tests."""
    p = np.zeros(5) if personality is None else np.asarray(personality, float)
    if decision_type == "location":
        cands = [_loc(f"L{i}", social=0.1 * i) for i in range(n_cand)]
    else:
        cands = [_act(f"A{i}", social=0.1 * i) for i in range(n_cand)]
    t = np.full(n_cand, 1.0 / n_cand) if target is None else np.asarray(target, float)
    return ControlledCase(
        personality=p, decision_type=decision_type, candidates=cands,
        selected_location=selected, recent_locations=list(recent_locs),
        recent_actions_same_location=list(recent_acts),
        candidate_history_features=None, target_distribution=t,
    )


class TestCommon:
    def test_config_hash_stable_and_sensitive(self):
        h1 = common.config_hash(DEFAULT_CONFIG)
        h2 = common.config_hash(ScorerConfig())
        h3 = common.config_hash(ScorerConfig(K_L=2))
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 64

    def test_pool_roundtrip(self, tmp_path):
        case = _mk_case()
        common.write_pool(tmp_path / "p.jsonl", [(case, {"id": "syn-000000", "source": "synthetic", "world": "full"})])
        [(back, tags)] = common.read_pool(tmp_path / "p.jsonl")
        assert tags == {"id": "syn-000000", "source": "synthetic", "world": "full"}
        assert back.decision_type == "location"
        np.testing.assert_allclose(back.target_distribution, case.target_distribution)

    def test_case_to_inputs_action_uses_newest_recent_location(self):
        home = _loc("home", social=0.4, privacy=0.9)
        case = _mk_case("action", selected="home", recent_locs=[_loc("far"), home])
        d = common.case_to_inputs(case)
        np.testing.assert_allclose(d["ctx"], home.to_padded12())
        np.testing.assert_allclose(d["target"], case.target_distribution)
        assert d["d"] == 1

    def test_case_to_inputs_rejects_broken_action_invariant(self):
        case = _mk_case("action", selected="home", recent_locs=[_loc("elsewhere")])
        with pytest.raises(ValueError):
            common.case_to_inputs(case)

    def test_ablation_zeroes_relations(self):
        from npc_policy.relations import Relations
        rel = Relations(np.array([0.5, 0.0]), np.array([0.7, 0.2]), np.array([0.3, 0.8]))
        case = _mk_case(n_cand=2)
        case.candidate_history_features = rel
        full = common.case_to_inputs(case, "full")
        none = common.case_to_inputs(case, "none")
        assert full["rel"].any()
        assert not none["rel"].any()
        # location_only keeps location-case relations, zeroes action-case ones
        loc_only = common.case_to_inputs(case, "location_only")
        assert loc_only["rel"].any()
        home = _loc("home")
        act = _mk_case("action", n_cand=2, selected="home", recent_locs=[home])
        act.candidate_history_features = rel
        assert not common.case_to_inputs(act, "location_only")["rel"].any()

    def test_metrics_pinned(self):
        t = np.array([1.0, 0.0])
        q = np.array([0.5, 0.5])
        assert common.kl_np(t, q) == pytest.approx(np.log(2.0))
        assert common.kl_np(t, t) == pytest.approx(0.0)
        assert common.jsd_np(t, t) == pytest.approx(0.0)
        assert common.top1_agree(np.array([0.9, 0.1]), np.array([0.6, 0.4]))
        assert not common.top1_agree(np.array([0.9, 0.1]), np.array([0.4, 0.6]))

    def test_run_matrix_counts(self):
        runs = common.run_matrix()
        assert len(runs) == 130
        assert len({r.run_id for r in runs}) == 130
        s0_main = [r for r in runs if r.split == "S0" and r.ablation == "full" and r.n_train is None]
        assert len(s0_main) == 20
        assert all(r.model in ("simple", "nonlinear") for r in runs if r.split != "S0")
        smoke = common.run_matrix(smoke=True)
        assert 0 < len(smoke) <= 6

    def test_build_model_names(self):
        import torch
        for name in common.S0_MODELS:
            m = common.build_model(name, seed=0)
            assert isinstance(m, torch.nn.Module)
        with pytest.raises(ValueError):
            common.build_model("mlp", 0)
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rq2_pipeline.py -q`
Expected: collection error / failures with `ModuleNotFoundError: No module named 'experiments.rq2'`.

- [ ] **Step 1.3: Implement**

Create empty `experiments/rq2/__init__.py`, then `experiments/rq2/common.py`:

```python
"""Shared utilities for the RQ2 Study 2A experiments.

Design: ``docs/specs/2026-07-10-rq2-2a-pipeline-design.md``. Pool records pair a
``ControlledCase`` with generation tags (``"gen"``: id / source / world) used only
by the split filters — tags never reach a model. All randomness in this package
derives from ``GEN_SEED`` (generation) or the run seed (training).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from npc_policy import DEFAULT_CONFIG, ControlledCase, ScorerConfig
from npc_policy.features import case_inputs
from npc_policy.representation import Personality

CODE = Path(__file__).resolve().parents[2]
DATA = CODE / "data"
RQ1_WORLDS = DATA / "rq1_cases" / "worlds"

GEN_SEED = 20260709            # dataset-generation base seed (RQ1 convention)
TRAIN_SIZE = 100_000           # per split, matched across splits [PROVISIONAL]
VAL_SIZE = 5_000
TEST_SIZE = 5_000

MODELS = ("simple", "nonlinear")
S0_MODELS = ("simple", "nonlinear", "agnostic_simple", "agnostic_nonlinear")
SEEDS = tuple(range(5))
G_SPLITS = ("G1", "G2", "G3", "G4", "G5", "G6")
ALL_SPLITS = ("S0",) + G_SPLITS
ABLATIONS = ("none", "location_only")   # "full" is the S0 main configuration
DATA_SIZES = (1_000, 5_000, 20_000)     # 100k point reuses the S0 main runs


def dirs(smoke: bool) -> tuple[Path, Path]:
    """(dataset dir, results dir); smoke mode is fully isolated from the full run."""
    if smoke:
        return DATA / "rq2_controlled_smoke", CODE / "results" / "rq2_smoke"
    return DATA / "rq2_controlled", CODE / "results" / "rq2"


def config_hash(config: ScorerConfig = DEFAULT_CONFIG) -> str:
    """SHA-256 of the serialised scorer config — pins the frozen teacher."""
    payload = json.dumps(dataclasses.asdict(config), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ pool I/O --
def write_pool(path: Path, records: list[tuple[ControlledCase, dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for case, tags in records:
            d = case.to_dict()
            d["gen"] = tags
            f.write(json.dumps(d) + "\n")


def read_pool(path: Path) -> list[tuple[ControlledCase, dict]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            tags = d.pop("gen", {})
            records.append((ControlledCase.from_dict(d), tags))
    return records


# ------------------------------------------------------- model input assembly --
def case_to_inputs(case: ControlledCase, ablation: str = "full") -> dict:
    """``ControlledCase`` → ``features.case_inputs`` dict plus ``"target"``.

    Action cases take the selected location's features from the newest
    ``recent_locations`` entry — the controller pushes the chosen location into
    ``H_L`` before the action choice, and generation guarantees the invariant.
    ``ablation`` zeroes relation *inputs* ("none": both levels; "location_only":
    action cases only); targets are never touched (retrained-ablation design).
    """
    if ablation not in ("full", "none", "location_only"):
        raise ValueError(f"unknown ablation {ablation!r}")
    selected = None
    if case.decision_type == "action":
        if (not case.recent_locations
                or case.recent_locations[-1].id != case.selected_location):
            raise ValueError(
                "action case must carry its selected location as the newest "
                "recent_locations entry"
            )
        selected = case.recent_locations[-1]
    relations = case.candidate_history_features
    if ablation == "none" or (ablation == "location_only" and case.decision_type == "action"):
        relations = None
    d = case_inputs(
        Personality(np.asarray(case.personality, dtype=float)),
        case.candidates, case.decision_type,
        relations=relations, selected_location=selected,
    )
    d["target"] = np.asarray(case.target_distribution, dtype=float)
    return d


# ------------------------------------------------------------------- metrics --
def kl_np(t: np.ndarray, q: np.ndarray) -> float:
    """``KL(t ‖ q)`` in nats; exact at ``t_i = 0`` (``q > 0`` from the softmax)."""
    t, q = np.asarray(t, dtype=float), np.asarray(q, dtype=float)
    pos = t > 0
    return float((t[pos] * (np.log(t[pos]) - np.log(q[pos]))).sum())


def jsd_np(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (np.asarray(p, dtype=float) + np.asarray(q, dtype=float))
    return 0.5 * kl_np(p, m) + 0.5 * kl_np(q, m)


def top1_agree(t: np.ndarray, q: np.ndarray) -> bool:
    return int(np.argmax(t)) == int(np.argmax(q))


# ----------------------------------------------------------------- run matrix --
@dataclass(frozen=True)
class RunSpec:
    """One training run: split × model × seed (+ ablation / data-size variants)."""

    split: str
    model: str
    seed: int
    ablation: str = "full"
    n_train: int | None = None      # None → the split's full train manifest

    @property
    def run_id(self) -> str:
        parts = [self.split, self.model]
        if self.ablation != "full":
            parts.append(f"abl_{self.ablation}")
        if self.n_train is not None:
            parts.append(f"n{self.n_train}")
        parts.append(f"s{self.seed}")
        return "__".join(parts)


def run_matrix(smoke: bool = False) -> list[RunSpec]:
    """The full 130-run matrix (design §3), or a 4-run smoke subset."""
    if smoke:
        return [
            RunSpec("S0", "simple", 0),
            RunSpec("S0", "nonlinear", 0),
            RunSpec("S0", "agnostic_simple", 0),
            RunSpec("G1", "simple", 0),
        ]
    runs = [RunSpec("S0", m, s) for m in S0_MODELS for s in SEEDS]                        # 20
    runs += [RunSpec(g, m, s) for g in G_SPLITS for m in MODELS for s in SEEDS]           # 60
    runs += [RunSpec("S0", m, s, ablation=a) for a in ABLATIONS for m in MODELS for s in SEEDS]  # 20
    runs += [RunSpec("S0", m, s, n_train=n) for n in DATA_SIZES for m in MODELS for s in SEEDS]  # 30
    return runs


def build_model(name: str, seed: int):
    """Fresh model, CPU/float64 (the model layer's native precision)."""
    import torch

    from npc_policy.learned import AgnosticPolicy, NonlinearPolicy, SimplePolicy

    torch.manual_seed(seed)
    if name == "simple":
        return SimplePolicy()
    if name == "nonlinear":
        return NonlinearPolicy()
    if name == "agnostic_simple":
        return AgnosticPolicy(SimplePolicy())
    if name == "agnostic_nonlinear":
        return AgnosticPolicy(NonlinearPolicy())
    raise ValueError(f"unknown model {name!r}")
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `python -m pytest tests/test_rq2_pipeline.py -q`
Expected: all `TestCommon` tests PASS. Also run the full suite `python -m pytest tests/ -q` — the pre-existing 49 tests must stay green.

- [ ] **Step 1.5: Commit**

```bash
git add experiments/rq2/__init__.py experiments/rq2/common.py tests/test_rq2_pipeline.py
git commit -m "rq2 2a: shared utilities (pool I/O, case assembly, metrics, run matrix)"
```

---

### Task 2: `gen_controlled.py` part 1 — samplers and rollout collector

**Files:**
- Create: `experiments/rq2/gen_controlled.py`
- Modify: `tests/test_rq2_pipeline.py` (append test class)

- [ ] **Step 2.1: Write the failing tests** (append to `tests/test_rq2_pipeline.py`)

```python
class TestGeneration:
    @pytest.fixture(scope="class")
    def gen(self):
        from experiments.rq2 import gen_controlled as g
        from npc_policy import load_world
        g_worlds = g.ensure_worlds()          # rq1 variants + arena_locked path dict
        world = load_world(g_worlds["full"])
        scorer = HandAuthoredScorer()
        return g, world, scorer

    def test_synthetic_location_labels_match_teacher(self, gen):
        g, world, scorer = gen
        s = g.SyntheticSampler(world, scorer, np.random.default_rng(1))
        for _ in range(20):
            c = s.location_case()
            assert 2 <= len(c.candidates) <= 8
            ids = [o.id for o in c.candidates]
            assert len(set(ids)) == len(ids)
            expect = scorer.distribution(
                Personality(c.personality), c.candidates,
                relations=c.candidate_history_features, level="location")
            np.testing.assert_allclose(c.target_distribution, expect, atol=1e-12)

    def test_synthetic_action_case_invariants(self, gen):
        g, world, scorer = gen
        s = g.SyntheticSampler(world, scorer, np.random.default_rng(2))
        for _ in range(20):
            c = s.action_case()
            assert c.decision_type == "action"
            assert c.recent_locations, "action case must carry its location context"
            assert c.recent_locations[-1].id == c.selected_location
            expect = scorer.distribution(
                Personality(c.personality), c.candidates,
                relations=c.candidate_history_features, level="action")
            np.testing.assert_allclose(c.target_distribution, expect, atol=1e-12)
            # feeds the model layer without tripping the action-context guard
            common.case_to_inputs(c)

    def test_rollout_labels_and_buffers(self, gen):
        g, world, scorer = gen
        recs = g.rollout_records("full", world, scorer, n_traj=1,
                                 rng=np.random.default_rng(3), rounds=6)
        assert len(recs) == 12                       # 6 location + 6 action cases
        for case, tags in recs:
            assert tags["source"] == "rollout" and tags["world"] == "full"
            expect = scorer.distribution(
                Personality(case.personality), case.candidates,
                relations=case.candidate_history_features,
                level=case.decision_type)
            np.testing.assert_allclose(case.target_distribution, expect, atol=1e-12)
            if case.decision_type == "action":
                assert case.recent_locations[-1].id == case.selected_location
        # very first location case: empty buffer -> zero relations
        first = recs[0][0]
        assert first.decision_type == "location"
        assert not first.recent_locations
        assert not first.candidate_history_features.rep.any()
```

- [ ] **Step 2.2: Run to verify failure**

Run: `python -m pytest tests/test_rq2_pipeline.py::TestGeneration -q`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError` (module missing).

- [ ] **Step 2.3: Implement** — create `experiments/rq2/gen_controlled.py`:

```python
"""Controlled-dataset generation for Study 2A (design §2).

Outputs (``data/rq2_controlled/``; smoke mode uses ``…_smoke/``):

  pool.jsonl              master pool with ``gen`` tags (id / source / world)
  test_<split>.jsonl      held-out sets (S0's is a random pool holdout)
  splits.json             per-split train/val case-id manifests + S0 test ids
  worlds/arena_locked.json  G6 top-up world variant
  meta.json               sizes, seed, frozen-teacher config hash

Run from ``code/``:  python -m experiments.rq2.gen_controlled [--smoke]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.rq1.gen_cases import build_worlds
from npc_policy import (
    DEFAULT_CONFIG,
    ControlledCase,
    DecisionController,
    HandAuthoredScorer,
    Option,
    Personality,
    RecentBuffer,
    load_world,
)
from npc_policy.relations import compute_relations
from npc_policy.schema import tag_index

from .common import (
    ALL_SPLITS,
    GEN_SEED,
    RQ1_WORLDS,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
    config_hash,
    dirs,
    write_pool,
)

ROLLOUT_VARIANTS = ("full", "celebration", "war_camp", "market_locked")
ROUNDS_PER_TRAJ = 50            # → 100 cases per trajectory (50 location + 50 action)

# Pool sizes. Master pool ≈ 170k so every filtered split retains ≥ 105k
# (TRAIN_SIZE + VAL_SIZE); the arena_locked block only feeds G6 (design §2).
FULL_SIZES = dict(
    n_syn_loc=49_000, n_syn_act=49_000,
    n_traj={"full": 210, "celebration": 70, "war_camp": 70, "market_locked": 70,
            "arena_locked": 300},
    train=TRAIN_SIZE, val=VAL_SIZE, n_test=TEST_SIZE, rounds=ROUNDS_PER_TRAJ,
)
SMOKE_SIZES = dict(
    n_syn_loc=700, n_syn_act=700,
    n_traj={"full": 3, "celebration": 1, "war_camp": 1, "market_locked": 1,
            "arena_locked": 4},
    train=1_200, val=200, n_test=300, rounds=ROUNDS_PER_TRAJ,
)


def base_id(option_id: str) -> str:
    """Family id: perturbed variants are ``<base>#p<k>``."""
    return option_id.split("#", 1)[0]


def ensure_worlds() -> dict[str, Path]:
    """RQ1 world variants (idempotent regeneration) + the arena-locked variant.

    Returns ``{variant_name: json_path}`` for every rollout world.
    """
    build_worlds()
    paths = {v: RQ1_WORLDS / f"{v}.json" for v in ROLLOUT_VARIANTS}
    base = json.loads(paths["full"].read_text(encoding="utf-8"))
    for loc in base["locations"]:
        if loc["id"] == "arena":
            loc["unlocked"] = False
    out = RQ1_WORLDS / "arena_locked.json"
    out.write_text(json.dumps(base, indent=2), encoding="utf-8")
    paths["arena_locked"] = out
    return paths


# ------------------------------------------------------------------ synthetic --
class SyntheticSampler:
    """Coverage sampler over the base world plus Gaussian-perturbed variants.

    Perturbed options get fresh ids (``tavern#p7``) so exact-repetition (``rep``)
    stays distinct from semantic similarity. ``mutate`` hooks let the targeted
    G-split generators edit the candidate list *before* relations and the teacher
    label are computed, so labels always match the stored features.
    """

    def __init__(self, world, scorer: HandAuthoredScorer,
                 rng: np.random.Generator, sigma: float = 0.1):
        self.world = world
        self.scorer = scorer
        self.rng = rng
        self.sigma = sigma
        self.locations = world.resolve()
        self._n = 0

    # -- option pools -------------------------------------------------------
    def _perturb(self, option: Option) -> Option:
        self._n += 1
        feats = np.clip(
            option.features + self.rng.normal(0.0, self.sigma, option.features.shape),
            0.0, 1.0,
        )
        return Option(id=f"{base_id(option.id)}#p{self._n}", features=feats,
                      level=option.level)

    def _variant(self, option: Option) -> Option:
        return self._perturb(option) if self.rng.random() < 0.5 else option

    def _random_location(self) -> Option:
        return self.locations[int(self.rng.integers(len(self.locations)))]

    def _personality(self) -> Personality:
        return Personality(self.rng.uniform(-1.0, 1.0, 5))

    def location_candidates(self, m: int) -> list[Option]:
        cands: list[Option] = []
        used: set[str] = set()
        while len(cands) < m:
            o = self._variant(self._random_location())
            if o.id in used:                       # duplicate base id → force variant
                o = self._perturb(o)
            used.add(o.id)
            cands.append(o)
        return cands

    # -- cases ----------------------------------------------------------------
    def location_case(self, personality: Personality | None = None,
                      m: int | None = None,
                      history: list[Option] | None = None,
                      mutate=None) -> ControlledCase:
        p = self._personality() if personality is None else personality
        m = int(self.rng.integers(2, 9)) if m is None else m
        cands = self.location_candidates(m)
        if mutate is not None:
            cands = mutate(cands)
        if history is None:
            k = int(self.rng.integers(0, DEFAULT_CONFIG.K_L + 1))
            history = [self._variant(self._random_location()) for _ in range(k)]
        rel = self._relations(cands, history, DEFAULT_CONFIG.K_L)
        target = self.scorer.distribution(p, cands, relations=rel, level="location")
        return ControlledCase(
            personality=p.vector, decision_type="location", candidates=cands,
            recent_locations=list(history), candidate_history_features=rel,
            target_distribution=target,
        )

    def action_case(self, personality: Personality | None = None,
                    at: Option | None = None,
                    history: list[Option] | None = None) -> ControlledCase:
        p = self._personality() if personality is None else personality
        loc = self._random_location() if at is None else at
        native = self.world.actions_at(base_id(loc.id))
        cands = []
        used: set[str] = set()
        for a in native:
            o = self._variant(a)
            if o.id in used:
                o = self._perturb(o)
            used.add(o.id)
            cands.append(o)
        if history is None:                        # same-location persistence: history
            k = int(self.rng.integers(0, DEFAULT_CONFIG.K_A + 1))   # from native actions
            history = [native[int(self.rng.integers(len(native)))] for _ in range(k)]
        rel = self._relations(cands, history, DEFAULT_CONFIG.K_A)
        target = self.scorer.distribution(p, cands, relations=rel, level="action")
        j = int(self.rng.integers(0, DEFAULT_CONFIG.K_L))           # older entries
        older = [self._variant(self._random_location()) for _ in range(j)]
        return ControlledCase(
            personality=p.vector, decision_type="action", candidates=cands,
            selected_location=loc.id, recent_locations=older + [loc],
            recent_actions_same_location=list(history),
            candidate_history_features=rel, target_distribution=target,
        )

    def _relations(self, cands, history, maxlen):
        if not history:
            return None
        buf = RecentBuffer(maxlen=maxlen)
        for o in history:
            buf.push(o)
        return compute_relations(cands, buf, self.scorer.config)


# -------------------------------------------------------------------- rollout --
def rollout_records(world_name: str, world, scorer: HandAuthoredScorer,
                    n_traj: int, rng: np.random.Generator,
                    rounds: int = ROUNDS_PER_TRAJ) -> list[tuple[ControlledCase, dict]]:
    """Trajectory cases: every decision's inputs + ``trace.P_rule`` (design §2)."""
    records: list[tuple[ControlledCase, dict]] = []
    for _ in range(n_traj):
        p = Personality(rng.uniform(-1.0, 1.0, 5))
        ctrl = DecisionController(
            scorer, config=scorer.config, mode="sample",
            rng=np.random.default_rng(int(rng.integers(2**32))),
        )
        for _ in range(rounds):
            locs = world.resolve()
            h_l = list(reversed(ctrl.H_L.recent_to_old()))     # oldest → newest
            d_loc = ctrl.choose_location(p, locs)
            records.append((
                ControlledCase(
                    personality=p.vector, decision_type="location",
                    candidates=locs, recent_locations=h_l,
                    candidate_history_features=d_loc.trace.relations,
                    target_distribution=d_loc.trace.P_rule,
                ),
                {"source": "rollout", "world": world_name},
            ))
            acts = world.actions_at(d_loc.option.id)
            h_l2 = list(reversed(ctrl.H_L.recent_to_old()))    # now ends with the choice
            h_a = list(reversed(ctrl.H_A.recent_to_old()))
            d_act = ctrl.choose_action(p, acts)
            records.append((
                ControlledCase(
                    personality=p.vector, decision_type="action",
                    candidates=acts, selected_location=d_loc.option.id,
                    recent_locations=h_l2, recent_actions_same_location=h_a,
                    candidate_history_features=d_act.trace.relations,
                    target_distribution=d_act.trace.P_rule,
                ),
                {"source": "rollout", "world": world_name},
            ))
    return records
```

- [ ] **Step 2.4: Run to verify pass**

Run: `python -m pytest tests/test_rq2_pipeline.py -q`
Expected: `TestCommon` + `TestGeneration` all PASS.

- [ ] **Step 2.5: Commit**

```bash
git add experiments/rq2/gen_controlled.py tests/test_rq2_pipeline.py
git commit -m "rq2 2a: synthetic sampler and rollout collector with frozen-teacher labels"
```

---

### Task 3: `gen_controlled.py` part 2 — split filters, targeted test sets, manifests, CLI

**Files:**
- Modify: `experiments/rq2/gen_controlled.py` (append)
- Modify: `tests/test_rq2_pipeline.py` (append test class)

- [ ] **Step 3.1: Write the failing tests** (append)

```python
class TestSplits:
    @pytest.fixture(scope="class")
    def tiny_dataset(self, tmp_path_factory):
        """End-to-end tiny generation into a temp dir (same code path as the CLI)."""
        from experiments.rq2 import gen_controlled as g
        out = tmp_path_factory.mktemp("rq2data")
        sizes = dict(n_syn_loc=120, n_syn_act=120,
                     n_traj={"full": 1, "celebration": 1, "war_camp": 1,
                             "market_locked": 1, "arena_locked": 2},
                     train=60, val=15, n_test=12, rounds=10)
        g.generate(sizes, out, seed=123)
        return g, out

    def test_outputs_exist(self, tiny_dataset):
        g, out = tiny_dataset
        assert (out / "pool.jsonl").exists()
        assert (out / "splits.json").exists()
        assert (out / "meta.json").exists()
        for split in common.ALL_SPLITS:
            assert (out / f"test_{split}.jsonl").exists()
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        assert meta["config_hash"] == common.config_hash()

    def test_manifests_disjoint_and_sized(self, tiny_dataset):
        g, out = tiny_dataset
        manifest = json.loads((out / "splits.json").read_text(encoding="utf-8"))
        s0_test = set(manifest["s0_test_ids"])
        for split, part in manifest["splits"].items():
            train, val = part["train"], part["val"]
            assert len(train) == 60 and len(val) == 15
            assert not set(train) & set(val)
            assert not set(train) & s0_test and not set(val) & s0_test

    def test_train_filters_hold(self, tiny_dataset):
        g, out = tiny_dataset
        records = common.read_pool(out / "pool.jsonl")
        by_id = {t["id"]: (c, t) for c, t in records}
        manifest = json.loads((out / "splits.json").read_text(encoding="utf-8"))
        for split in common.ALL_SPLITS:
            for cid in manifest["splits"][split]["train"]:
                case, tags = by_id[cid]
                assert g.TRAIN_FILTERS[split](case, tags), f"{split}: {cid}"
        # arena_locked cases may appear in G6 only
        for split in common.ALL_SPLITS:
            if split == "G6":
                continue
            for cid in manifest["splits"][split]["train"]:
                assert by_id[cid][1]["world"] != "arena_locked"

    def test_targeted_test_sets_satisfy_conditions(self, tiny_dataset):
        g, out = tiny_dataset
        checks = {
            "G1": lambda c: c.personality[0] > 0.5 and c.personality[1] < -0.5,
            "G2": lambda c: c.decision_type == "location" and any(
                o.tag("risk") > 0.6 and o.tag("privacy") > 0.6 for o in c.candidates),
            "G3": lambda c: c.decision_type == "location" and len(c.candidates) in (2, 8),
            "G5": lambda c: g.g5_has_3run(c),
            "G6": lambda c: g.g6_touches_arena(c),
        }
        for split, ok in checks.items():
            cases = [c for c, _ in common.read_pool(out / f"test_{split}.jsonl")]
            assert len(cases) == 12
            assert all(ok(c) for c in cases), split
        g4 = common.read_pool(out / "test_G4.jsonl")
        assert all(t["world"] in ("celebration", "war_camp", "market_locked")
                   for _, t in g4)

    def test_test_labels_match_teacher(self, tiny_dataset):
        g, out = tiny_dataset
        scorer = HandAuthoredScorer()
        for split in ("G1", "G2", "G5", "G6"):
            for c, _ in common.read_pool(out / f"test_{split}.jsonl"):
                expect = scorer.distribution(
                    Personality(c.personality), c.candidates,
                    relations=c.candidate_history_features, level=c.decision_type)
                np.testing.assert_allclose(c.target_distribution, expect, atol=1e-12)

    def test_predicates_on_crafted_cases(self):
        from experiments.rq2 import gen_controlled as g
        inside = _mk_case(personality=[0.6, -0.7, 0, 0, 0])
        outside = _mk_case(personality=[0.6, 0.0, 0, 0, 0])
        assert g.g1_region(inside) and not g.g1_region(outside)
        risky = _mk_case()
        risky.candidates[0] = _loc("den", risk=0.8, privacy=0.9)
        assert g.g2_combo(risky) and not g.g2_combo(_mk_case())
        rep3 = _mk_case(recent_locs=[_loc("a"), _loc("a"), _loc("a")])
        rep2 = _mk_case(recent_locs=[_loc("a"), _loc("a"), _loc("b")])
        assert g.g5_has_3run(rep3) and not g.g5_has_3run(rep2)
        arena = _mk_case()
        arena.candidates[0] = _loc("arena", risk=0.8)
        assert g.g6_touches_arena(arena)
        pert = _mk_case()
        pert.candidates[0] = _loc("arena#p9", risk=0.8)
        assert g.g6_touches_arena(pert), "perturbed arena variants count as arena"
        assert not g.g6_touches_arena(_mk_case())
```

- [ ] **Step 3.2: Run to verify failure**

Run: `python -m pytest tests/test_rq2_pipeline.py::TestSplits -q`
Expected: FAIL with `AttributeError` (`generate`, `TRAIN_FILTERS`, … missing).

- [ ] **Step 3.3: Implement** — append to `experiments/rq2/gen_controlled.py`:

```python
# ------------------------------------------------------------- split filters --
def _max_run(ids: list[str]) -> int:
    best = run = 0
    prev = None
    for x in ids:
        run = run + 1 if x == prev else 1
        prev = x
        best = max(best, run)
    return best


def g1_region(case: ControlledCase) -> bool:
    """Excluded personality region: O > 0.5 ∧ C < −0.5 (research spec §6)."""
    return case.personality[0] > 0.5 and case.personality[1] < -0.5


def g2_combo(case: ControlledCase) -> bool:
    """Location candidate with risk > 0.6 ∧ privacy > 0.6 present."""
    return case.decision_type == "location" and any(
        o.tag("risk") > 0.6 and o.tag("privacy") > 0.6 for o in case.candidates
    )


def g3_train_ok(case: ControlledCase) -> bool:
    """Train on location sets of 3–6 only; action cases are unaffected."""
    return case.decision_type != "location" or 3 <= len(case.candidates) <= 6


def g5_has_3run(case: ControlledCase) -> bool:
    """Three consecutive same-family entries in the relevant same-type buffer."""
    buf = (case.recent_locations if case.decision_type == "location"
           else case.recent_actions_same_location)
    return _max_run([base_id(o.id) for o in buf]) >= 3


def g6_touches_arena(case: ControlledCase) -> bool:
    ids = [base_id(o.id) for o in case.candidates]
    ids += [base_id(o.id) for o in case.recent_locations]
    return "arena" in ids or case.selected_location == "arena"


def _core(tags: dict) -> bool:
    """arena_locked top-up cases feed G6 only."""
    return tags.get("world") != "arena_locked"


TRAIN_FILTERS = {
    "S0": lambda c, t: _core(t),
    "G1": lambda c, t: _core(t) and not g1_region(c),
    "G2": lambda c, t: _core(t) and not g2_combo(c),
    "G3": lambda c, t: _core(t) and g3_train_ok(c),
    "G4": lambda c, t: t.get("world") == "full",
    "G5": lambda c, t: _core(t) and not g5_has_3run(c),
    "G6": lambda c, t: not g6_touches_arena(c),
}


# ------------------------------------------------------- targeted test sets --
def targeted_records(split: str, sampler: SyntheticSampler, worlds: dict,
                     scorer: HandAuthoredScorer, n: int,
                     rng: np.random.Generator, rounds: int) -> list[tuple[ControlledCase, dict]]:
    """Generate ``n`` cases satisfying ``split``'s held-out condition (design §2)."""
    tag = {"source": "targeted", "world": "full"}
    out: list[tuple[ControlledCase, dict]] = []

    if split == "S0":
        raise ValueError("S0's test set is a pool holdout, not targeted")

    if split == "G1":
        def region_p():
            v = rng.uniform(-1.0, 1.0, 5)
            v[0] = rng.uniform(0.5 + 1e-9, 1.0)
            v[1] = rng.uniform(-1.0, -0.5 - 1e-9)
            return Personality(v)
        while len(out) < n:
            case = (sampler.location_case(personality=region_p())
                    if rng.random() < 0.5 else sampler.action_case(personality=region_p()))
            out.append((case, dict(tag)))

    elif split == "G2":
        risk_i = tag_index("location", "risk")
        priv_i = tag_index("location", "privacy")

        def spike(cands: list[Option]) -> list[Option]:
            k = int(rng.integers(len(cands)))
            f = cands[k].features.copy()
            f[risk_i] = rng.uniform(0.65, 1.0)
            f[priv_i] = rng.uniform(0.65, 1.0)
            cands[k] = Option(id=cands[k].id, features=f, level="location")
            return cands
        while len(out) < n:
            out.append((sampler.location_case(mutate=spike), dict(tag)))

    elif split == "G3":
        while len(out) < n:
            m = 2 if rng.random() < 0.5 else 8
            out.append((sampler.location_case(m=m), dict(tag)))

    elif split == "G4":
        variants = ("celebration", "war_camp", "market_locked")
        i = 0
        while len(out) < n:
            name = variants[i % 3]
            i += 1
            world = load_world(worlds[name])
            recs = rollout_records(name, world, scorer, n_traj=1, rng=rng, rounds=rounds)
            for case, t in recs:
                t["source"] = "targeted"
                out.append((case, t))
        out = out[:n]

    elif split == "G5":
        while len(out) < n:
            if rng.random() < 0.5:
                x = sampler._variant(sampler._random_location())
                out.append((sampler.location_case(history=[x, x, x]), dict(tag)))
            else:
                loc = sampler._random_location()
                a = sampler.world.actions_at(base_id(loc.id))[0]
                out.append((sampler.action_case(at=loc, history=[a, a, a]), dict(tag)))

    elif split == "G6":
        arena = sampler.world.effective_location("arena")

        def force_arena(cands: list[Option]) -> list[Option]:
            if not any(base_id(o.id) == "arena" for o in cands):
                cands[int(rng.integers(len(cands)))] = arena
            return cands
        while len(out) < n:
            case = (sampler.location_case(mutate=force_arena)
                    if rng.random() < 0.5 else sampler.action_case(at=arena))
            out.append((case, dict(tag)))

    else:
        raise ValueError(f"unknown split {split!r}")
    return out


# ------------------------------------------------------------------ assembly --
def generate(sizes: dict, out_dir: Path, seed: int = GEN_SEED) -> dict:
    """Full generation pass; returns the meta dict (also written to meta.json)."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    worlds = ensure_worlds()
    scorer = HandAuthoredScorer()            # DEFAULT_CONFIG — the frozen teacher
    base_world = load_world(worlds["full"])

    # -- master pool ---------------------------------------------------------
    rng = np.random.default_rng([seed, 0])
    sampler = SyntheticSampler(base_world, scorer, rng)
    records: list[tuple[ControlledCase, dict]] = []
    for _ in range(sizes["n_syn_loc"]):
        records.append((sampler.location_case(), {"source": "synthetic", "world": "full"}))
    for _ in range(sizes["n_syn_act"]):
        records.append((sampler.action_case(), {"source": "synthetic", "world": "full"}))
    for w_idx, (name, n_traj) in enumerate(sorted(sizes["n_traj"].items())):
        world = load_world(worlds[name])
        records += rollout_records(name, world, scorer, n_traj,
                                   np.random.default_rng([seed, 1, w_idx]),
                                   rounds=sizes["rounds"])
    for i, (_, tags) in enumerate(records):
        tags["id"] = f"{tags['source'][:4]}-{i:07d}"

    # -- S0 holdout + split manifests -----------------------------------------
    core_ids = [t["id"] for c, t in records if _core(t)]
    hold_rng = np.random.default_rng([seed, 2])
    hold_rng.shuffle(core_ids)
    s0_test_ids = set(core_ids[: sizes["n_test"]])

    by_id = {t["id"]: (c, t) for c, t in records}
    splits: dict[str, dict] = {}
    for k, split in enumerate(ALL_SPLITS):
        elig = [t["id"] for c, t in records
                if t["id"] not in s0_test_ids and TRAIN_FILTERS[split](c, t)]
        need = sizes["train"] + sizes["val"]
        if len(elig) < need:
            raise RuntimeError(
                f"{split}: filtered pool has {len(elig)} cases, needs {need}; "
                "increase generation sizes"
            )
        srng = np.random.default_rng([seed, 3, k])
        srng.shuffle(elig)
        splits[split] = {"train": elig[: sizes["train"]],
                         "val": elig[sizes["train"]: need]}

    # -- write ----------------------------------------------------------------
    write_pool(out_dir / "pool.jsonl", records)
    write_pool(out_dir / "test_S0.jsonl",
               [by_id[i] for i in core_ids[: sizes["n_test"]]])
    trng = np.random.default_rng([seed, 4])
    test_counts = {"S0": sizes["n_test"]}
    for split in ALL_SPLITS[1:]:
        recs = targeted_records(split, sampler, worlds, scorer,
                                sizes["n_test"], trng, sizes["rounds"])
        write_pool(out_dir / f"test_{split}.jsonl", recs)
        test_counts[split] = len(recs)
    (out_dir / "splits.json").write_text(
        json.dumps({"s0_test_ids": sorted(s0_test_ids), "splits": splits}),
        encoding="utf-8",
    )
    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "config_hash": config_hash(),
        "pool_cases": len(records),
        "per_split": {s: {"train": len(splits[s]["train"]), "val": len(splits[s]["val"]),
                          "test": test_counts[s]} for s in ALL_SPLITS},
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Generate the Study 2A controlled dataset")
    ap.add_argument("--smoke", action="store_true", help="small isolated end-to-end pass")
    args = ap.parse_args(argv)
    out_dir, _ = dirs(args.smoke)
    meta = generate(SMOKE_SIZES if args.smoke else FULL_SIZES, out_dir)
    print(f"written: {out_dir}")
    print(f"  pool cases: {meta['pool_cases']}, teacher config: {meta['config_hash'][:12]}…")
    for s, n in meta["per_split"].items():
        print(f"  {s}: train {n['train']} / val {n['val']} / test {n['test']}")
    print(f"  elapsed: {meta['elapsed_s']}s")


if __name__ == "__main__":
    main()
```

Note for the implementer: `np.random.default_rng` accepts a list seed (`[seed, k]`) — this is the documented SeedSequence spawn pattern, deterministic and independent per stream.

- [ ] **Step 3.4: Run to verify pass**

Run: `python -m pytest tests/test_rq2_pipeline.py -q`
Expected: all PASS (the tiny end-to-end fixture takes a few seconds).

- [ ] **Step 3.5: Sanity-run the smoke CLI**

Run: `python -m experiments.rq2.gen_controlled --smoke`
Expected: prints pool/split/test counts; `data/rq2_controlled_smoke/` contains `pool.jsonl`, `splits.json`, `meta.json`, 7 `test_*.jsonl`.

- [ ] **Step 3.6: Commit**

```bash
git add experiments/rq2/gen_controlled.py tests/test_rq2_pipeline.py
git commit -m "rq2 2a: split filters, targeted test sets, manifests, generation CLI"
```

---

### Task 4: `train.py` — resumable training runs

**Files:**
- Create: `experiments/rq2/train.py`
- Modify: `tests/test_rq2_pipeline.py` (append test class)

- [ ] **Step 4.1: Write the failing tests** (append)

```python
class TestTraining:
    @pytest.fixture(scope="class")
    def tiny_cases(self):
        """Representable teacher (bilinear, N temperature off) — design §6 item 3."""
        from experiments.rq2 import gen_controlled as g
        from npc_policy import load_world
        from npc_policy.config import LevelParams
        worlds = g.ensure_worlds()
        world = load_world(worlds["full"])
        cfg = ScorerConfig(
            base_form="bilinear",
            location=LevelParams(tau_0=0.9, lambda_N=0.0),
            action=LevelParams(lambda_N=0.0),
        )
        scorer = HandAuthoredScorer(config=cfg)
        s = g.SyntheticSampler(world, scorer, np.random.default_rng(7))
        cases = ([s.location_case() for _ in range(150)]
                 + [s.action_case() for _ in range(150)])
        return cases[:240], cases[240:]        # train, val

    def test_simple_converges_on_representable_teacher(self, tiny_cases):
        import torch
        from experiments.rq2 import train as tr
        train_cases, val_cases = tiny_cases
        result, state = tr.train_one(
            common.RunSpec("S0", "simple", 0), train_cases, val_cases,
            device=torch.device("cpu"),
            lr=0.05, batch_size=60, max_epochs=500, patience=500,
        )
        # representable target (bilinear teacher, N temperature off) + overdetermined
        # (240 cases ≫ 228 params) → near-zero KL certifies representability, not
        # interpolation (dev_log.md 2026-07-09 lesson). Adam is slower than the
        # LBFGS certificate in test_learned.py, hence the looser 1e-3 bar.
        assert result["best_val_kl"] < 1e-3
        assert result["dtype"] == "float64"
        assert all(v.dtype == torch.float64 for v in state.values())

    def test_early_stopping_triggers(self, tiny_cases):
        import torch
        from experiments.rq2 import train as tr
        train_cases, val_cases = tiny_cases
        result, _ = tr.train_one(
            common.RunSpec("S0", "simple", 0), train_cases[:40], val_cases[:20],
            device=torch.device("cpu"),
            lr=0.0, batch_size=40, max_epochs=100, patience=5,
        )
        assert result["epochs_run"] == 6       # epoch 0 sets best; 5 stale epochs; stop

    def test_run_all_resumes(self, tiny_cases, tmp_path):
        import torch
        from experiments.rq2 import train as tr
        train_cases, val_cases = tiny_cases
        specs = [common.RunSpec("S0", "simple", 0), common.RunSpec("S0", "simple", 1)]
        calls = []

        def fake_train(spec, tc, vc, device, **kw):
            calls.append(spec.run_id)
            return ({"run_id": spec.run_id, "best_val_kl": 0.5, "dtype": "float64"},
                    common.build_model(spec.model, spec.seed).state_dict())

        loader = lambda spec: (train_cases[:10], val_cases[:10])
        tr.run_all(specs, loader, tmp_path, torch.device("cpu"), train_fn=fake_train)
        assert sorted(calls) == [s.run_id for s in specs]
        assert (tmp_path / "runs" / "S0__simple__s0.json").exists()
        assert (tmp_path / "models" / "S0__simple__s0.pt").exists()
        calls.clear()
        tr.run_all(specs, loader, tmp_path, torch.device("cpu"), train_fn=fake_train)
        assert calls == []                     # everything skipped on re-run

    def test_agnostic_ignores_personality_after_training(self, tiny_cases):
        import torch
        from experiments.rq2 import train as tr
        from npc_policy.learned import predict_distribution
        train_cases, val_cases = tiny_cases
        _, state = tr.train_one(
            common.RunSpec("S0", "agnostic_simple", 0), train_cases[:60], val_cases[:20],
            device=torch.device("cpu"), lr=0.05, batch_size=60, max_epochs=20, patience=20,
        )
        model = common.build_model("agnostic_simple", 0)
        model.load_state_dict(state)
        case = train_cases[0]
        d1 = predict_distribution(model, Personality(np.array([1.0, -1, 1, -1, 1])),
                                  case.candidates, "location",
                                  relations=case.candidate_history_features)
        d2 = predict_distribution(model, Personality(np.zeros(5)),
                                  case.candidates, "location",
                                  relations=case.candidate_history_features)
        np.testing.assert_allclose(d1, d2, atol=1e-12)
```

- [ ] **Step 4.2: Run to verify failure**

Run: `python -m pytest tests/test_rq2_pipeline.py::TestTraining -q`
Expected: FAIL (`experiments.rq2.train` missing).

- [ ] **Step 4.3: Implement** — create `experiments/rq2/train.py`:

```python
"""Study 2A training runs — resumable, device-agnostic (design §3).

One run = split × model × seed (``common.run_matrix``, 130 runs full / 4 smoke).
Each run writes ``results/rq2/runs/<run_id>.json`` and ``models/<run_id>.pt``
on completion; re-invoking the command skips finished ids, so an interrupted
session continues where it left off.

Run from ``code/``:  python -m experiments.rq2.train [--smoke] [--device auto|cpu|cuda]
                                                     [--only PREFIX]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from npc_policy import ControlledCase
from npc_policy.learned import PolicyBatch, kl_loss

from .common import (
    RunSpec,
    build_model,
    case_to_inputs,
    config_hash,
    dirs,
    read_pool,
    run_matrix,
)


def pick_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def batch_to(b: PolicyBatch, device: torch.device, dtype: torch.dtype) -> PolicyBatch:
    """Move a batch to the training device/dtype (design §3: f32 CUDA, f64 CPU)."""
    return replace(
        b,
        p=b.p.to(device=device, dtype=dtype),
        d=b.d.to(device=device),
        ctx=b.ctx.to(device=device, dtype=dtype),
        cand=b.cand.to(device=device, dtype=dtype),
        rel=b.rel.to(device=device, dtype=dtype),
        mask=b.mask.to(device=device),
        target=None if b.target is None else b.target.to(device=device, dtype=dtype),
    )


def _mean_val_kl(model: torch.nn.Module, val_batches: list[PolicyBatch]) -> float:
    model.eval()
    tot = n = 0
    with torch.no_grad():
        for b in val_batches:
            tot += float(kl_loss(model(b), b.target, b.mask)) * b.mask.shape[0]
            n += b.mask.shape[0]
    return tot / n


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
) -> tuple[dict, dict]:
    """Train one run; returns (result dict, best float64-CPU state_dict)."""
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    inputs = [case_to_inputs(c, spec.ablation) for c in train_cases]
    val_inputs = [case_to_inputs(c, spec.ablation) for c in val_cases]
    val_batches = [
        batch_to(PolicyBatch.from_cases(val_inputs[i:i + 512]), device, dtype)
        for i in range(0, len(val_inputs), 512)
    ]

    model = build_model(spec.model, spec.seed).to(device=device, dtype=dtype)
    weight_decay = 1e-4 if "nonlinear" in spec.model else 0.0
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    shuffle_rng = np.random.default_rng(spec.seed)

    t0 = time.time()
    best_kl, best_state, best_epoch = float("inf"), None, -1
    epochs_run = 0
    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        model.train()
        order = shuffle_rng.permutation(len(inputs))
        for lo in range(0, len(order), batch_size):
            chunk = [inputs[int(j)] for j in order[lo:lo + batch_size]]
            batch = batch_to(PolicyBatch.from_cases(chunk), device, dtype)
            opt.zero_grad()
            loss = kl_loss(model(batch), batch.target, batch.mask)
            loss.backward()
            opt.step()
        val_kl = _mean_val_kl(model, val_batches)
        if val_kl < best_kl - 1e-6:
            best_kl, best_epoch = val_kl, epoch
            best_state = {k: v.detach().to("cpu", torch.float64).clone()
                          for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

    result = {
        "run_id": spec.run_id,
        **asdict(spec),
        "best_val_kl": best_kl,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "wall_time_s": round(time.time() - t0, 1),
        "device": device.type,
        "dtype": "float32" if dtype == torch.float32 else "float64",
        "n_train_cases": len(train_cases),
        "config_hash": config_hash(),
    }
    return result, best_state


def run_all(specs, load_cases, results_dir: Path, device: torch.device,
            train_fn=train_one, **train_kw) -> None:
    """Run every spec whose result file does not exist yet (resume = skip)."""
    runs_dir = results_dir / "runs"
    models_dir = results_dir / "models"
    runs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in specs if not (runs_dir / f"{s.run_id}.json").exists()]
    print(f"{len(specs)} runs, {len(specs) - len(todo)} already done, {len(todo)} to go "
          f"(device: {device.type})")
    for i, spec in enumerate(todo, 1):
        train_cases, val_cases = load_cases(spec)
        result, state = train_fn(spec, train_cases, val_cases, device, **train_kw)
        torch.save({"model": spec.model, "state_dict": state},
                   models_dir / f"{spec.run_id}.pt")
        (runs_dir / f"{spec.run_id}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        print(f"[{i}/{len(todo)}] {spec.run_id}: val KL {result['best_val_kl']:.4g} "
              f"({result['epochs_run']} epochs, {result['wall_time_s']}s)")


def make_loader(data_dir: Path):
    """Case loader over the pool + manifests. Loads the pool once, lazily."""
    state: dict = {}

    def load_cases(spec: RunSpec):
        if "by_id" not in state:
            print("loading pool …")
            state["by_id"] = {t["id"]: c for c, t in read_pool(data_dir / "pool.jsonl")}
            state["splits"] = json.loads(
                (data_dir / "splits.json").read_text(encoding="utf-8"))["splits"]
        part = state["splits"][spec.split]
        train_ids = part["train"]
        if spec.n_train is not None:
            train_ids = train_ids[: spec.n_train]      # nested subsets by construction
        by_id = state["by_id"]
        return [by_id[i] for i in train_ids], [by_id[i] for i in part["val"]]

    return load_cases


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train all Study 2A runs (resumable)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--only", default=None,
                    help="only run ids starting with this prefix (debugging)")
    args = ap.parse_args(argv)
    data_dir, results_dir = dirs(args.smoke)
    if not (data_dir / "pool.jsonl").exists():
        raise SystemExit(f"dataset missing: {data_dir} — run gen_controlled first")
    device = pick_device(args.device)
    specs = run_matrix(smoke=args.smoke)
    if args.only:
        specs = [s for s in specs if s.run_id.startswith(args.only)]
    kw = dict(max_epochs=8) if args.smoke else {}
    run_all(specs, make_loader(data_dir), results_dir, device, **kw)
    print(f"done: {results_dir / 'runs'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.4: Run to verify pass**

Run: `python -m pytest tests/test_rq2_pipeline.py -q`
Expected: all PASS (the convergence test takes tens of seconds on CPU).

- [ ] **Step 4.5: Sanity-run the smoke CLI** (needs Task 3's smoke dataset)

Run: `python -m experiments.rq2.train --smoke`
Expected: 4 runs train (8 epochs each), result JSONs + weights appear under `results/rq2_smoke/`. Re-run the same command: prints `4 runs, 4 already done, 0 to go`.

- [ ] **Step 4.6: Commit**

```bash
git add experiments/rq2/train.py tests/test_rq2_pipeline.py
git commit -m "rq2 2a: resumable training runs with device-aware dtype"
```

---

### Task 5: `run_2a.py` — metrics, table, figures, diagnostics

**Files:**
- Create: `experiments/rq2/run_2a.py`
- Modify: `tests/test_rq2_pipeline.py` (append test class)

- [ ] **Step 5.1: Write the failing tests** (append)

```python
class TestRun2A:
    def test_eval_cases_uniform_pin(self):
        from experiments.rq2 import run_2a
        from npc_policy.learned import UniformBaseline
        case = _mk_case(n_cand=2, target=[1.0, 0.0])
        rows = run_2a.eval_cases(UniformBaseline(), [case])
        assert len(rows) == 1
        assert rows[0]["decision_type"] == "location"
        assert rows[0]["kl"] == pytest.approx(np.log(2.0))
        assert rows[0]["top1"] == 1               # argmax tie resolves to index 0

    def test_eval_cases_applies_ablation(self):
        from experiments.rq2 import run_2a
        from npc_policy.learned import SimplePolicy
        import torch
        from npc_policy.relations import Relations
        torch.manual_seed(0)
        model = SimplePolicy()
        with torch.no_grad():
            model.w += torch.randn_like(model.w)
        case = _mk_case(n_cand=3)
        case.candidate_history_features = Relations(
            np.array([0.9, 0.0, 0.0]), np.array([0.8, 0.1, 0.1]), np.array([0.2, 0.9, 0.9]))
        r_full = run_2a.eval_cases(model, [case], ablation="full")[0]
        r_none = run_2a.eval_cases(model, [case], ablation="none")[0]
        assert r_full["kl"] != pytest.approx(r_none["kl"])

    def test_aggregate_mean_std(self):
        from experiments.rq2 import run_2a
        per_run = [
            {"split": "S0", "model": "simple", "ablation": "full", "n_train": None,
             "seed": s, "eval_split": "S0", "decision_type": "all",
             "kl": v, "jsd": v / 2, "top1": 1.0, "n_cases": 10}
            for s, v in enumerate([0.1, 0.2, 0.3])
        ]
        table = run_2a.aggregate(per_run)
        [row] = table
        assert row["kl_mean"] == pytest.approx(0.2)
        assert row["kl_std"] == pytest.approx(np.std([0.1, 0.2, 0.3]))
        assert row["n_seeds"] == 3

    def test_load_student_roundtrip(self, tmp_path):
        import torch
        from experiments.rq2 import run_2a
        from npc_policy.learned import predict_distribution
        model = common.build_model("nonlinear", seed=3)
        (tmp_path / "models").mkdir()
        torch.save({"model": "nonlinear", "state_dict": model.state_dict()},
                   tmp_path / "models" / "X.pt")
        back = run_2a.load_student(tmp_path, "X")
        case = _mk_case(n_cand=4)
        p = Personality(np.array([0.3, -0.2, 0.5, 0.0, -0.9]))
        np.testing.assert_allclose(
            predict_distribution(model, p, case.candidates, "location"),
            predict_distribution(back, p, case.candidates, "location"), atol=1e-12)
```

- [ ] **Step 5.2: Run to verify failure**

Run: `python -m pytest tests/test_rq2_pipeline.py::TestRun2A -q` — Expected: FAIL (module missing).

- [ ] **Step 5.3: Implement** — create `experiments/rq2/run_2a.py`:

```python
"""Study 2A metrics: main table, figures, diagnostics (design §4).

Reads the run results and models written by ``train.py`` plus the test sets from
``gen_controlled.py``; writes CSVs and figures to ``results/rq2/``. G-split models
are evaluated on their own test set and on S0 (no-general-degradation check);
the S0 main models are evaluated on EVERY test set, so each G-split's
filtered-vs-S0-trained difference isolates the exclusion effect from the test
set's composition. All metrics per decision type and combined ("all").

Run from ``code/``:  python -m experiments.rq2.run_2a [--smoke]
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
from npc_policy import ControlledCase
from npc_policy.learned import PolicyBatch, UniformBaseline

from .common import (
    ALL_SPLITS,
    DATA_SIZES,
    G_SPLITS,
    build_model,
    case_to_inputs,
    dirs,
    jsd_np,
    kl_np,
    read_pool,
    top1_agree,
)

FAMILY_COLORS = {"simple": "#0072B2", "nonlinear": "#E69F00",
                 "agnostic_simple": "#999999", "agnostic_nonlinear": "#CC79A7",
                 "uniform": "#CCCCCC"}


def load_student(results_dir: Path, run_id: str) -> torch.nn.Module:
    payload = torch.load(results_dir / "models" / f"{run_id}.pt", weights_only=False)
    model = build_model(payload["model"], seed=0)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def eval_cases(model: torch.nn.Module, cases: list[ControlledCase],
               ablation: str = "full", chunk: int = 512) -> list[dict]:
    """Per-case KL/JSD/top-1 rows; inputs get the run's ablation transform."""
    rows: list[dict] = []
    model.eval()
    for lo in range(0, len(cases), chunk):
        part = cases[lo:lo + chunk]
        batch = PolicyBatch.from_cases([case_to_inputs(c, ablation) for c in part])
        with torch.no_grad():
            probs = model(batch).exp().numpy()
        for i, c in enumerate(part):
            q = probs[i, : len(c.candidates)]
            t = np.asarray(c.target_distribution, dtype=float)
            rows.append({
                "decision_type": c.decision_type,
                "kl": kl_np(t, q), "jsd": jsd_np(t, q),
                "top1": int(top1_agree(t, q)),
            })
    return rows


def _summarise(rows: list[dict]) -> list[dict]:
    """Collapse per-case rows to per-decision-type (+ "all") means."""
    out = []
    for dt in ("location", "action", "all"):
        sel = rows if dt == "all" else [r for r in rows if r["decision_type"] == dt]
        if not sel:
            continue
        out.append({"decision_type": dt, "n_cases": len(sel),
                    "kl": float(np.mean([r["kl"] for r in sel])),
                    "jsd": float(np.mean([r["jsd"] for r in sel])),
                    "top1": float(np.mean([r["top1"] for r in sel]))})
    return out


def aggregate(per_run: list[dict]) -> list[dict]:
    """Mean ± std across seeds for each (split, model, ablation, n_train,
    eval_split, decision_type) group."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in per_run:
        key = (r["split"], r["model"], r["ablation"], r["n_train"],
               r["eval_split"], r["decision_type"])
        groups[key].append(r)
    table = []
    for (split, model, abl, n_train, eval_split, dt), rs in sorted(
            groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        table.append({
            "split": split, "model": model, "ablation": abl,
            "n_train": n_train, "eval_split": eval_split, "decision_type": dt,
            "n_cases": rs[0]["n_cases"], "n_seeds": len(rs),
            "kl_mean": float(np.mean([r["kl"] for r in rs])),
            "kl_std": float(np.std([r["kl"] for r in rs])),
            "jsd_mean": float(np.mean([r["jsd"] for r in rs])),
            "jsd_std": float(np.std([r["jsd"] for r in rs])),
            "top1_mean": float(np.mean([r["top1"] for r in rs])),
            "top1_std": float(np.std([r["top1"] for r in rs])),
        })
    return table


def _lookup(table, **want):
    """Rows of the aggregated table matching all given fields."""
    return [row for row in table if all(row[k] == v for k, v in want.items())]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Study 2A metrics and figures")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    setup_style()
    data_dir, results_dir = dirs(args.smoke)
    runs_dir = results_dir / "runs"
    run_files = sorted(runs_dir.glob("*.json"))
    if not run_files:
        raise SystemExit(f"no run results in {runs_dir} — run train first")

    tests = {s: [c for c, _ in read_pool(data_dir / f"test_{s}.jsonl")]
             for s in ALL_SPLITS if (data_dir / f"test_{s}.jsonl").exists()}

    # -- per-run evaluation ----------------------------------------------------
    per_run: list[dict] = []
    for f in run_files:
        meta = json.loads(f.read_text(encoding="utf-8"))
        model = load_student(results_dir, meta["run_id"])
        if (meta["split"] == "S0" and meta["ablation"] == "full"
                and meta["n_train"] is None):
            eval_splits = list(tests)     # S0 main runs: every test set — the
        elif meta["split"] == "S0":       # S0-trained reference isolates each
            eval_splits = ["S0"]          # G-split's exclusion effect from its
        else:                             # test-set composition
            eval_splits = [meta["split"], "S0"]
        for split in eval_splits:
            for s in _summarise(eval_cases(model, tests[split], meta["ablation"])):
                per_run.append({**s, "split": meta["split"], "model": meta["model"],
                                "ablation": meta["ablation"], "n_train": meta["n_train"],
                                "seed": meta["seed"], "eval_split": split})
        print(f"evaluated {meta['run_id']}")
    # untrained floor, once per split
    for split, cases in tests.items():
        for s in _summarise(eval_cases(UniformBaseline(), cases)):
            per_run.append({**s, "split": split, "model": "uniform", "ablation": "full",
                            "n_train": None, "seed": 0, "eval_split": split})

    table = aggregate(per_run)
    write_csv(results_dir / "main_table.csv", list(table[0].keys()),
              [list(r.values()) for r in table])

    # -- figure: simple-vs-nonlinear gap across splits ---------------------------
    mains = {"S0": "S0"} | {g: g for g in G_SPLITS}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(mains))
    series = [("simple", "simple", False), ("nonlinear", "nonlinear", False),
              ("simple (S0-trained)", "simple", True),
              ("nonlinear (S0-trained)", "nonlinear", True)]
    for i, (label, fam, s0_trained) in enumerate(series):
        ys, es = [], []
        for split in mains:
            rows = _lookup(table, split="S0" if s0_trained else split, model=fam,
                           ablation="full", n_train=None, eval_split=split,
                           decision_type="all")
            ys.append(rows[0]["kl_mean"] if rows else np.nan)
            es.append(rows[0]["kl_std"] if rows else 0.0)
        ax.bar(x + (i - 1.5) * 0.21, ys, width=0.19, yerr=es,
               color=FAMILY_COLORS[fam], alpha=0.45 if s0_trained else 1.0,
               label=label, capsize=2)
    ax.set_xticks(x, list(mains))
    ax.set_ylabel("KL(teacher ‖ student), nats")
    ax.set_title("2A — student fidelity per split (own test set)")
    ax.legend(frameon=False)
    fig.savefig(results_dir / "gap_by_split.png", bbox_inches="tight")
    plt.close(fig)

    # -- figure: data-size curve -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sizes = list(DATA_SIZES)
    for fam in ("simple", "nonlinear"):
        xs, ys, es = [], [], []
        for n in sizes + [None]:                      # None = the full-data S0 runs
            rows = _lookup(table, split="S0", model=fam, ablation="full",
                           n_train=n, eval_split="S0", decision_type="all")
            if rows:
                xs.append(100_000 if n is None else n)   # None = the 100k S0 main runs
                ys.append(rows[0]["kl_mean"])
                es.append(rows[0]["kl_std"])
        ax.errorbar(xs, ys, yerr=es, marker="o", color=FAMILY_COLORS[fam], label=fam)
    ax.set_xscale("log")
    ax.set_xlabel("training cases")
    ax.set_ylabel("KL(teacher ‖ student), nats")
    ax.set_title("2A — data-size curve (S0)")
    ax.legend(frameon=False)
    fig.savefig(results_dir / "data_size_curve.png", bbox_inches="tight")
    plt.close(fig)

    # -- diagnostics --------------------------------------------------------------
    diag_rows: list[list] = []
    for f in run_files:                                # c_L ⊙ o weight-norm check
        meta = json.loads(f.read_text(encoding="utf-8"))
        if meta["model"] not in ("simple", "agnostic_simple"):
            continue
        payload = torch.load(results_dir / "models" / f"{meta['run_id']}.pt",
                             weights_only=False)
        key = "w" if meta["model"] == "simple" else "inner.w"
        w = payload["state_dict"][key]
        diag_rows.append(["w_ctx_norm", meta["run_id"],
                          float(w[0, -12:].norm()), float(w[1, -12:].norm())])
    for fam in ("simple", "nonlinear"):                # personality-information gap
        main = _lookup(table, split="S0", model=fam, ablation="full", n_train=None,
                       eval_split="S0", decision_type="all")
        agn = _lookup(table, split="S0", model=f"agnostic_{fam}", ablation="full",
                      n_train=None, eval_split="S0", decision_type="all")
        if main and agn:
            diag_rows.append(["agnostic_gap", fam,
                              agn[0]["kl_mean"] - main[0]["kl_mean"], ""])
        for abl in ("no_context", "location_only"):    # ablation deltas
            row = _lookup(table, split="S0", model=fam, ablation=abl, n_train=None,
                          eval_split="S0", decision_type="all")
            if main and row:
                diag_rows.append([f"ablation_delta_{abl}", fam,
                                  row[0]["kl_mean"] - main[0]["kl_mean"], ""])
    write_csv(results_dir / "diagnostics.csv",
              ["kind", "who", "value_a", "value_b"], diag_rows)

    print(f"written: {results_dir / 'main_table.csv'}, gap_by_split.png, "
          f"data_size_curve.png, diagnostics.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.4: Run to verify pass**

Run: `python -m pytest tests/test_rq2_pipeline.py -q` — Expected: all PASS.

- [ ] **Step 5.5: Sanity-run the smoke CLI** (needs Tasks 3–4 smoke outputs)

Run: `python -m experiments.rq2.run_2a --smoke`
Expected: prints per-run evaluation lines; `results/rq2_smoke/` gains `main_table.csv`, `gap_by_split.png`, `data_size_curve.png`, `diagnostics.csv` (smoke has few runs, so several figure groups are simply absent — that is fine).

- [ ] **Step 5.6: Commit**

```bash
git add experiments/rq2/run_2a.py tests/test_rq2_pipeline.py
git commit -m "rq2 2a: evaluation, main table, figures, diagnostics"
```

---

### Task 6: `run_e_diag.py` — student adapter and E1–E4 structural diagnostic

**Files:**
- Create: `experiments/rq2/run_e_diag.py`
- Modify: `tests/test_rq2_pipeline.py` (append test class)

- [ ] **Step 6.1: Write the failing tests** (append)

```python
class TestEDiag:
    @pytest.fixture()
    def setup(self):
        import torch
        from experiments.rq2 import run_e_diag as ed
        from experiments.rq2 import gen_controlled as g
        from npc_policy import load_world
        from npc_policy.learned import SimplePolicy
        torch.manual_seed(11)
        model = SimplePolicy()
        with torch.no_grad():
            model.w += 0.3 * torch.randn_like(model.w)
        world = load_world(g.ensure_worlds()["full"])
        return ed, ed.StudentTraceAdapter(model), world, model

    def test_adapter_matches_predict_distribution(self, setup):
        from npc_policy import RecentBuffer
        from npc_policy.learned import predict_distribution
        from npc_policy.relations import compute_relations
        ed, adapter, world, model = setup
        p = Personality(np.array([0.5, -0.5, 0.2, 0.0, -0.3]))
        locs = world.resolve()
        # empty buffer
        np.testing.assert_allclose(
            adapter.trace(p, locs, buffer=None, level="location").P_rule,
            predict_distribution(model, p, locs, "location"), atol=1e-12)
        # non-empty buffer -> relations computed exactly as the controller would
        buf = RecentBuffer(maxlen=3)
        buf.push(locs[0]); buf.push(locs[2])
        rel = compute_relations(locs, buf, DEFAULT_CONFIG)
        np.testing.assert_allclose(
            adapter.trace(p, locs, buffer=buf, level="location").P_rule,
            predict_distribution(model, p, locs, "location", relations=rel), atol=1e-12)
        # action level uses current_location as the context
        adapter.current_location = locs[1]
        acts = world.actions_at(locs[1].id)
        np.testing.assert_allclose(
            adapter.trace(p, acts, buffer=None, level="action").P_rule,
            predict_distribution(model, p, acts, "action",
                                 selected_location=locs[1]), atol=1e-12)

    def test_adapter_action_without_location_raises(self, setup):
        ed, adapter, world, model = setup
        adapter.current_location = None
        with pytest.raises(ValueError):
            adapter.trace(Personality(np.zeros(5)),
                          world.actions_at("tavern"), level="action")

    def test_student_trajectory_runs_and_tracks_location(self, setup):
        ed, adapter, world, model = setup
        p = Personality(np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
        for memory in ("full", "location_only", "none"):
            visits, acts = ed.student_trajectory(adapter, world, p, seed=0,
                                                 rounds=4, memory=memory)
            assert len(visits) == len(acts) == 4
            assert adapter.current_location.id == visits[-1]
```

- [ ] **Step 6.2: Run to verify failure**

Run: `python -m pytest tests/test_rq2_pipeline.py::TestEDiag -q` — Expected: FAIL (module missing).

- [ ] **Step 6.3: Implement** — create `experiments/rq2/run_e_diag.py`:

```python
"""E1–E4 structural diagnostic on the trained S0 students (design §5).

A ``StudentTraceAdapter`` duck-types the scorer interface the RQ1 pipeline uses
(``trace(personality, candidates, buffer=…, level=…)``), so trained students run
on the frozen RQ1 matched cases without modifying ``experiments/rq1``. Action
choices need the selected location's context, which that interface does not
carry — the adapter holds ``current_location``, set by the local trajectory
runner after each location decision.

Pre-registered expectation (research spec §8): the simple model fails to track
the N-temperature channel (E1 N-sweep entropy curve); the nonlinear model
follows it.

Run from ``code/``:  python -m experiments.rq2.run_e_diag [--smoke]
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.rq1.common import (
    LOC_COLORS,
    TRAIT_COLORS,
    TRAIT_SHORT,
    TRAITS,
    TRAJ_TEMPERATURE,
    entropy,
    load_cases,
    mantel,
    pairwise_jsd,
    personality_of,
    run_trajectory,
    setup_style,
    spearman,
    trajectory_metrics,
    world_for,
    write_csv,
)
from npc_policy import (
    DEFAULT_CONFIG,
    DecisionController,
    HandAuthoredScorer,
    Personality,
    load_personalities,
)
from npc_policy.learned import predict_distribution
from npc_policy.relations import Relations, compute_relations

from .common import DATA, dirs
from .run_2a import load_student

FAMILIES = ("simple", "nonlinear")
STUDENT_STYLE = {"simple": ("#0072B2", "--"), "nonlinear": ("#E69F00", ":")}


class StudentTraceAdapter:
    """Makes a learned policy a drop-in for ``HandAuthoredScorer`` in the
    controller and the matched-case analyses. Only ``P_rule`` (and the relations
    actually used) is meaningful on the returned trace object."""

    def __init__(self, model, config=DEFAULT_CONFIG):
        self.model = model
        self.config = config
        self.current_location = None      # set by the trajectory runner

    def trace(self, personality, candidates, buffer=None, relations=None,
              level="location"):
        if relations is None and buffer is not None and not buffer.is_empty():
            relations = compute_relations(candidates, buffer, self.config)
        selected = self.current_location if level == "action" else None
        P = predict_distribution(self.model, personality, candidates, level,
                                 relations=relations, selected_location=selected)
        if relations is None:
            m = len(candidates)
            relations = Relations(np.zeros(m), np.zeros(m), np.zeros(m))
        return SimpleNamespace(P_rule=P, relations=relations, base=None,
                               P_base=None, mu=None, gamma=None, q=None, T_N=None)

    def distribution(self, personality, candidates, buffer=None, relations=None,
                     level="location"):
        return self.trace(personality, candidates, buffer, relations, level).P_rule


def student_trajectory(adapter: StudentTraceAdapter, world, personality, seed: int,
                       rounds: int = 50, memory: str = "full"):
    """RQ1 ``run_trajectory`` semantics with the location-context hook added."""
    if memory not in ("full", "location_only", "none"):
        raise ValueError(f"unknown memory condition {memory!r}")
    ctrl = DecisionController(
        adapter, config=adapter.config, mode="sample",
        rng=np.random.default_rng(seed), selection_temperature=TRAJ_TEMPERATURE,
    )
    visits, acts = [], []
    for _ in range(rounds):
        d = ctrl.choose_location(personality, world.resolve())
        adapter.current_location = world.effective_location(d.option.id)
        a = ctrl.choose_action(personality, world.actions_at(d.option.id))
        visits.append(d.option.id)
        acts.append(a.option.id)
        if memory == "none":
            ctrl.H_L.clear()
            ctrl.H_A.clear()
        elif memory == "location_only":
            ctrl.H_A.clear()
    return visits, acts


def discover_students(results_dir) -> dict[str, list]:
    """S0 main-run students per family (whatever seeds exist — works for smoke)."""
    out: dict[str, list] = {}
    for fam in FAMILIES:
        models = []
        for f in sorted((results_dir / "runs").glob(f"S0__{fam}__s?.json")):
            models.append(load_student(results_dir, f.stem))
        if models:
            out[fam] = models
    return out


def _seed_mean(models, personality, candidates, level) -> np.ndarray:
    return np.mean([predict_distribution(m, personality, candidates, level)
                    for m in models], axis=0)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="E1–E4 diagnostic on trained students")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    setup_style()
    _, results_dir = dirs(args.smoke)
    out = results_dir / "e_diag"
    out.mkdir(parents=True, exist_ok=True)
    students = discover_students(results_dir)
    if not students:
        raise SystemExit(f"no S0 student runs under {results_dir / 'runs'} — train first")
    scorer = HandAuthoredScorer()
    cases = load_cases()
    sweep = cases["profiles"]["sweep"]
    values = sorted({e["value"] for e in sweep})
    world = world_for("full")
    locs = world.resolve()
    n_profiles = 40 if args.smoke else len(cases["profiles"]["random"])
    traj_seeds = range(1) if args.smoke else range(2)
    conditions = ("full",) if args.smoke else ("full", "location_only", "none")

    def sweep_p(trait, v):
        return personality_of(next(e for e in sweep
                                   if e["trait"] == trait and e["value"] == v))

    # ---- E1 overlay: teacher vs student curves + N entropy ---------------------
    ent_rows = []
    for fam, models in students.items():
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
        color, dash = STUDENT_STYLE[fam]
        for ax, trait in zip(axes.flat, TRAITS):
            P_t = np.stack([scorer.distribution(sweep_p(trait, v), locs,
                                                level="location") for v in values])
            P_s = np.stack([_seed_mean(models, sweep_p(trait, v), locs, "location")
                            for v in values])
            for k, o in enumerate(locs):
                ax.plot(values, P_t[:, k], color=LOC_COLORS[o.id], label=o.id)
                ax.plot(values, P_s[:, k], color=LOC_COLORS[o.id], linestyle="--",
                        alpha=0.75)
            ax.set_title(f"{TRAIT_SHORT[trait]} (teacher solid / student dashed)")
            ax.set_ylabel("P")
        ax = axes.flat[5]
        ent_t = [entropy(scorer.distribution(sweep_p("neuroticism", v), locs,
                                             level="location")) for v in values]
        ent_s = [entropy(_seed_mean(models, sweep_p("neuroticism", v), locs,
                                    "location")) for v in values]
        ax.plot(values, ent_t, color=TRAIT_COLORS["neuroticism"], label="teacher")
        ax.plot(values, ent_s, color=color, linestyle=dash, label=fam)
        ax.set_title("N temperature: entropy of P (pre-registered check)")
        ax.set_xlabel("N value")
        ax.set_ylabel("entropy (nats)")
        ax.legend(frameon=False)
        fig.suptitle(f"E1 overlay — teacher vs {fam} (S0 students, seed mean)", y=1.0)
        fig.tight_layout()
        fig.savefig(out / f"e1_overlay_{fam}.png", bbox_inches="tight")
        plt.close(fig)
        for v, a, b in zip(values, ent_t, ent_s):
            ent_rows.append([fam, v, f"{a:.6f}", f"{b:.6f}"])
    write_csv(out / "e1_n_entropy.csv",
              ["family", "N", "teacher_entropy", "student_entropy"], ent_rows)

    # ---- E2: profile-distinguishability correlation ----------------------------
    profiles = [personality_of(e) for e in cases["profiles"]["random"][:n_profiles]]
    P_t = np.stack([scorer.distribution(p, locs, level="location") for p in profiles])
    D_t = pairwise_jsd(P_t)
    iu = np.triu_indices(len(profiles), k=1)
    e2_rows = []
    for fam, models in students.items():
        P_s = np.stack([_seed_mean(models, p, locs, "location") for p in profiles])
        D_s = pairwise_jsd(P_s)
        rho, pval = mantel(D_t, D_s, n_perm=199 if args.smoke else 999)
        e2_rows.append([fam, f"{spearman(D_t[iu], D_s[iu]):.4f}",
                        f"{rho:.4f}", f"{pval:.4f}"])
        print(f"E2 {fam}: mantel rho {rho:.3f} (p {pval:.3f})")
    write_csv(out / "e2_correlation.csv",
              ["family", "spearman_upper", "mantel_rho", "mantel_p"], e2_rows)

    # ---- E3/E4: trajectory statistics ------------------------------------------
    named = load_personalities(DATA / "personalities.json")
    rows = []
    for prof_name, p in named.items():
        for cond in conditions:
            runs = [trajectory_metrics(*run_trajectory(scorer, world, p, seed=s,
                                                       memory=cond))
                    for s in range(10)]
            rows.append(["teacher", prof_name, cond] + _metric_means(runs))
            for fam, models in students.items():
                runs = []
                for mi, m in enumerate(models):
                    adapter = StudentTraceAdapter(m)
                    for s in traj_seeds:
                        runs.append(trajectory_metrics(*student_trajectory(
                            adapter, world, p, seed=1000 * mi + s, memory=cond)))
                rows.append([fam, prof_name, cond] + _metric_means(runs))
    header = ["policy", "profile", "memory", "visit_entropy", "max_share",
              "distinct", "repeat_rate", "action_repeat_rate"]
    write_csv(out / "e3_e4_traj_stats.csv", header, rows)

    # small comparison figure: visit entropy per profile, full memory
    fig, ax = plt.subplots(figsize=(10, 4.5))
    names = list(named)
    x = np.arange(len(names))
    series = ["teacher"] + list(students)
    for i, pol in enumerate(series):
        ys = [float(r[3]) for r in rows if r[0] == pol and r[2] == "full"
              and r[1] in names]
        ax.bar(x + (i - (len(series) - 1) / 2) * 0.26, ys, width=0.24,
               label=pol,
               color={"teacher": "#444444", "simple": "#0072B2",
                      "nonlinear": "#E69F00"}[pol])
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_ylabel("visit entropy (nats)")
    ax.set_title("E3 — trajectory diversity, teacher vs students (full memory)")
    ax.legend(frameon=False)
    fig.savefig(out / "e3_visit_entropy.png", bbox_inches="tight")
    plt.close(fig)
    print(f"written: {out}")


def _metric_means(runs: list[dict]) -> list[str]:
    keys = ("visit_entropy", "max_share", "distinct", "repeat_rate",
            "action_repeat_rate")
    return [f"{float(np.mean([r.get(k, 0.0) for r in runs])):.4f}" for k in keys]


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.4: Run to verify pass**

Run: `python -m pytest tests/test_rq2_pipeline.py -q` — Expected: all PASS.

- [ ] **Step 6.5: Sanity-run the smoke CLI** (needs Task 4's smoke models and `data/rq1_cases/cases.json`; if the latter is missing, run `python -m experiments.rq1.gen_cases` once first)

Run: `python -m experiments.rq2.run_e_diag --smoke`
Expected: `results/rq2_smoke/e_diag/` gains `e1_overlay_simple.png`, `e1_overlay_nonlinear.png`, `e1_n_entropy.csv`, `e2_correlation.csv`, `e3_e4_traj_stats.csv`, `e3_visit_entropy.png`.

- [ ] **Step 6.6: Commit**

```bash
git add experiments/rq2/run_e_diag.py tests/test_rq2_pipeline.py
git commit -m "rq2 2a: student trace adapter and E1-E4 structural diagnostic"
```

---

### Task 7: `docs/rq2_runbook.md` + final verification

**Files:**
- Create: `docs/rq2_runbook.md`

- [ ] **Step 7.1: Write the runbook** — create `docs/rq2_runbook.md` with exactly this content:

````markdown
# RQ2 · 2A 实验操作手册

这份手册给实验执行者（你自己）看：按顺序敲命令即可，不需要读代码。
设计文档：`docs/specs/2026-07-10-rq2-2a-pipeline-design.md`。

## 0. 环境准备（一次性）

需要 Python 3.11+，依赖 numpy、matplotlib、torch：

```powershell
pip install numpy matplotlib
# CPU 版 torch（本机没独立显卡时）：
pip install torch --index-url https://download.pytorch.org/whl/cpu
# GPU 版 torch（服务器 / 有 NVIDIA 显卡，CUDA 12.x）：
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

所有命令都在 `code/` 目录下执行。装好后先跑一遍单元测试确认环境没问题：

```powershell
python -m pytest tests/ -q
```

预期最后一行是全部通过（无 failed）。

## 1. 四条命令

| 步骤 | 命令 | 作用 | 预计耗时 |
|---|---|---|---|
| 1 | `python -m experiments.rq2.gen_controlled` | 生成数据集 | 几分钟 |
| 2 | `python -m experiments.rq2.train` | 130 次训练（可断点续跑） | CPU 数小时 / GPU 更快 |
| 3 | `python -m experiments.rq2.run_2a` | 汇总指标、出表出图 | 几分钟 |
| 4 | `python -m experiments.rq2.run_e_diag` | E1–E4 结构诊断 | 几十分钟 |

规则：**一条跑完（回到提示符）再敲下一条**，顺序不能乱。
懒人写法（自动按顺序执行）：

```powershell
python -m experiments.rq2.gen_controlled; python -m experiments.rq2.train; python -m experiments.rq2.run_2a; python -m experiments.rq2.run_e_diag
```

## 2. 先冒烟，再全量

第一次一定先加 `--smoke` 跑一轮小规模流程（总共约 10 分钟），确认四步都能走通：

```powershell
python -m experiments.rq2.gen_controlled --smoke
python -m experiments.rq2.train --smoke
python -m experiments.rq2.run_2a --smoke
python -m experiments.rq2.run_e_diag --smoke
```

冒烟的输出在 `data/rq2_controlled_smoke/` 和 `results/rq2_smoke/`，和全量完全隔离，
确认没问题后可以整目录删掉。之后跑不带 `--smoke` 的全量四条。

## 3. 每步跑完应该看到什么

**第 1 步 gen_controlled** → `data/rq2_controlled/`：
- `pool.jsonl`（主数据池，约 17 万行，几百 MB）
- `test_S0.jsonl` … `test_G6.jsonl`（7 个测试集）
- `splits.json`、`meta.json`（划分清单和元信息）
- 屏幕上每个划分打印 `train 100000 / val 5000 / test 5000`

**第 2 步 train** → `results/rq2/`：
- `runs/*.json`（每次训练一个结果文件，全量共 130 个）
- `models/*.pt`（对应权重）
- 屏幕上每完成一次打印 `[k/130] 运行名: val KL …`
- **中断了怎么办**：直接重新敲同一条命令，开头会打印
  `130 runs, N already done, M to go`，自动接着跑。

**第 3 步 run_2a** → `results/rq2/`：
- `main_table.csv`（主表：每划分×每模型的 KL / JSD / top-1）
- `gap_by_split.png`（简单 vs 非线性差距图）
- `data_size_curve.png`（数据量曲线）
- `diagnostics.csv`（c_L⊙o 权重范数、零人格对照差距、消融差值）

**第 4 步 run_e_diag** → `results/rq2/e_diag/`：
- `e1_overlay_simple.png`、`e1_overlay_nonlinear.png`（教师/学生曲线叠加，
  右下角 N 温度熵曲线就是预注册检验）
- `e1_n_entropy.csv`、`e2_correlation.csv`、`e3_e4_traj_stats.csv`、
  `e3_visit_entropy.png`

四步全部跑完后告诉 Claude，把 `results/rq2/` 交给它分析。

## 4. 在服务器 / GPU 上跑（可选）

1. 把代码弄上去：`git clone <你的私有仓库>` 或直接把 `code/` 打包上传。
2. 按第 0 节装依赖（GPU 机器装 cu121 版 torch）。
3. 训练命令默认自动检测 GPU（`--device auto`）；强制指定用
   `python -m experiments.rq2.train --device cuda`。
4. 断开 SSH 也继续跑：

   ```bash
   nohup python -m experiments.rq2.train > train.log 2>&1 &
   tail -f train.log        # 随时看进度
   ```

5. 跑完把 `results/rq2/` 和 `data/rq2_controlled/` 整目录拷回本地
   （scp / 网盘均可），Claude 在本地读文件分析。

技术说明：GPU 上自动用 float32 训练（更快），权重保存时统一转回 float64，
第 3、4 步在哪台机器上跑结果都一样。

## 5. 常见问题

- **`dataset missing: … — run gen_controlled first`**：第 2 步在第 1 步之前跑了，
  或 `--smoke` 和全量混用了。冒烟四条全带 `--smoke`，全量四条全不带。
- **第 2 步某一次训练报错退出**：重新敲同一条命令即可（已完成的会跳过）。
  反复在同一个运行上报错就把屏幕输出发给 Claude。
- **只想重跑某几个运行**：删掉 `results/rq2/runs/` 里对应的 `.json`（和
  `models/` 里同名 `.pt`），再敲训练命令；或用
  `python -m experiments.rq2.train --only S0__simple` 只跑指定前缀。
- **第 4 步提示缺 `data/rq1_cases/cases.json`**：先跑一次
  `python -m experiments.rq1.gen_cases`。
- **内存**：第 2 步会把主数据池整个读进内存，需要约 2–4 GB 空闲内存。
````

- [ ] **Step 7.2: Full-suite verification**

Run: `python -m pytest tests/ -q`
Expected: all tests pass (49 pre-existing + all new ones), 0 warnings.

Run the complete smoke sequence end-to-end, in order:

```powershell
python -m experiments.rq2.gen_controlled --smoke
python -m experiments.rq2.train --smoke
python -m experiments.rq2.run_2a --smoke
python -m experiments.rq2.run_e_diag --smoke
```

Expected: every command exits cleanly; the files listed in runbook §3 exist under the `_smoke` directories. (This validates the pipeline; the user will re-run the same smoke sequence himself from the runbook, then the full run.)

- [ ] **Step 7.3: Commit**

```bash
git add docs/rq2_runbook.md
git commit -m "rq2 2a: Chinese runbook for the four-step pipeline"
```

---

## Post-review amendments (as-executed deviations)

- **Ablation renamed `"none"` → `"no_context"`** (Task 1 quality review): `"none"` read as "no ablation" while actually meaning "no recent-choice context", and the string is baked into resumable run ids. `ABLATIONS = ("no_context", "location_only")`; `case_to_inputs` accepts `("full", "no_context", "location_only")`. Task 5's diagnostics loop updated above; any other literal `"none"` for ablations in this plan should be read as `"no_context"`. (The trajectory **memory conditions** in Task 6 — `"full"/"location_only"/"none"` — follow the RQ1 convention and are NOT renamed.)
- **G5 redefined: saturated history** (Task 2/3 quality review, Critical): recency
  weights normalise over buffer length, so a single-family buffer of ANY length
  yields the same model-visible relations as a 3-run (`rep = 1.0` ceiling) — the
  original "no 3-runs in train" filter left input-identical shorter cases in
  train, making G5 vacuous. As-built: the train filter excludes every case whose
  relevant non-empty buffer is single-family (`g5_saturated_history`); train max
  `rep` ≈ 0.82, test `rep = 1.0` is genuinely unseen. Targeted G5 test cases
  additionally guarantee the repeated option IS one of the candidates (else the
  rep ceiling never appears in the test inputs).
- **run_2a eval scheme extended** (same review, I3): per-run rows carry
  `eval_split` (the actual test set evaluated on) instead of the own/S0 binary;
  S0 main runs are evaluated on every test set; the gap figure gains
  S0-trained reference bars. Task 5 code blocks above already reflect this.
- **Generation hardening** (same review): targeted records get traceable ids
  (`tgt-<split>-NNNN`); each targeted split uses a fresh sampler on stream
  `[seed, 5, k]` (pool-size changes no longer shift test sets); sampler buffer
  lengths read `self.scorer.config.K_L/K_A` instead of `DEFAULT_CONFIG`;
  G5 action-test buffers use a random native action, not always the first;
  `meta.json` records per-test-file composition (source × decision type) and
  world-file hashes; tests strengthened: independent negated held-out checks on
  every split's train+val manifests (anti-inverted-filter), stored relations
  recomputable from stored buffers, two-run byte-identity determinism check,
  G3/G4 included in the label-match loop.
- **Generation sizes raised** (Task 2/3 implementer, measured): the sizes above under-fill G6 — rollout location cases in arena-unlocked worlds always contain `arena`, so only ~35% of synthetic location cases survive the G6 filter and the plan's pool leaves G6 at ~93k < 105k. As-built: `arena_locked` trajectories 300 → 500 full (pool ≈ 190k), 4 → 8 smoke; the `RuntimeError` guard in `generate()` still protects any future size change.
- **Fixture style** (pytest 9 deprecation): class-scoped fixtures are declared as plain functions with `@pytest.fixture(scope="class")`, not instance methods, to keep the suite at 0 warnings.
- **Task 1 hardening** (quality review): atomic `write_pool` (tmp + `os.replace`); `read_pool` requires the `gen` tag; `case_to_inputs` also rejects a location case carrying a `selected_location`, and copies `p`/`target` arrays; `kl_np` floors `q` at `np.finfo(float).tiny`; JSD pinned at nonzero values and a maximal-case pool round-trip test added; docstrings clarified (buffer order oldest→newest on disk, `config_hash` pins config values not scorer code, `build_model` seeds the global torch RNG, `top1_agree` tie-breaking).

- **Final-review hardening** (holistic review, commit 00f616c): E1 N-entropy and
  E2 Mantel computed per seed then mean ± std (entropy of the seed-mean
  distribution is Jensen-inflated exactly where seeds disagree — the
  pre-registered regime); `train.py` and `run_e_diag.py` refuse to run if
  `config_hash()` differs from the dataset's `meta.json` (frozen-teacher drift
  guard); run result/model files written atomically; RuntimeError on
  all-NaN val KL; aggregate std uses ddof=1; `eval_splits_for()` extracted and
  unit-pinned; tests added for best-state snapshotting and `make_loader`
  prefix nesting; `run_all` progress print tolerates injected train_fn results.
- **Execution-mode change** (user, token cost): Groups C–E were implemented
  inline by the coordinator instead of per-task subagents; per-task dual
  reviews were replaced by the single final holistic review above. Groups A–B
  retained the full subagent + dual-review flow.

## Post-plan notes for the coordinator

- Smoke outputs created during Step 7.2 (`data/rq2_controlled_smoke/`, `results/rq2_smoke/`) are verification artefacts — do **not** commit them; leave them for the user to compare against when he runs the smoke himself.
- `data/rq1_cases/worlds/arena_locked.json` is (re)generated by `ensure_worlds()`; commit it only if the other world variant files are already tracked (follow the existing repo state).
- Nothing in this plan modifies `npc_policy/` or `experiments/rq1/`.






