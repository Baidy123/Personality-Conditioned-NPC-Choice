# RQ2 Model Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the two RQ2 learned-policy architectures (simple linear, nonlinear dual-tower), the agnostic control wrapper, the padded-batch container, and the KL loss, with property tests pinning correctness.

**Architecture:** Per-candidate scoring + masked softmax inside a padded `(B, M)` batch (design record: `docs/specs/2026-07-09-rq2-model-layer-design.md`). `npc_policy/features.py` (numpy) is the single authority for the 114-dim `phi` layout; `npc_policy/learned.py` (the only torch module) rebuilds it and a parity test pins the two together. Every model exposes a training path (`forward(batch) → masked log-probs`) and a scorer-parallel single-case path (`predict_distribution → np.ndarray`).

**Tech Stack:** Python 3.12, numpy 1.26, PyTorch CPU (installed), pytest. All torch tensors are float64 (`.double()`) for parity with the numpy reference.

**Working directory for all commands:** `C:\Users\76992\Desktop\MSc_dissertation\code`

**No git repo:** the project folder is not under version control, so each task ends with a full-suite run (`python -m pytest tests/ -q`) instead of a commit. If a repo is initialised later, commit at those same points.

---

### Task 1: `case_inputs` assembly helper in `features.py`

The single place that turns scorer-style raw materials (Personality, Options, Relations, decision type, selected location) into the numpy dict every batch is built from.

**Files:**
- Modify: `npc_policy/features.py` (append after `phi_matrix`)
- Create: `tests/test_learned.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_learned.py`:

```python
"""Model-layer tests (spec: docs/specs/2026-07-09-rq2-model-layer-design.md §5)."""

from __future__ import annotations

import numpy as np
import pytest

from npc_policy.features import (
    N_MODEL_TAGS,
    N_RELATIONS,
    N_TRAITS,
    PHI_DIM,
    case_inputs,
    phi_matrix,
)
from npc_policy.relations import Relations
from npc_policy.representation import Option, Personality


# --- shared random-case helpers --------------------------------------------------

def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _random_options(rng: np.random.Generator, level: str, m: int) -> list[Option]:
    n_native = 9 if level == "location" else 11
    return [
        Option(id=f"{level[:3]}_{i}", features=rng.random(n_native), level=level)
        for i in range(m)
    ]


def _random_relations(rng: np.random.Generator, m: int) -> Relations:
    sim = rng.random(m)
    return Relations(rep=rng.random(m), sim=sim, nov=1.0 - sim)


def _random_case(
    rng: np.random.Generator,
    decision_type: str = "location",
    m: int = 4,
    with_relations: bool = True,
) -> dict:
    level = decision_type
    candidates = _random_options(rng, level, m)
    relations = _random_relations(rng, m) if with_relations else None
    selected = (
        _random_options(rng, "location", 1)[0] if decision_type == "action" else None
    )
    personality = Personality(rng.uniform(-1.0, 1.0, 5))
    return case_inputs(
        personality, candidates, decision_type,
        relations=relations, selected_location=selected,
    )


# --- Task 1: case_inputs ----------------------------------------------------------

class TestCaseInputs:
    def test_location_case_shapes_and_zero_context(self):
        rng = _rng()
        p = Personality(rng.uniform(-1, 1, 5))
        cands = _random_options(rng, "location", 3)
        out = case_inputs(p, cands, "location", relations=_random_relations(rng, 3))
        assert out["p"].shape == (N_TRAITS,)
        assert out["d"] == 0
        assert out["cand"].shape == (3, N_MODEL_TAGS)
        assert out["rel"].shape == (3, N_RELATIONS)
        assert np.all(out["ctx"] == 0.0)

    def test_action_case_carries_selected_location_context(self):
        rng = _rng(1)
        p = Personality(rng.uniform(-1, 1, 5))
        loc = _random_options(rng, "location", 1)[0]
        acts = _random_options(rng, "action", 4)
        out = case_inputs(p, acts, "action", selected_location=loc)
        assert out["d"] == 1
        np.testing.assert_allclose(out["ctx"], loc.to_padded12())
        # no relations given -> zero matrix (empty buffer / no-context ablation)
        assert np.all(out["rel"] == 0.0)

    def test_location_case_rejects_selected_location(self):
        rng = _rng(2)
        p = Personality(rng.uniform(-1, 1, 5))
        cands = _random_options(rng, "location", 2)
        with pytest.raises(ValueError):
            case_inputs(p, cands, "location", selected_location=cands[0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_learned.py -v`
Expected: FAIL — `ImportError: cannot import name 'case_inputs'`

- [ ] **Step 3: Implement `case_inputs`**

Append to `npc_policy/features.py`:

```python
def case_inputs(
    personality: Personality,
    candidates: list[Option],
    decision_type: str,
    relations: Relations | None = None,
    selected_location: Option | None = None,
) -> dict:
    """Assemble one case's raw model inputs (numpy, unpadded).

    The same raw materials the scorer receives; ``PolicyBatch.from_cases``
    (``learned.py``) consumes lists of these dicts. ``relations=None`` yields the
    all-zero relation matrix (empty buffer or the no-context ablation).
    """
    if decision_type == "location" and selected_location is not None:
        raise ValueError("location cases use an empty selected-location context")
    cand12 = padded_candidates(candidates)
    return {
        "p": np.asarray(personality.vector, dtype=float),
        "d": decision_type_index(decision_type),
        "ctx": context_vector(selected_location),
        "cand": cand12,
        "rel": relations_matrix(relations, cand12.shape[0]),
    }
```

Also extend the imports at the top of `features.py`:

```python
from .representation import Option, Personality
```

(`Personality` is new; `Option` is already imported.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_learned.py -v`
Expected: 3 PASS

- [ ] **Step 5: Full suite**

Run: `python -m pytest tests/ -q`
Expected: all green (existing scorer/schema/config tests unaffected)

---

### Task 2: pin the `phi` layout with a hand-computed example

`phi_matrix` already exists; this test freezes the *semantic order* (a parity test alone would pass if numpy and torch made the same mistake).

**Files:**
- Test: `tests/test_learned.py` (append)

- [ ] **Step 1: Write the test (expected to pass immediately — a regression pin)**

Append to `tests/test_learned.py`:

```python
# --- Task 2: phi layout pin -------------------------------------------------------

class TestPhiLayout:
    def test_hand_computed_segments_with_single_trait(self):
        # p = e_O (first trait 1, rest 0) makes the bilinear blocks readable.
        p = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        o = np.linspace(0.1, 1.0, N_MODEL_TAGS)          # distinct values
        rel = np.array([0.2, 0.5, 0.5])
        ctx = np.linspace(1.0, 0.1, N_MODEL_TAGS)
        phi = phi_matrix(p, o[None, :], rel[None, :], ctx)
        assert phi.shape == (1, PHI_DIM)
        row = phi[0]
        i = 0
        np.testing.assert_allclose(row[i:i + 12], o); i += 12          # o
        np.testing.assert_allclose(row[i:i + 12], o**2); i += 12       # o²
        np.testing.assert_allclose(row[i:i + 3], rel); i += 3          # rel
        # p ⊗ o, trait-major: trait 0 block = o, traits 1-4 blocks = 0
        np.testing.assert_allclose(row[i:i + 12], o)
        np.testing.assert_allclose(row[i + 12:i + 60], 0.0); i += 60
        # p ⊗ rel, trait-major: trait 0 block = rel, rest 0
        np.testing.assert_allclose(row[i:i + 3], rel)
        np.testing.assert_allclose(row[i + 3:i + 15], 0.0); i += 15
        np.testing.assert_allclose(row[i:i + 12], ctx * o); i += 12    # c_L ⊙ o
        assert i == PHI_DIM
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_learned.py::TestPhiLayout -v`
Expected: PASS (if it fails, `phi_matrix` violates the documented order — fix `features.py`, not the test)

---

### Task 3: `PolicyBatch` (padded + mask) and `masked_log_softmax` + `UniformBaseline`

**Files:**
- Create: `npc_policy/learned.py`
- Test: `tests/test_learned.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_learned.py`:

```python
import torch

from npc_policy.learned import (
    PolicyBatch,
    UniformBaseline,
    masked_log_softmax,
)


# --- Task 3: PolicyBatch + masked softmax + uniform baseline ----------------------

class TestPolicyBatch:
    def test_padding_and_mask(self):
        rng = _rng(3)
        cases = [_random_case(rng, "location", m=2), _random_case(rng, "action", m=5)]
        batch = PolicyBatch.from_cases(cases)
        assert batch.cand.shape == (2, 5, N_MODEL_TAGS)
        assert batch.rel.shape == (2, 5, N_RELATIONS)
        assert batch.mask.tolist() == [[True, True, False, False, False], [True] * 5]
        assert batch.d.tolist() == [0, 1]
        assert batch.cand.dtype == torch.double
        # padding rows are zero
        assert torch.all(batch.cand[0, 2:] == 0.0)

    def test_optional_targets_are_padded(self):
        rng = _rng(4)
        cases = [_random_case(rng, "location", m=2), _random_case(rng, "location", m=3)]
        cases[0]["target"] = np.array([0.7, 0.3])
        cases[1]["target"] = np.array([0.2, 0.5, 0.3])
        batch = PolicyBatch.from_cases(cases)
        assert batch.target.shape == (2, 3)
        assert float(batch.target[0, 2]) == 0.0

    def test_no_targets_gives_none(self):
        rng = _rng(5)
        batch = PolicyBatch.from_cases([_random_case(rng, "location", m=2)])
        assert batch.target is None


class TestMaskedSoftmaxAndUniform:
    def test_masked_log_softmax_zeroes_padding(self):
        scores = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.double)
        mask = torch.tensor([[True, True, False]])
        probs = masked_log_softmax(scores, mask).exp()
        assert float(probs[0, 2]) == 0.0
        assert float(probs.sum()) == pytest.approx(1.0)

    def test_uniform_baseline(self):
        rng = _rng(6)
        batch = PolicyBatch.from_cases(
            [_random_case(rng, "location", m=3), _random_case(rng, "action", m=5)]
        )
        probs = UniformBaseline()(batch).exp()
        np.testing.assert_allclose(probs[0, :3].numpy(), 1 / 3, atol=1e-12)
        np.testing.assert_allclose(probs[0, 3:].numpy(), 0.0, atol=0.0)
        np.testing.assert_allclose(probs[1].numpy(), 1 / 5, atol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_learned.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'npc_policy.learned'`

- [ ] **Step 3: Create `npc_policy/learned.py`**

```python
"""RQ2 learned policies — the only torch module in ``npc_policy``.

Design record: ``docs/specs/2026-07-09-rq2-model-layer-design.md``; model forms:
``docs/specs/2026-07-09-rq2-models-and-2a-experiments.md`` §2–§4.

All models score each candidate independently and softmax within the batch mask,
so variable candidate-set sizes and candidate-order equivariance hold by
construction (matching the hand-authored teacher's structure). Batches are padded
to the largest candidate count in the batch; padding positions get score −inf,
hence probability exactly 0 and no gradient.

Everything runs in float64 for parity with the numpy reference in ``features.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import Tensor, nn

from .features import (
    N_MODEL_TAGS,
    N_RELATIONS,
    N_TRAITS,
    PHI_DIM,
    case_inputs,
)
from .relations import Relations
from .representation import Option, Personality


@dataclass(frozen=True)
class PolicyBatch:
    """One padded batch of decision cases (see module docstring)."""

    p: Tensor       # (B, 5)   float64 personality
    d: Tensor       # (B,)     long    decision type: 0 location, 1 action
    ctx: Tensor     # (B, 12)  float64 selected-location context (zeros for location)
    cand: Tensor    # (B, M, 12) float64 padded candidate features
    rel: Tensor     # (B, M, 3)  float64 rep/sim/nov
    mask: Tensor    # (B, M)   bool    True at real candidates
    target: Tensor | None = None   # (B, M) float64 teacher distribution, if provided

    @classmethod
    def from_cases(cls, cases: list[dict]) -> "PolicyBatch":
        """Build a batch from ``features.case_inputs`` dicts.

        If **every** case dict carries a ``"target"`` distribution, targets are
        padded into ``batch.target``; otherwise ``target`` is ``None``.
        """
        if not cases:
            raise ValueError("batch must contain at least one case")
        B = len(cases)
        M = max(c["cand"].shape[0] for c in cases)

        p = torch.zeros(B, N_TRAITS, dtype=torch.double)
        d = torch.zeros(B, dtype=torch.long)
        ctx = torch.zeros(B, N_MODEL_TAGS, dtype=torch.double)
        cand = torch.zeros(B, M, N_MODEL_TAGS, dtype=torch.double)
        rel = torch.zeros(B, M, N_RELATIONS, dtype=torch.double)
        mask = torch.zeros(B, M, dtype=torch.bool)

        with_target = all("target" in c for c in cases)
        target = torch.zeros(B, M, dtype=torch.double) if with_target else None

        for b, c in enumerate(cases):
            m = c["cand"].shape[0]
            p[b] = torch.from_numpy(np.asarray(c["p"], dtype=float))
            d[b] = int(c["d"])
            ctx[b] = torch.from_numpy(np.asarray(c["ctx"], dtype=float))
            cand[b, :m] = torch.from_numpy(np.asarray(c["cand"], dtype=float))
            rel[b, :m] = torch.from_numpy(np.asarray(c["rel"], dtype=float))
            mask[b, :m] = True
            if target is not None:
                target[b, :m] = torch.from_numpy(np.asarray(c["target"], dtype=float))
        return cls(p=p, d=d, ctx=ctx, cand=cand, rel=rel, mask=mask, target=target)


def masked_log_softmax(scores: Tensor, mask: Tensor) -> Tensor:
    """Log-softmax over the last dim with padding forced to log-prob −inf."""
    return torch.log_softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)


class UniformBaseline(nn.Module):
    """Untrained floor: uniform over each case's real candidates."""

    def forward(self, batch: PolicyBatch) -> Tensor:
        scores = torch.zeros_like(batch.mask, dtype=batch.p.dtype)
        return masked_log_softmax(scores, batch.mask)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_learned.py -v`
Expected: all PASS

- [ ] **Step 5: Full suite**

Run: `python -m pytest tests/ -q`
Expected: all green

---

### Task 4: torch `phi_torch` + parity with numpy

**Files:**
- Modify: `npc_policy/learned.py`
- Test: `tests/test_learned.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_learned.py`:

```python
from npc_policy.learned import phi_torch


# --- Task 4: torch phi parity -----------------------------------------------------

class TestPhiParity:
    @pytest.mark.parametrize("decision_type", ["location", "action"])
    @pytest.mark.parametrize("with_relations", [True, False])
    def test_torch_matches_numpy_reference(self, decision_type, with_relations):
        rng = _rng(7)
        cases = [
            _random_case(rng, decision_type, m=m, with_relations=with_relations)
            for m in (2, 4)
        ]
        batch = PolicyBatch.from_cases(cases)
        phi_t = phi_torch(batch.p, batch.cand, batch.rel, batch.ctx).numpy()
        for b, c in enumerate(cases):
            m = c["cand"].shape[0]
            phi_np = phi_matrix(c["p"], c["cand"], c["rel"], c["ctx"])
            np.testing.assert_allclose(phi_t[b, :m], phi_np, atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_learned.py::TestPhiParity -v`
Expected: FAIL — `ImportError: cannot import name 'phi_torch'`

- [ ] **Step 3: Implement `phi_torch`**

Append to `npc_policy/learned.py`:

```python
def phi_torch(p: Tensor, cand: Tensor, rel: Tensor, ctx: Tensor) -> Tensor:
    """Rebuild ``features.phi_matrix`` in torch for a padded batch.

    Layout authority is the numpy version; ``tests/test_learned.py`` pins parity.
    Shapes: ``p (B,5)``, ``cand (B,M,12)``, ``rel (B,M,3)``, ``ctx (B,12)`` →
    ``(B, M, PHI_DIM)``. Trait-major bilinear blocks: index ``t * n + f``.
    """
    B, M, _ = cand.shape
    p_x_o = (p[:, None, :, None] * cand[:, :, None, :]).reshape(B, M, -1)
    p_x_rel = (p[:, None, :, None] * rel[:, :, None, :]).reshape(B, M, -1)
    return torch.cat(
        [cand, cand**2, rel, p_x_o, p_x_rel, ctx[:, None, :] * cand], dim=-1
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_learned.py -v`
Expected: all PASS

---

### Task 5: `SimplePolicy` + single-case `predict_distribution`

**Files:**
- Modify: `npc_policy/learned.py`
- Test: `tests/test_learned.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_learned.py`:

```python
from npc_policy.learned import SimplePolicy, predict_distribution


def _randomised(model: "torch.nn.Module", seed: int) -> "torch.nn.Module":
    """Give a model non-trivial random weights (deterministic)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for prm in model.parameters():
            prm.copy_(torch.randn(prm.shape, generator=gen, dtype=prm.dtype))
    return model


def _predict_case(model, rng, decision_type="location", m=4, seed_p=0):
    level = decision_type
    candidates = _random_options(rng, level, m)
    relations = _random_relations(rng, m)
    selected = (
        _random_options(rng, "location", 1)[0] if decision_type == "action" else None
    )
    personality = Personality(_rng(seed_p).uniform(-1, 1, 5))
    dist = predict_distribution(
        model, personality, candidates, decision_type,
        relations=relations, selected_location=selected,
    )
    return dist, personality, candidates, relations, selected


# --- Task 5: SimplePolicy ---------------------------------------------------------

class TestSimplePolicy:
    def test_probability_validity(self):
        rng = _rng(8)
        model = _randomised(SimplePolicy(), seed=1)
        batch = PolicyBatch.from_cases(
            [_random_case(rng, "location", m=3), _random_case(rng, "action", m=6)]
        )
        probs = model(batch).exp().detach()
        sums = (probs * batch.mask).sum(-1)
        np.testing.assert_allclose(sums.numpy(), 1.0, atol=1e-12)
        assert torch.all(probs[~batch.mask] == 0.0)

    def test_permutation_equivariance(self):
        rng = _rng(9)
        model = _randomised(SimplePolicy(), seed=2)
        dist, personality, candidates, relations, _ = _predict_case(model, rng, m=5)
        perm = _rng(10).permutation(5)
        shuffled = [candidates[i] for i in perm]
        rel_perm = Relations(
            rep=relations.rep[perm], sim=relations.sim[perm], nov=relations.nov[perm]
        )
        dist2 = predict_distribution(
            model, personality, shuffled, "location", relations=rel_perm
        )
        np.testing.assert_allclose(dist2, dist[perm], atol=1e-12)

    def test_decision_type_uses_separate_weights(self):
        # same padded features scored as location vs action must differ
        # (unless weights coincide, which random init makes negligible)
        rng = _rng(11)
        model = _randomised(SimplePolicy(), seed=3)
        acts = _random_options(rng, "action", 3)
        sel = _random_options(rng, "location", 1)[0]
        p = Personality(rng.uniform(-1, 1, 5))
        d_act = predict_distribution(model, p, acts, "action", selected_location=sel)
        # score the same candidates under the location weight row by rebuilding
        # the case dict manually (bypasses the level/type pairing on purpose;
        # the selected-location context is kept so both rows see identical phi)
        case = case_inputs(p, acts, "action", selected_location=sel)
        case["d"] = 0
        batch = PolicyBatch.from_cases([case])
        d_loc = model(batch).exp().detach().numpy()[0]
        assert not np.allclose(d_act, d_loc)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_learned.py -v`
Expected: FAIL — `ImportError: cannot import name 'SimplePolicy'`

- [ ] **Step 3: Implement `SimplePolicy` and `predict_distribution`**

Append to `npc_policy/learned.py`:

```python
class SimplePolicy(nn.Module):
    """Linear scorer over ``phi`` with one weight vector per decision type.

    228 parameters (2 × PHI_DIM). Initialised to zero → starts at the uniform
    distribution; the KL objective is convex in ``w``.
    """

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(2, PHI_DIM, dtype=torch.double))

    def forward(self, batch: PolicyBatch) -> Tensor:
        phi = phi_torch(batch.p, batch.cand, batch.rel, batch.ctx)  # (B, M, PHI)
        scores = (phi * self.w[batch.d][:, None, :]).sum(-1)        # (B, M)
        return masked_log_softmax(scores, batch.mask)


def predict_distribution(
    model: nn.Module,
    personality: Personality,
    candidates: list[Option],
    decision_type: str,
    relations: Relations | None = None,
    selected_location: Option | None = None,
) -> np.ndarray:
    """Single-case choice distribution — parallel to ``HandAuthoredScorer.distribution``.

    Runs in eval mode under ``no_grad`` and restores the previous training flag,
    so trained students drop into the RQ1 E1–E4 pipeline and Study 3 unchanged.
    """
    case = case_inputs(
        personality, candidates, decision_type,
        relations=relations, selected_location=selected_location,
    )
    batch = PolicyBatch.from_cases([case])
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            log_q = model(batch)
    finally:
        if was_training:
            model.train()
    return log_q[0, : len(candidates)].exp().numpy()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_learned.py -v`
Expected: all PASS

- [ ] **Step 5: Full suite**

Run: `python -m pytest tests/ -q`
Expected: all green

---

### Task 6: `NonlinearPolicy` (+ batch/single consistency for both models)

**Files:**
- Modify: `npc_policy/learned.py`
- Test: `tests/test_learned.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_learned.py`:

```python
from npc_policy.learned import NonlinearPolicy


def _seeded_nonlinear(seed: int) -> NonlinearPolicy:
    torch.manual_seed(seed)
    return NonlinearPolicy()


# --- Task 6: NonlinearPolicy + batch/single consistency ---------------------------

class TestNonlinearPolicy:
    def test_probability_validity(self):
        rng = _rng(12)
        torch.manual_seed(0)
        model = NonlinearPolicy()
        batch = PolicyBatch.from_cases(
            [_random_case(rng, "location", m=2), _random_case(rng, "action", m=6)]
        )
        model.eval()
        probs = model(batch).exp().detach()
        sums = (probs * batch.mask).sum(-1)
        np.testing.assert_allclose(sums.numpy(), 1.0, atol=1e-12)
        assert torch.all(probs[~batch.mask] == 0.0)

    def test_permutation_equivariance(self):
        rng = _rng(13)
        torch.manual_seed(1)
        model = NonlinearPolicy()
        dist, personality, candidates, relations, sel = _predict_case(
            model, rng, decision_type="action", m=5
        )
        perm = _rng(14).permutation(5)
        rel_perm = Relations(
            rep=relations.rep[perm], sim=relations.sim[perm], nov=relations.nov[perm]
        )
        dist2 = predict_distribution(
            model, personality, [candidates[i] for i in perm], "action",
            relations=rel_perm, selected_location=sel,
        )
        np.testing.assert_allclose(dist2, dist[perm], atol=1e-12)


class TestBatchSingleConsistency:
    @pytest.mark.parametrize("make_model", [
        lambda: _randomised(SimplePolicy(), seed=4),
        lambda: _seeded_nonlinear(2),
    ])
    def test_case_alone_equals_case_in_mixed_batch(self, make_model):
        model = make_model()
        model.eval()
        rng = _rng(15)
        cases = [
            _random_case(rng, "location", m=2),
            _random_case(rng, "action", m=5),
            _random_case(rng, "location", m=8),
        ]
        with torch.no_grad():
            batched = model(PolicyBatch.from_cases(cases)).exp()
            for b, c in enumerate(cases):
                alone = model(PolicyBatch.from_cases([c])).exp()
                m = c["cand"].shape[0]
                np.testing.assert_allclose(
                    batched[b, :m].numpy(), alone[0, :m].numpy(), atol=1e-12
                )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_learned.py -v`
Expected: FAIL — `ImportError: cannot import name 'NonlinearPolicy'`

- [ ] **Step 3: Implement `NonlinearPolicy`**

Append to `npc_policy/learned.py`:

```python
class NonlinearPolicy(nn.Module):
    """Dual-tower compatibility network (research spec §3; sizes provisional).

    ``head`` sees ``[e_p, e_o, e_p ⊙ e_o, e_c]`` — the elementwise product gives
    it a multiplicative personality–option interaction directly.
    """

    def __init__(
        self,
        tower_width: int = 64,
        embed_dim: int = 32,
        ctx_dim: int = 16,
        head_width: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.p_tower = nn.Sequential(
            nn.Linear(N_TRAITS, tower_width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(tower_width, embed_dim),
        )
        self.o_tower = nn.Sequential(
            nn.Linear(N_MODEL_TAGS + N_RELATIONS, tower_width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(tower_width, embed_dim),
        )
        self.ctx_encoder = nn.Sequential(
            nn.Linear(2 + N_MODEL_TAGS, ctx_dim), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(3 * embed_dim + ctx_dim, head_width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(head_width, 1),
        )
        self.double()

    def forward(self, batch: PolicyBatch) -> Tensor:
        B, M, _ = batch.cand.shape
        e_p = self.p_tower(batch.p)                                   # (B, E)
        e_o = self.o_tower(torch.cat([batch.cand, batch.rel], -1))    # (B, M, E)
        d_onehot = nn.functional.one_hot(batch.d, 2).to(batch.ctx.dtype)
        e_c = self.ctx_encoder(torch.cat([d_onehot, batch.ctx], -1))  # (B, C)
        e_p_rows = e_p[:, None, :].expand(-1, M, -1)
        e_c_rows = e_c[:, None, :].expand(-1, M, -1)
        h = torch.cat([e_p_rows, e_o, e_p_rows * e_o, e_c_rows], -1)
        scores = self.head(h).squeeze(-1)                             # (B, M)
        return masked_log_softmax(scores, batch.mask)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_learned.py -v`
Expected: all PASS

- [ ] **Step 5: Full suite**

Run: `python -m pytest tests/ -q`
Expected: all green

---

### Task 7: `AgnosticPolicy` wrapper

**Files:**
- Modify: `npc_policy/learned.py`
- Test: `tests/test_learned.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_learned.py`:

```python
from npc_policy.learned import AgnosticPolicy


# --- Task 7: agnostic control -----------------------------------------------------

class TestAgnosticPolicy:
    @pytest.mark.parametrize("make_inner", [
        lambda: _randomised(SimplePolicy(), seed=5),
        lambda: _seeded_nonlinear(3),
    ])
    def test_blind_to_personality(self, make_inner):
        model = AgnosticPolicy(make_inner())
        rng = _rng(16)
        candidates = _random_options(rng, "location", 4)
        relations = _random_relations(rng, 4)
        dists = [
            predict_distribution(
                model, Personality(_rng(s).uniform(-1, 1, 5)),
                candidates, "location", relations=relations,
            )
            for s in (20, 21, 22)
        ]
        np.testing.assert_allclose(dists[0], dists[1], atol=1e-15)
        np.testing.assert_allclose(dists[0], dists[2], atol=1e-15)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_learned.py::TestAgnosticPolicy -v`
Expected: FAIL — `ImportError: cannot import name 'AgnosticPolicy'`

- [ ] **Step 3: Implement `AgnosticPolicy`**

Append to `npc_policy/learned.py`:

```python
class AgnosticPolicy(nn.Module):
    """Personality-agnostic control: zeroes ``p`` before delegating.

    Wraps either architecture at train **and** test time, so the trained control
    differs from the personality-conditioned model only in what it can see.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, batch: PolicyBatch) -> Tensor:
        return self.inner(replace(batch, p=torch.zeros_like(batch.p)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_learned.py -v`
Expected: all PASS

---

### Task 8: `kl_loss`

**Files:**
- Modify: `npc_policy/learned.py`
- Test: `tests/test_learned.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_learned.py`:

```python
from npc_policy.learned import kl_loss


# --- Task 8: KL loss ---------------------------------------------------------------

class TestKlLoss:
    def test_hand_computed_value(self):
        # case 1: q = (0.5, 0.5), t = (0.8, 0.2)  -> KL = 0.8 ln 1.6 + 0.2 ln 0.4
        # case 2 (padded to 3): q = t = (0.6, 0.4) -> KL = 0
        log_q = torch.log(torch.tensor(
            [[0.5, 0.5, 1e-300], [0.6, 0.4, 1e-300]], dtype=torch.double
        ))
        target = torch.tensor([[0.8, 0.2, 0.0], [0.6, 0.4, 0.0]], dtype=torch.double)
        mask = torch.tensor([[True, True, False], [True, True, False]])
        expected = (0.8 * np.log(0.8 / 0.5) + 0.2 * np.log(0.2 / 0.5)) / 2
        assert float(kl_loss(log_q, target, mask)) == pytest.approx(expected, abs=1e-12)

    def test_nan_safe_at_zero_target_and_padding(self):
        # padding log-probs are -inf (real model output); target 0 there must not
        # poison the loss with nan/inf
        rng = _rng(17)
        model = _randomised(SimplePolicy(), seed=6)
        case = _random_case(rng, "location", m=2)
        case["target"] = np.array([1.0, 0.0])          # a zero *inside* the mask too
        pad = _random_case(rng, "location", m=4)
        pad["target"] = np.array([0.25, 0.25, 0.25, 0.25])
        batch = PolicyBatch.from_cases([case, pad])
        loss = kl_loss(model(batch), batch.target, batch.mask)
        assert torch.isfinite(loss)

    def test_zero_when_model_matches_target(self):
        rng = _rng(18)
        cases = [_random_case(rng, "location", m=3) for _ in range(2)]
        for c in cases:
            c["target"] = np.full(3, 1 / 3)
        batch = PolicyBatch.from_cases(cases)
        loss = kl_loss(UniformBaseline()(batch), batch.target, batch.mask)
        assert float(loss) == pytest.approx(0.0, abs=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_learned.py::TestKlLoss -v`
Expected: FAIL — `ImportError: cannot import name 'kl_loss'`

- [ ] **Step 3: Implement `kl_loss`**

Append to `npc_policy/learned.py`:

```python
def kl_loss(log_q: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean-over-cases ``KL(target ‖ q)`` within the mask.

    ``xlogy`` handles ``target = 0`` inside the mask; padding positions (where
    ``log_q`` is −inf by construction) are excluded by zeroing both factors, so
    the loss stays finite and padding contributes no gradient.
    """
    t = target.masked_fill(~mask, 0.0)
    log_q_safe = log_q.masked_fill(~mask, 0.0)
    per_case = (torch.special.xlogy(t, t) - t * log_q_safe).sum(-1)
    return per_case.mean()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_learned.py -v`
Expected: all PASS

---

### Task 9: gradient smoke tests (trainability + representability)

No production code — these tests certify the learning pathway and the §4c
representability claim before any real training round.

**Files:**
- Test: `tests/test_learned.py` (append)

- [ ] **Step 1: Write the tests**

Append to `tests/test_learned.py`:

```python
from npc_policy.config import ScorerConfig
from npc_policy.scorer import HandAuthoredScorer


def _teacher_cases(rng, scorer, n, decision_type="location", m=4):
    """Random cases labelled with the scorer's full distribution (no history)."""
    cases = []
    for _ in range(n):
        candidates = _random_options(rng, decision_type, m)
        personality = Personality(rng.uniform(-1, 1, 5))
        target = scorer.distribution(personality, candidates, level=decision_type)
        selected = (
            _random_options(rng, "location", 1)[0]
            if decision_type == "action" else None
        )
        case = case_inputs(
            personality, candidates, decision_type, selected_location=selected
        )
        case["target"] = target
        cases.append(case)
    return cases


# --- Task 9: gradient smoke tests --------------------------------------------------

class TestGradientSmoke:
    def test_nonlinear_overfits_small_teacher_set(self):
        rng = _rng(19)
        torch.manual_seed(4)
        scorer = HandAuthoredScorer()                    # ideal-point teacher
        batch = PolicyBatch.from_cases(_teacher_cases(rng, scorer, n=30))
        model = NonlinearPolicy(dropout=0.0)             # overfit test: no noise
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        initial = float(kl_loss(model(batch), batch.target, batch.mask))
        for _ in range(300):
            opt.zero_grad()
            loss = kl_loss(model(batch), batch.target, batch.mask)
            loss.backward()
            opt.step()
        final = float(kl_loss(model(batch), batch.target, batch.mask))
        assert final < 0.5 * initial
        assert final < 0.05

    def test_simple_exactly_fits_bilinear_teacher(self):
        # spec §4c: with zero relations the bilinear teacher is inside the
        # simple model's representable class -> near-zero KL must be reachable
        rng = _rng(20)
        scorer = HandAuthoredScorer(config=ScorerConfig(base_form="bilinear"))
        cases = (
            _teacher_cases(rng, scorer, n=20, decision_type="location", m=4)
            + _teacher_cases(rng, scorer, n=20, decision_type="action", m=5)
        )
        batch = PolicyBatch.from_cases(cases)
        model = SimplePolicy()
        opt = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=300)

        def closure():
            opt.zero_grad()
            loss = kl_loss(model(batch), batch.target, batch.mask)
            loss.backward()
            return loss

        opt.step(closure)
        final = float(kl_loss(model(batch), batch.target, batch.mask))
        assert final < 1e-4
```

- [ ] **Step 2: Run the smoke tests**

Run: `python -m pytest tests/test_learned.py::TestGradientSmoke -v`
Expected: 2 PASS (each < ~30 s on CPU). If `test_simple_exactly_fits_bilinear_teacher`
plateaus above 1e-4, raise `max_iter` to 500 before touching anything else — the
objective is convex, so failure to converge is an optimiser budget issue, not a
model bug.

- [ ] **Step 3: Full suite**

Run: `python -m pytest tests/ -q`
Expected: all green

---

### Task 10: wrap-up

**Files:**
- Modify: `docs/specs/2026-07-09-rq2-model-layer-design.md` (status line only)

- [ ] **Step 1: Full suite, final**

Run: `python -m pytest tests/ -q`
Expected: all green (existing + new)

- [ ] **Step 2: Update the design-record status line**

In `docs/specs/2026-07-09-rq2-model-layer-design.md`, change:

```markdown
Status: design approved in conversation 2026-07-09. Scope: **model layer only**
```

to:

```markdown
Status: design approved 2026-07-09; implemented (all §5 tests green). Scope: **model layer only**
```

- [ ] **Step 3: Report**

Summarise to the user: files created/modified, test count, anything that deviated
from this plan and why.

---

## Post-review amendments (as-executed deviations)

Changes made after per-group code review and the final whole-implementation
review; where they differ from the task blocks above, the code is authoritative:

1. **Task 1 (group-A quality review):** `case_inputs` also raises `ValueError`
   when an action case lacks a `selected_location` (symmetric guard), plus test
   `test_action_case_requires_selected_location`.
2. **Task 5 / Task 9 helper (coordinator patches, folded into the blocks
   above):** action-case calls supply `selected_location` to comply with the
   guard; `_seeded_nonlinear` replaces the lambda-tuple trick in two
   parametrize lists.
3. **Task 3 (final review):** `PolicyBatch.from_cases` raises on mixed target
   presence instead of silently returning `target=None`; test
   `test_mixed_targets_raise` added.
4. **Task 9 (final review):** the representability smoke test uses
   `lambda_N = 0` at both levels (the N temperature is on §4c's
   not-representable list — with the default 1.5 the teacher is NOT inside the
   simple model's class) and 200 cases per decision type (overdetermined vs
   228 parameters, so interpolation alone cannot pass it). Final KL ≈ 2.5e-9.
   Both smoke-test KL evaluations run under `torch.no_grad()`.

Final state: 49 tests passed, 0 warnings, full suite ~6 s.

## Self-review notes

- **Spec coverage:** design §1 (modules) → Tasks 1–8; §2 (batch) → Task 3; §3
  (models) → Tasks 5–7; §4 (interfaces incl. `predict_distribution`, `kl_loss`) →
  Tasks 5, 8; §5 tests 1–6 → Tasks 4, 5/6, 6, 7, 3/5/6, 9 respectively; §6
  boundaries respected (no `experiments/` files).
- **Types:** `case_inputs` returns a plain dict consumed by
  `PolicyBatch.from_cases`; `phi_torch(p, cand, rel, ctx)` matches
  `phi_matrix(p, cand12, rel, ctx12)` argument order; `kl_loss(log_q, target,
  mask)` used identically in Tasks 8–9.
- **Known judgement calls:** `SimplePolicy` starts at zero weights (uniform
  start, convex objective); `NonlinearPolicy.double()` keeps everything float64;
  equivariance/consistency tolerances at 1e-12 (float64, exact ops expected).
