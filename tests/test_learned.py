"""Model-layer tests."""

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

    def test_action_case_requires_selected_location(self):
        rng = _rng(3)
        p = Personality(rng.uniform(-1, 1, 5))
        acts = _random_options(rng, "action", 2)
        with pytest.raises(ValueError):
            case_inputs(p, acts, "action")


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

    def test_mixed_targets_raise(self):
        rng = _rng(21)
        cases = [_random_case(rng, "location", m=2), _random_case(rng, "location", m=3)]
        cases[0]["target"] = np.array([0.7, 0.3])
        with pytest.raises(ValueError):
            PolicyBatch.from_cases(cases)


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


from npc_policy.config import LevelParams, ScorerConfig
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
        with torch.no_grad():
            initial = float(kl_loss(model(batch), batch.target, batch.mask))
        for _ in range(300):
            opt.zero_grad()
            loss = kl_loss(model(batch), batch.target, batch.mask)
            loss.backward()
            opt.step()
        with torch.no_grad():
            final = float(kl_loss(model(batch), batch.target, batch.mask))
        assert final < 0.5 * initial
        assert final < 0.05

    def test_simple_exactly_fits_bilinear_teacher(self):
        # with zero relations and the N temperature disabled, the
        # bilinear teacher lies inside the simple model's representable class.
        # lambda_N = 0 because the personality-dependent temperature is on the
        # spec's not-representable list; 200 cases per decision type so the fit
        # is overdetermined (>> 228 parameters) and cannot succeed by
        # interpolation alone.
        rng = _rng(20)
        cfg = ScorerConfig(
            base_form="bilinear",
            location=LevelParams(tau_0=0.9, lambda_N=0.0),
            action=LevelParams(lambda_N=0.0),
        )
        scorer = HandAuthoredScorer(config=cfg)
        cases = (
            _teacher_cases(rng, scorer, n=200, decision_type="location", m=4)
            + _teacher_cases(rng, scorer, n=200, decision_type="action", m=5)
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
        final = float(kl_loss(model(batch), batch.target, batch.mask).detach())
        assert final < 1e-4
