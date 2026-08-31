"""RQ2 learned policies — the only torch module in ``npc_policy``.


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
        padded into ``batch.target``; if **no** case does, ``target`` is ``None``;
        a mix raises (a partially labelled batch would silently disable the loss).
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

        n_with = sum("target" in c for c in cases)
        if 0 < n_with < len(cases):
            raise ValueError("either every case or no case may carry a 'target'")
        with_target = n_with == len(cases)
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


class NonlinearPolicy(nn.Module):
    """Dual-tower compatibility network (sizes provisional).

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
