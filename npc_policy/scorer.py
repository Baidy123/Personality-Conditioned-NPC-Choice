"""Hand-authored scorer: base personality-option score + recent-choice reweighting.

Implements the provisional equations of ``project_flow.md`` §2. For a decision
level ``d in {L, A}`` over candidates ``o_1..o_m`` (in their native schema) with
relation features ``rep/sim/nov`` from the relevant bounded buffer:

    base_i      = p^T W^d o_i^d
    P_base      = softmax(base / tau_0)

    compat_E_i  = W^d[E, :] · o_i^d          # E-row compatibility (history term)

    q_i         = log(P_base_i + epsilon)
                  - lambda_R * rep_i
                  + lambda_O * O * nov_i
                  + lambda_E * E * compat_E_i * sim_i

    T_N         = 1 + lambda_N * max(N, 0)
    P_rule      = softmax(q / T_N)

``W^L`` (location) and ``W^A`` (action) are separate; coefficients are per-level
(``config.LevelParams``). When the relevant buffer is empty all relation features
are 0, so ``P_rule == P_base`` (the first action after a location change uses the
base distribution).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_CONFIG, ScorerConfig
from .relations import Relations, compute_relations
from .representation import Option, Personality, RecentBuffer, stack_features
from .schema import trait_index
from .weights import default_W_action, default_W_location

_E_IDX = trait_index("extraversion")


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax over a 1-D array."""
    z = np.asarray(x, dtype=float) / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


@dataclass
class ScoreTrace:
    """Intermediate quantities, kept for the RQ1 transparency analysis."""

    base: np.ndarray
    P_base: np.ndarray
    relations: Relations
    compat_E: np.ndarray
    q: np.ndarray
    T_N: float
    P_rule: np.ndarray


class HandAuthoredScorer:
    """Transparent policy / controlled-learning teacher / fixed comparison baseline."""

    def __init__(
        self,
        W_location: np.ndarray | None = None,
        W_action: np.ndarray | None = None,
        config: ScorerConfig = DEFAULT_CONFIG,
    ):
        self.W_L = default_W_location() if W_location is None else np.asarray(W_location, dtype=float)
        self.W_A = default_W_action() if W_action is None else np.asarray(W_action, dtype=float)
        self.config = config

    def W_for(self, level: str) -> np.ndarray:
        if level == "location":
            return self.W_L
        if level == "action":
            return self.W_A
        raise ValueError(f"level must be 'location' or 'action', got {level!r}")

    # -- base score -------------------------------------------------------------
    def base_scores(
        self, personality: Personality, candidates: list[Option], level: str
    ) -> np.ndarray:
        """``base_i = p^T W^d o_i^d`` for every candidate."""
        W = self.W_for(level)
        feats = stack_features(candidates)         # (m, d)
        wp = personality.vector @ W                 # (d,)
        return feats @ wp                           # (m,)

    # -- full distribution ------------------------------------------------------
    def distribution(
        self,
        personality: Personality,
        candidates: list[Option],
        buffer: RecentBuffer | None = None,
        relations: Relations | None = None,
        level: str = "location",
    ) -> np.ndarray:
        """Return the reweighted choice distribution ``P_rule`` over candidates.

        Provide either a ``buffer`` (relations are computed from it) or pre-computed
        ``relations``. With neither, relations are all-zero (no-context ablation).
        ``level`` selects the location vs action schema, weights, and coefficients.
        """
        return self.trace(personality, candidates, buffer, relations, level).P_rule

    def trace(
        self,
        personality: Personality,
        candidates: list[Option],
        buffer: RecentBuffer | None = None,
        relations: Relations | None = None,
        level: str = "location",
    ) -> ScoreTrace:
        """Like :meth:`distribution` but returns all intermediate quantities."""
        if not candidates:
            raise ValueError("candidate set must be non-empty")

        cfg = self.config
        prm = cfg.params_for(level)
        W = self.W_for(level)

        feats = stack_features(candidates)          # (m, d)
        base = feats @ (personality.vector @ W)     # (m,)
        P_base = softmax(base, prm.tau_0)

        if relations is None:
            if buffer is None:
                m = len(candidates)
                relations = Relations(np.zeros(m), np.zeros(m), np.zeros(m))
            else:
                relations = compute_relations(candidates, buffer, cfg)

        # E-row compatibility, reused from the base table (no second weight table).
        compat_E = feats @ W[_E_IDX, :]             # (m,)

        q = (
            np.log(P_base + cfg.epsilon)
            - prm.lambda_R * relations.rep
            + prm.lambda_O * personality.O * relations.nov
            + prm.lambda_E * personality.E * compat_E * relations.sim
        )
        T_N = 1.0 + prm.lambda_N * max(personality.N, 0.0)
        P_rule = softmax(q, T_N)

        return ScoreTrace(
            base=base,
            P_base=P_base,
            relations=relations,
            compat_E=compat_E,
            q=q,
            T_N=T_N,
            P_rule=P_rule,
        )
