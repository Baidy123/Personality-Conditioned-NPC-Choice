"""Hand-authored scorer v1.1: ideal-point base + familiarity reweighting.

For a decision
level ``d in {L, A}`` over candidates ``o_1..o_m`` (native schema) with relation
features ``rep``/``fam`` from the relevant bounded buffer — ``fam`` is the
recency-weighted semantic similarity, stored as ``Relations.sim``; ``nov`` is a
learned-model input feature only:

    # base_form = "ideal_point" (default)
    mu_f(p)  = clip(b_f + Sigma_t C[t, f] * p_t, 0, 1)        # intensity features
    base_i   = - Sigma_f w_f * (o_i[f] - mu_f(p))^2
               + Sigma_t Sigma_f p_t * W_rel[t, f] * o_i[f]   # action relational features

    # base_form = "bilinear" (debugging fallback)
    base_i   = p^T W^d o_i^d

    # memory + temperature (shared by both base forms)
    gamma    = lambda_C * C - lambda_O * O + lambda_Nf * N   # v1.3: + N term
    rho      = lambda_R * (1 - kappa_C * C)          # v1.2: C-modulated satiation
    q_i      = base_i / tau_0 + gamma * fam_i - rho * rep_i
    T_N      = exp(lambda_N * N)
    P_rule   = softmax(q / T_N)

``rep`` is the repetition (satiation) penalty; since v1.2 its strength is
modulated by Conscientiousness — high C tolerates routine, low C gets bored
faster (pilot evidence: tuning round 4). ``gamma`` makes high C
favour recently similar options (routine), high O avoid them (novelty
seeking), and — since v1.3 — high N cling to them (anxiety keeps to the
recently familiar; tuning round 7). The two N channels model
different facets: the temperature is Volatility (erratic base preferences),
the familiarity term is Withdrawal/Anxiety (stick to the recent and safe).
They partially cancel in magnitude — the temperature divides the familiarity
bonus by ``e^lambda_N`` at N=+1 — which is why ``lambda_Nf`` is larger than
``lambda_O``/``lambda_C``. When the relevant buffer is empty all relation
features are 0, so ``P_rule = softmax(base / tau_0 / T_N)`` — the first action
after a location change uses the unadjusted base distribution (only the
temperature applies).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import DEFAULT_CONFIG, ScorerConfig
from .relations import Relations, compute_relations
from .representation import Option, Personality, RecentBuffer, stack_features
from .schema import n_intensity
from .weights import (
    default_b_action,
    default_b_location,
    default_C_action,
    default_C_location,
    default_w_action,
    default_w_location,
    default_W_action,
    default_W_location,
    default_W_rel_action,
)


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
    P_base: np.ndarray            # softmax(base / tau_0): memory-free distribution
    relations: Relations
    mu: np.ndarray | None         # ideal levels used; None under the bilinear fallback
    gamma: float                  # lambda_C * C - lambda_O * O + lambda_Nf * N
    q: np.ndarray
    T_N: float
    P_rule: np.ndarray


def _table(override: np.ndarray | None, default: Callable[[], np.ndarray]) -> np.ndarray:
    return default() if override is None else np.asarray(override, dtype=float)


class HandAuthoredScorer:
    """Transparent policy / controlled-learning teacher / fixed comparison baseline."""

    def __init__(
        self,
        W_location: np.ndarray | None = None,
        W_action: np.ndarray | None = None,
        b_location: np.ndarray | None = None,
        C_location: np.ndarray | None = None,
        w_location: np.ndarray | None = None,
        b_action: np.ndarray | None = None,
        C_action: np.ndarray | None = None,
        w_action: np.ndarray | None = None,
        W_rel_action: np.ndarray | None = None,
        config: ScorerConfig = DEFAULT_CONFIG,
    ):
        # bilinear fallback tables
        self.W_L = _table(W_location, default_W_location)
        self.W_A = _table(W_action, default_W_action)
        # ideal-point tables (v1.1 default form)
        self.b = {
            "location": _table(b_location, default_b_location),
            "action": _table(b_action, default_b_action),
        }
        self.C = {
            "location": _table(C_location, default_C_location),
            "action": _table(C_action, default_C_action),
        }
        self.w = {
            "location": _table(w_location, default_w_location),
            "action": _table(w_action, default_w_action),
        }
        self.W_rel_A = _table(W_rel_action, default_W_rel_action)
        self.config = config

    def W_for(self, level: str) -> np.ndarray:
        if level == "location":
            return self.W_L
        if level == "action":
            return self.W_A
        raise ValueError(f"level must be 'location' or 'action', got {level!r}")

    # -- base score -------------------------------------------------------------
    def ideal_levels(self, personality: Personality, level: str) -> np.ndarray:
        """``mu_f(p)`` over the level's intensity features."""
        return np.clip(self.b[level] + personality.vector @ self.C[level], 0.0, 1.0)

    def base_scores(
        self, personality: Personality, candidates: list[Option], level: str
    ) -> np.ndarray:
        """v1.1 base score for every candidate (form selected by ``config.base_form``)."""
        feats = stack_features(candidates)          # (m, d)
        if self.config.base_form == "bilinear":
            return feats @ (personality.vector @ self.W_for(level))
        n_int = n_intensity(level)
        mu = self.ideal_levels(personality, level)  # (n_int,)
        base = -((feats[:, :n_int] - mu) ** 2) @ self.w[level]
        if level == "action":
            base = base + feats[:, n_int:] @ (personality.vector @ self.W_rel_A)
        return base

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
        ``level`` selects the location vs action schema, tables, and coefficients.
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
        prm = cfg.params_for(level)                 # also validates level

        base = self.base_scores(personality, candidates, level)
        P_base = softmax(base, prm.tau_0)

        if relations is None:
            if buffer is None:
                m = len(candidates)
                relations = Relations(np.zeros(m), np.zeros(m), np.zeros(m))
            else:
                relations = compute_relations(candidates, buffer, cfg)

        gamma = (prm.lambda_C * personality.C - prm.lambda_O * personality.O
                 + prm.lambda_Nf * personality.N)
        rho = prm.lambda_R * (1.0 - prm.kappa_C * personality.C)
        q = base / prm.tau_0 + gamma * relations.sim - rho * relations.rep
        T_N = math.exp(prm.lambda_N * personality.N)
        P_rule = softmax(q, T_N)

        mu = None if cfg.base_form == "bilinear" else self.ideal_levels(personality, level)
        return ScoreTrace(
            base=base,
            P_base=P_base,
            relations=relations,
            mu=mu,
            gamma=gamma,
            q=q,
            T_N=T_N,
            P_rule=P_rule,
        )
