"""Derived recent-choice relation features: rep / sim / nov.

For each current candidate ``o_i`` and a relevant bounded buffer ``H`` (newest
first), with recency weights ``alpha_j >= 0`` summing to one
(``project_flow.md`` §2):

    rep_i = sum_j alpha_j * 1[id(o_i) == id(h_j)]
    sim_i = sum_j alpha_j * cosine(o_i.features, h_j.features)
    nov_i = 1 - sim_i

If the buffer is empty, ``rep = sim = nov = 0`` and no recent-choice adjustment is
applied (note: ``nov`` is 0 here, not 1 — the empty-buffer case is special-cased).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_CONFIG, ScorerConfig
from .representation import Option, RecentBuffer, stack_features


@dataclass(frozen=True)
class Relations:
    """Per-candidate relation features, each shape ``(m,)``."""

    rep: np.ndarray
    sim: np.ndarray
    nov: np.ndarray

    def as_matrix(self) -> np.ndarray:
        """Stack into ``(m, 3)`` for concatenation with option features."""
        return np.stack([self.rep, self.sim, self.nov], axis=1)


def recency_weights(n: int, decay: float) -> np.ndarray:
    """Geometric, normalised recency weights for ``n`` buffer entries (newest first).

    ``alpha_j ∝ decay**j``; ``decay == 1.0`` gives uniform weights. Returns an empty
    array for ``n == 0``.
    """
    if n == 0:
        return np.empty(0, dtype=float)
    w = decay ** np.arange(n, dtype=float)
    return w / w.sum()


def _cosine(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity between vector ``a`` (d,) and each row of ``B`` (n, d).

    A zero-norm vector yields 0 similarity rather than NaN.
    """
    a_norm = np.linalg.norm(a)
    b_norms = np.linalg.norm(B, axis=1)
    denom = a_norm * b_norms
    dots = B @ a
    out = np.zeros_like(dots)
    nz = denom > 0
    out[nz] = dots[nz] / denom[nz]
    return out


def compute_relations(
    candidates: list[Option],
    buffer: RecentBuffer,
    config: ScorerConfig = DEFAULT_CONFIG,
) -> Relations:
    """Compute rep / sim / nov for every candidate against ``buffer``."""
    m = len(candidates)
    if m == 0:
        empty = np.empty(0, dtype=float)
        return Relations(empty, empty.copy(), empty.copy())

    if buffer.is_empty():
        zeros = np.zeros(m, dtype=float)
        return Relations(zeros, zeros.copy(), zeros.copy())

    history = buffer.recent_to_old()          # newest -> oldest
    alpha = recency_weights(len(history), config.recency_decay)
    hist_feats = stack_features(history)       # (n, N_TAGS)
    hist_ids = [h.id for h in history]

    cand_feats = stack_features(candidates)    # (m, N_TAGS)

    rep = np.empty(m, dtype=float)
    sim = np.empty(m, dtype=float)
    for i, cand in enumerate(candidates):
        match = np.array([cand.id == hid for hid in hist_ids], dtype=float)
        rep[i] = float(alpha @ match)
        cos = _cosine(cand_feats[i], hist_feats)   # (n,)
        sim[i] = float(alpha @ cos)
    nov = 1.0 - sim
    return Relations(rep=rep, sim=sim, nov=nov)
