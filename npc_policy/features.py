"""Learned-model input features — the fixed per-candidate layout ``phi_i``.

This module is the numpy reference for the feature order; ``learned.py`` rebuilds the same
layout in torch (kept in sync by ``tests/test_learned.py``).

``phi_i`` contains only candidate-varying terms — the softmax over one candidate
set cancels anything constant across candidates, so pure functions of the
personality, the decision type, or the selected-location context are excluded:

    phi_i = concat( o_i,              # 12  padded candidate features
                    o_i²,             # 12  elementwise squares (ideal-point base)
                    rel_i,            #  3  rep, sim, nov
                    vec(p ⊗ o_i),     # 60  trait-major bilinear block
                    vec(p ⊗ rel_i),   # 15  trait-major relation block
                    c_L ⊙ o_i )       # 12  selected-location × candidate match
                                      # ---
                                      # 114 = PHI_DIM
"""

from __future__ import annotations

import numpy as np

from .relations import Relations
from .representation import Option, Personality
from .schema import N_MODEL_TAGS

N_TRAITS = 5
N_RELATIONS = 3
PHI_DIM = 2 * N_MODEL_TAGS + N_RELATIONS + N_TRAITS * (N_MODEL_TAGS + N_RELATIONS) + N_MODEL_TAGS

DECISION_TYPES = ("location", "action")


def decision_type_index(decision_type: str) -> int:
    """0 for ``location``, 1 for ``action`` — the weight-table row / one-hot index."""
    try:
        return DECISION_TYPES.index(decision_type)
    except ValueError:
        raise ValueError(
            f"decision_type must be one of {DECISION_TYPES}, got {decision_type!r}"
        ) from None


def padded_candidates(candidates: list[Option]) -> np.ndarray:
    """Stack candidates into the unified ``(m, 12)`` model matrix."""
    if not candidates:
        raise ValueError("candidate set must be non-empty")
    return np.stack([o.to_padded12() for o in candidates], axis=0)


def context_vector(selected_location: Option | None) -> np.ndarray:
    """Padded selected-location context ``c_L``; zeros for location cases."""
    if selected_location is None:
        return np.zeros(N_MODEL_TAGS, dtype=float)
    if selected_location.level != "location":
        raise ValueError("selected-location context must be a location option")
    return selected_location.to_padded12()


def relations_matrix(relations: Relations | None, m: int) -> np.ndarray:
    """``(m, 3)`` rep/sim/nov matrix; all-zero when no relations are given
    (empty buffer or the no-context ablation)."""
    if relations is None:
        return np.zeros((m, N_RELATIONS), dtype=float)
    mat = relations.as_matrix()
    if mat.shape != (m, N_RELATIONS):
        raise ValueError(f"relations shape {mat.shape} != ({m}, {N_RELATIONS})")
    return mat


def phi_matrix(
    p: np.ndarray,          # (5,)
    cand12: np.ndarray,     # (m, 12)
    rel: np.ndarray,        # (m, 3)
    ctx12: np.ndarray,      # (12,)
) -> np.ndarray:
    """Build the ``(m, PHI_DIM)`` simple-policy feature matrix.

    Bilinear blocks are trait-major: entry ``t * n + f`` is ``p[t] * x[f]``.
    """
    p = np.asarray(p, dtype=float).reshape(N_TRAITS)
    cand12 = np.asarray(cand12, dtype=float)
    rel = np.asarray(rel, dtype=float)
    ctx12 = np.asarray(ctx12, dtype=float).reshape(N_MODEL_TAGS)
    m = cand12.shape[0]

    p_x_o = np.einsum("t,mf->mtf", p, cand12).reshape(m, N_TRAITS * N_MODEL_TAGS)
    p_x_rel = np.einsum("t,mr->mtr", p, rel).reshape(m, N_TRAITS * N_RELATIONS)
    return np.concatenate(
        [cand12, cand12**2, rel, p_x_o, p_x_rel, ctx12[None, :] * cand12], axis=1
    )


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
    Location cases must not carry a ``selected_location``; action cases must.
    """
    if decision_type == "location" and selected_location is not None:
        raise ValueError("location cases use an empty selected-location context")
    if decision_type == "action" and selected_location is None:
        raise ValueError("action cases require the selected location's context")
    cand12 = padded_candidates(candidates)
    return {
        "p": np.asarray(personality.vector, dtype=float),
        "d": decision_type_index(decision_type),
        "ctx": context_vector(selected_location),
        "cand": cand12,
        "rel": relations_matrix(relations, cand12.shape[0]),
    }
