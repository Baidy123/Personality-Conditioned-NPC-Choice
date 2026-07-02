"""Provisional trait-to-feature weight tables ``W^L`` and ``W^A`` (``project_flow.md`` §2).

``W^L`` is ``(N_TRAITS, 8)`` over ``LOCATION_TAGS``; ``W^A`` is ``(N_TRAITS, 11)`` over
``ACTION_TAGS``. Rows follow ``OCEAN`` order. The base score is
``base_i^d = p^T W^d o_i^d``. The same E row also drives the history term
``compat_E_i^d = W^d[E, :] · o_i^d``.

**All values are hand-authored starting points (PROVISIONAL, ``[values TBD]`` in
§2).** They encode plausible directions only and must be examined empirically
(RQ1), not assumed correct. The two tables are independent because a feature can
act differently at the two stages.

Design decisions baked in here:
- The Extraversion (E) row is deliberately **sparse** (approach A): only the
  strongly E-related features are non-zero, so both the base score and ``compat_E``
  stay interpretable. Location E: social, stimulation, privacy. Action E: social,
  stimulation, cooperation.
- v1.1: ``W^L`` / ``W^A`` are kept only as the bilinear fallback; the default
  ideal-point form uses ``b``/``C``/``w`` (+ ``W_rel`` for actions) below.
"""

from __future__ import annotations

import numpy as np

from .schema import ACTION_TAGS, LOCATION_TAGS, OCEAN

# LOCATION_TAGS = [social, stimulation, structure, cognitive, physical, risk, exploration, privacy]
_W_L = {
    #                    soc   stim  struct cog   phys  risk  expl  priv
    "openness":         [ 0.0,  0.1, -0.3,  0.6,  0.0,  0.3,  0.8,  0.2],
    "conscientiousness":[ 0.0, -0.3,  0.8,  0.4,  0.1, -0.5, -0.1,  0.1],
    "extraversion":     [ 0.8,  0.6,  0.0,  0.0,  0.0,  0.0,  0.0, -0.7],  # sparse (A)
    "agreeableness":    [ 0.3,  0.0,  0.1,  0.0,  0.0, -0.1,  0.0, -0.1],
    "neuroticism":      [-0.3, -0.3,  0.3,  0.1,  0.0, -0.4, -0.2,  0.5],
}

# ACTION_TAGS = [social, stimulation, structure, cognitive, physical, risk, exploration,
#                cooperation, helping, conflict, control]
_W_A = {
    #                    soc   stim  struct cog   phys  risk  expl  coop  help  confl ctrl
    "openness":         [ 0.0,  0.1, -0.3,  0.6,  0.0,  0.3,  0.8,  0.1,  0.1,  0.0,  0.0],
    "conscientiousness":[ 0.0, -0.3,  0.8,  0.4,  0.1, -0.5, -0.1,  0.2,  0.2, -0.3,  0.3],
    "extraversion":     [ 0.8,  0.6,  0.0,  0.0,  0.0,  0.0,  0.0,  0.5,  0.0,  0.0,  0.0],  # sparse (A)
    # A->conflict -0.7 -> -0.9 in tuning round 3: low-A combat drive is the only
    # action-level channel for "loves legitimate fighting" (docs/tuning_log.md).
    "agreeableness":    [ 0.3,  0.0,  0.1,  0.0,  0.0, -0.1,  0.0,  0.6,  0.8, -0.9, -0.2],
    "neuroticism":      [-0.3, -0.3,  0.3,  0.1,  0.0, -0.4, -0.2,  0.0,  0.1,  0.2, -0.1],
}


def default_W_location() -> np.ndarray:
    """Return the provisional ``(N_TRAITS, 8)`` location weight matrix."""
    return np.array([_W_L[trait] for trait in OCEAN], dtype=float)


def default_W_action() -> np.ndarray:
    """Return the provisional ``(N_TRAITS, 11)`` action weight matrix."""
    return np.array([_W_A[trait] for trait in OCEAN], dtype=float)


# --- v1.1 ideal-point tables (PROVISIONAL; draft values from docs/v11_tables.png) ---
# b: baseline ideal level per intensity feature (neutral personality).
# C: trait -> ideal-level shift (rows in OCEAN order).
# w: deviation cost per intensity feature.
# Action level reuses the location drafts without the privacy column; refine
# per-level during pilot tuning. W_rel (linear, action relational features) is
# seeded from the last four columns of the fallback W^A.

# LOCATION_TAGS = [social, stimulation, structure, cognitive, physical, risk, exploration, privacy]
# Tuning round 1 (2026-07-02, docs/tuning_log.md): b_social 0.50->0.475,
# w_social 1.0->0.8, C[O,cognitive] 0.30->0.45, w_cognitive 0.6->0.7.
_B_L = [0.475, 0.50, 0.50, 0.40, 0.40, 0.20, 0.40, 0.40]
_C_L = {
    #                    soc    stim   struct cog    phys   risk   expl   priv
    "openness":         [ 0.00,  0.10, -0.20,  0.45,  0.00,  0.20,  0.50,  0.00],
    "conscientiousness":[ 0.00, -0.20,  0.40,  0.10,  0.10, -0.30, -0.10,  0.10],
    "extraversion":     [ 0.45,  0.35,  0.00,  0.00,  0.00,  0.00,  0.00, -0.40],
    "agreeableness":    [ 0.15,  0.00,  0.10,  0.00,  0.00, -0.10,  0.00, -0.05],
    "neuroticism":      [-0.15, -0.20,  0.15,  0.00,  0.00, -0.35, -0.10,  0.30],
}
_W_COST_L = [0.8, 0.8, 0.8, 0.7, 0.5, 0.9, 0.6, 0.7]

_B_A = _B_L[:7]
_C_A = {trait: row[:7] for trait, row in _C_L.items()}
_W_COST_A = _W_COST_L[:7]


def default_b_location() -> np.ndarray:
    """Provisional ``(8,)`` baseline ideal levels for location intensity features."""
    return np.array(_B_L, dtype=float)


def default_C_location() -> np.ndarray:
    """Provisional ``(5, 8)`` trait-to-ideal-shift matrix for locations."""
    return np.array([_C_L[trait] for trait in OCEAN], dtype=float)


def default_w_location() -> np.ndarray:
    """Provisional ``(8,)`` deviation costs for location intensity features."""
    return np.array(_W_COST_L, dtype=float)


def default_b_action() -> np.ndarray:
    """Provisional ``(7,)`` baseline ideal levels for action intensity features."""
    return np.array(_B_A, dtype=float)


def default_C_action() -> np.ndarray:
    """Provisional ``(5, 7)`` trait-to-ideal-shift matrix for actions."""
    return np.array([_C_A[trait] for trait in OCEAN], dtype=float)


def default_w_action() -> np.ndarray:
    """Provisional ``(7,)`` deviation costs for action intensity features."""
    return np.array(_W_COST_A, dtype=float)


def default_W_rel_action() -> np.ndarray:
    """Provisional ``(5, 4)`` linear weights for the action relational features."""
    return default_W_action()[:, 7:]


# Validate shapes at import so a typo in any table fails loudly.
assert default_W_location().shape == (len(OCEAN), len(LOCATION_TAGS))
assert default_W_action().shape == (len(OCEAN), len(ACTION_TAGS))
assert default_b_location().shape == (len(LOCATION_TAGS),)
assert default_C_location().shape == (len(OCEAN), len(LOCATION_TAGS))
assert default_w_location().shape == (len(LOCATION_TAGS),)
assert default_b_action().shape == (7,)
assert default_C_action().shape == (len(OCEAN), 7)
assert default_w_action().shape == (7,)
assert default_W_rel_action().shape == (len(OCEAN), 4)
