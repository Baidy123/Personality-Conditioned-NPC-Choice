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
    "agreeableness":    [ 0.3,  0.0,  0.1,  0.0,  0.0, -0.1,  0.0,  0.6,  0.8, -0.7, -0.2],
    "neuroticism":      [-0.3, -0.3,  0.3,  0.1,  0.0, -0.4, -0.2,  0.0,  0.1,  0.2, -0.1],
}


def default_W_location() -> np.ndarray:
    """Return the provisional ``(N_TRAITS, 8)`` location weight matrix."""
    return np.array([_W_L[trait] for trait in OCEAN], dtype=float)


def default_W_action() -> np.ndarray:
    """Return the provisional ``(N_TRAITS, 11)`` action weight matrix."""
    return np.array([_W_A[trait] for trait in OCEAN], dtype=float)


# Validate shapes at import so a typo in either table fails loudly.
assert default_W_location().shape == (len(OCEAN), len(LOCATION_TAGS))
assert default_W_action().shape == (len(OCEAN), len(ACTION_TAGS))
