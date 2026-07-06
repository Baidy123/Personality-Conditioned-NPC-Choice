"""All provisional scorer constants in one place.

Every value here is **PROVISIONAL / TBD** (``project_flow.md`` §9). They are
collected so pilot tuning touches a single object. Defaults are deliberately
mild starting points, not validated values.

Per-decision-level coefficients (the ``lambda``s and ``tau_0``) live in
``LevelParams``; ``ScorerConfig`` holds one set for location and one for action.
Both default to identical values, so behaviour matches a single tied set until
one level is deliberately tuned (``project_flow.md`` §2/§9, the ``^d`` superscript).

v1.1 (spec: docs/specs/2026-07-02-equation-v1.1-design.md):
``lambda_E`` and ``epsilon`` were removed; ``lambda_C`` and ``base_form`` added.
v1.3 (spec: docs/specs/2026-07-06-equation-v1.3-design.md): ``lambda_Nf``
added — Neuroticism joins the familiarity channel (anxiety clings to recently
familiar options).
"""

from __future__ import annotations

from dataclasses import dataclass, field

BASE_FORMS = ("ideal_point", "bilinear")


@dataclass(frozen=True)
class LevelParams:
    """Equation coefficients for one decision level (location or action)."""

    # lambda_R / lambda_N tuned in round 1; kappa_C added in round 4 (v1.2);
    # round 7 (v1.3) rebalanced the memory term so it expresses personality
    # instead of acting as a uniform anti-repeat force: lambda_R 1.2 -> 0.7,
    # kappa_C 0.5 -> 0.8, lambda_O/lambda_C 1.0 -> 2.0, lambda_Nf added
    # (2026-07-02 / 2026-07-06, docs/tuning_log.md).
    tau_0: float = 1.0     # base sharpness knob                       [PROVISIONAL]
    lambda_R: float = 0.7  # repetition penalty (satiation)            [PROVISIONAL]
    kappa_C: float = 0.8   # C discount on satiation: high C tolerates
                           # routine, low C gets bored faster (v1.2)   [PROVISIONAL]
    lambda_O: float = 2.0  # Openness familiarity aversion             [PROVISIONAL]
    lambda_C: float = 2.0  # Conscientiousness familiarity preference  [PROVISIONAL]
    lambda_N: float = 1.5  # Neuroticism temperature scale, exp form   [PROVISIONAL]
    lambda_Nf: float = 3.0 # Neuroticism familiarity clinging (v1.3).
                           # Sits inside the N temperature, which divides
                           # logit gaps by e^lambda_N at N=+1, so the
                           # coefficient must outsize lambda_O/lambda_C
                           # to land a comparable effect.               [PROVISIONAL]

    def __post_init__(self) -> None:
        if self.tau_0 <= 0.0:
            raise ValueError("tau_0 must be > 0")
        if not (0.0 <= self.kappa_C <= 1.0):
            raise ValueError("kappa_C must be in [0, 1]")


@dataclass(frozen=True)
class ScorerConfig:
    # --- buffer lengths (project_flow.md §1) -----------------------------------
    K_L: int = 3          # recent-location FIFO length        [PROVISIONAL]
    K_A: int = 3          # local recent-action FIFO length    [PROVISIONAL]

    # --- recency weighting (relations.py) — shared across levels ---------------
    # alpha_j >= 0 and sum to 1; newest entry first. Geometric decay with this
    # ratio, then normalised. recency_decay == 1.0 gives uniform weights.
    recency_decay: float = 0.6                            # [PROVISIONAL]

    # --- base-score form (v1.1) -------------------------------------------------
    # "ideal_point" is the default; "bilinear" is a debugging fallback only.
    base_form: str = "ideal_point"

    # --- per-level equation coefficients ---------------------------------------
    # Levels started tied; round 5 (docs/tuning_log.md) sharpened the location
    # base (tau_0 0.9) after the training_yard/arena merge shrank the world and
    # pushed the openness endpoint TVD under its 0.15 floor. Grid-searched:
    # 0.95-0.9 pass everything, 0.85 collapses Shadowheart's trajectory (C3).
    location: LevelParams = field(default_factory=lambda: LevelParams(tau_0=0.9))
    action: LevelParams = field(default_factory=LevelParams)

    def __post_init__(self) -> None:
        if self.K_L < 1 or self.K_A < 1:
            raise ValueError("buffer lengths must be >= 1")
        if not (0.0 < self.recency_decay <= 1.0):
            raise ValueError("recency_decay must be in (0, 1]")
        if self.base_form not in BASE_FORMS:
            raise ValueError(f"base_form must be one of {BASE_FORMS}, got {self.base_form!r}")

    def params_for(self, level: str) -> LevelParams:
        """Return the coefficient set for ``'location'`` or ``'action'``."""
        if level == "location":
            return self.location
        if level == "action":
            return self.action
        raise ValueError(f"level must be 'location' or 'action', got {level!r}")


DEFAULT_CONFIG = ScorerConfig()
