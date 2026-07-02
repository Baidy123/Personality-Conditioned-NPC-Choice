"""Decision controller — owns the two recent-choice buffers and the nested
location -> action decision flow (``project_flow.md`` §5).

This is the "manager": callers never touch the buffers directly. The controller
guarantees the structural rules that are easy to get wrong by hand:

- relation features are read from the relevant buffer *before* the scorer runs,
  then the selected option is appended *after* (the buffer holds past choices);
- when the chosen location differs from the previous one, the local action buffer
  ``H_t^A`` is cleared *before* the next action choice, so the first action at a new
  location uses the base distribution (no recent-action context).

The controller drives one NPC's behaviour over successive decision cycles. For
matched-case analysis (RQ1) where buffers are set up by hand, call the scorer
directly instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .config import DEFAULT_CONFIG, ScorerConfig
from .representation import Option, Personality, RecentBuffer
from .scorer import HandAuthoredScorer, ScoreTrace

SelectionMode = Literal["argmax", "sample"]


@dataclass(frozen=True)
class Decision:
    """Outcome of one location or action choice.

    ``trace`` carries the full scorer breakdown (base / P_base / relations /
    P_rule), which is what dataset generation records.
    """

    option: Option          # the chosen option
    index: int              # its index in the candidate list passed in
    trace: ScoreTrace       # scorer intermediates, incl. P_rule

    @property
    def distribution(self) -> np.ndarray:
        return self.trace.P_rule


class DecisionController:
    """Holds ``H_t^L`` and ``H_t^A`` and runs the nested choice for one NPC."""

    def __init__(
        self,
        scorer: HandAuthoredScorer,
        config: ScorerConfig = DEFAULT_CONFIG,
        mode: SelectionMode = "argmax",
        rng: np.random.Generator | None = None,
        selection_temperature: float = 1.0,
        min_p: float = 0.0,
    ):
        if mode not in ("argmax", "sample"):
            raise ValueError("mode must be 'argmax' or 'sample'")
        if selection_temperature <= 0.0:
            raise ValueError("selection_temperature must be > 0")
        if not (0.0 <= min_p < 1.0):
            raise ValueError("min_p must be in [0, 1)")
        self.scorer = scorer
        self.config = config
        self.mode = mode
        self.rng = rng if rng is not None else np.random.default_rng()
        # Sampling sharpness: draw from P_rule ** (1 / T) renormalised. T = 1 leaves
        # the distribution unchanged; T < 1 sharpens (suppresses the low-prob tail);
        # T -> 0 approaches argmax. Only affects ``mode == 'sample'``.
        self.selection_temperature = selection_temperature
        # Game-layer fuse: options below this probability (in P_rule, before
        # sharpening) are excluded from sampling. Presentation-level truncation
        # only — the recorded distribution / research object stays P_rule.
        self.min_p = min_p

        self.H_L = RecentBuffer(maxlen=config.K_L)   # recent locations
        self.H_A = RecentBuffer(maxlen=config.K_A)   # local recent actions
        self._last_location_id: str | None = None

    # -- selection --------------------------------------------------------------
    def _select(self, dist: np.ndarray) -> int:
        if self.mode == "argmax":
            return int(np.argmax(dist))
        p = np.asarray(dist, dtype=float)
        if self.min_p > 0.0:                         # game-layer fuse
            p = np.where(p >= self.min_p, p, 0.0)
            p = p / p.sum()                          # >= 1 entry survives (max >= mean)
        if self.selection_temperature != 1.0:        # sharpen before sampling
            p = p ** (1.0 / self.selection_temperature)
            p = p / p.sum()
        return int(self.rng.choice(len(p), p=p))

    # -- decisions --------------------------------------------------------------
    def choose_location(
        self, personality: Personality, locations: list[Option]
    ) -> Decision:
        """Choose a location using ``H_t^L``; then commit it and apply the
        action-buffer reset rule."""
        # read H_L (holds L_{t-1}, L_{t-2}, ...) -> scorer reweights -> pick
        trace = self.scorer.trace(personality, locations, buffer=self.H_L, level="location")
        idx = self._select(trace.P_rule)
        chosen = locations[idx]

        # reset local action buffer BEFORE the next action choice if location changed
        if chosen.id != self._last_location_id:
            self.H_A.clear()
        # commit the location into the recent-location buffer
        self.H_L.push(chosen)
        self._last_location_id = chosen.id
        return Decision(option=chosen, index=idx, trace=trace)

    def choose_action(
        self, personality: Personality, actions: list[Option]
    ) -> Decision:
        """Choose an action from the current location's action set using ``H_t^A``
        (empty right after a location change), then record it."""
        trace = self.scorer.trace(personality, actions, buffer=self.H_A, level="action")
        idx = self._select(trace.P_rule)
        chosen = actions[idx]
        self.H_A.push(chosen)
        return Decision(option=chosen, index=idx, trace=trace)

    # -- lifecycle --------------------------------------------------------------
    def reset(self) -> None:
        """Clear both buffers — start a fresh behaviour sequence."""
        self.H_L.clear()
        self.H_A.clear()
        self._last_location_id = None

    @property
    def current_location_id(self) -> str | None:
        return self._last_location_id
