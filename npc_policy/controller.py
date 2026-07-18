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

    ``distribution`` is the policy's choice distribution over the candidate
    list. ``trace`` carries the scorer's full breakdown when the driving policy
    is the hand-authored scorer; learned policies provide no trace.
    """

    option: Option          # the chosen option
    index: int              # its index in the candidate list passed in
    distribution: np.ndarray
    trace: ScoreTrace | None = None


class DecisionController:
    """Holds ``H_t^L`` and ``H_t^A`` and runs the nested choice for one NPC,
    driven by any policy (see ``npc_policy.policies``)."""

    def __init__(
        self,
        policy,
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
        self.policy = policy
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
        self._last_location: Option | None = None    # action-context for learned policies

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

    # -- policy dispatch --------------------------------------------------------
    def _distribution(
        self,
        personality: Personality,
        candidates: list[Option],
        buffer: RecentBuffer,
        level: str,
        selected_location: Option | None,
    ) -> tuple[np.ndarray, ScoreTrace | None]:
        """Ask the policy for its choice distribution.

        Policies exposing ``trace()`` (the hand-authored scorer, or scorer
        mimics like rq2's ``StudentTraceAdapter``) keep their full trace on the
        ``Decision`` (rq1/rq2 dataset generation records it); any other policy
        follows the unified ``distribution`` convention of
        ``npc_policy.policies`` and yields ``trace=None``.
        """
        if hasattr(self.policy, "trace"):
            trace = self.policy.trace(personality, candidates, buffer=buffer, level=level)
            return trace.P_rule, trace
        dist = self.policy.distribution(
            personality, candidates, buffer=buffer, level=level,
            selected_location=selected_location,
        )
        return np.asarray(dist, dtype=float), None

    # -- decisions --------------------------------------------------------------
    def choose_location(
        self, personality: Personality, locations: list[Option]
    ) -> Decision:
        """Choose a location using ``H_t^L``; then commit it and apply the
        action-buffer reset rule."""
        # read H_L (holds L_{t-1}, L_{t-2}, ...) -> policy reweights -> pick
        dist, trace = self._distribution(personality, locations, self.H_L,
                                         "location", None)
        idx = self._select(dist)
        chosen = locations[idx]

        # reset local action buffer BEFORE the next action choice if location changed
        if chosen.id != self._last_location_id:
            self.H_A.clear()
        # commit the location into the recent-location buffer
        self.H_L.push(chosen)
        self._last_location_id = chosen.id
        self._last_location = chosen
        return Decision(option=chosen, index=idx, distribution=dist, trace=trace)

    def choose_action(
        self, personality: Personality, actions: list[Option]
    ) -> Decision:
        """Choose an action from the current location's action set using ``H_t^A``
        (empty right after a location change), then record it. Learned policies
        additionally receive the selected location as context."""
        dist, trace = self._distribution(personality, actions, self.H_A,
                                         "action", self._last_location)
        idx = self._select(dist)
        chosen = actions[idx]
        self.H_A.push(chosen)
        return Decision(option=chosen, index=idx, distribution=dist, trace=trace)

    def commit_forced(self, location: Option, action: Option) -> None:
        """Record an externally scripted, non-autonomous choice pair
        (game-layer ``force_npc`` override).

        Buffer updates follow the same structural rules as autonomous choice,
        so subsequent autonomous decisions see the world as it actually
        happened. The caller is responsible for marking the step as an
        override — forced pairs must never enter training or evaluation data.
        """
        if location.id != self._last_location_id:
            self.H_A.clear()
        self.H_L.push(location)
        self._last_location_id = location.id
        self._last_location = location
        self.H_A.push(action)

    def snapshot(self) -> dict:
        """Full mutable state (buffers, last location, rng) for rollback.

        Lets a caller stepping several NPCs make the whole cycle atomic:
        snapshot every controller, and restore on any failure."""
        return {
            "H_L": self.H_L.snapshot(),
            "H_A": self.H_A.snapshot(),
            "last_id": self._last_location_id,
            "last": self._last_location,
            "rng_state": self.rng.bit_generator.state,
        }

    def restore(self, snap: dict) -> None:
        """Return to a previous :meth:`snapshot` exactly (incl. rng stream)."""
        self.H_L.restore(snap["H_L"])
        self.H_A.restore(snap["H_A"])
        self._last_location_id = snap["last_id"]
        self._last_location = snap["last"]
        self.rng.bit_generator.state = snap["rng_state"]

    # -- lifecycle --------------------------------------------------------------
    def reset(self) -> None:
        """Clear both buffers — start a fresh behaviour sequence."""
        self.H_L.clear()
        self.H_A.clear()
        self._last_location_id = None
        self._last_location = None

    @property
    def current_location_id(self) -> str | None:
        return self._last_location_id
