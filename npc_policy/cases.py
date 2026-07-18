"""Decision-case data formats for both datasets (``project_flow.md`` §3).

Both datasets share the same input structure:

    input = personality
          + decision_type (location | action)
          + selected_location (for action cases; null for location cases)
          + recent_locations  (H_t^L)
          + recent_actions_same_location (H_t^A; empty after a location change)
          + candidates (the resolved option set)
          + candidate_history_features [(rep_i, sim_i, nov_i)]

They differ only in the label:

- ``ControlledCase`` carries the scorer's full ``target_distribution`` (soft label);
- ``IndependentCase`` carries a single ``target_choice`` index (provisional v1 hard
  label; pairwise / ranked forms remain open, ``project_flow.md`` §3b/§9).

``to_dict`` / ``from_dict`` give a stable, JSON-serialisable on-disk form. Provenance
fields on ``IndependentCase`` keep the manual-review trail required by the design
(no AI label is treated as ground truth without it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .relations import Relations
from .representation import Option

DecisionType = Literal["location", "action"]


# --- option (de)serialisation --------------------------------------------------
def _option_to_dict(o: Option) -> dict[str, Any]:
    return {"id": o.id, "features": o.features.tolist(), "level": o.level}


def _option_from_dict(d: dict[str, Any]) -> Option:
    return Option(
        id=d["id"],
        features=np.asarray(d["features"], dtype=float),
        level=d["level"],
    )


def _relations_to_dict(r: Relations | None) -> dict[str, list[float]] | None:
    if r is None:
        return None
    return {"rep": r.rep.tolist(), "sim": r.sim.tolist(), "nov": r.nov.tolist()}


def _relations_from_dict(d: dict[str, Any] | None) -> Relations | None:
    if d is None:
        return None
    return Relations(
        rep=np.asarray(d["rep"], dtype=float),
        sim=np.asarray(d["sim"], dtype=float),
        nov=np.asarray(d["nov"], dtype=float),
    )


@dataclass
class _CaseInput:
    personality: np.ndarray                 # (5,)
    decision_type: DecisionType
    candidates: list[Option]
    selected_location: str | None = None    # location id for action cases
    recent_locations: list[Option] = field(default_factory=list)          # H_t^L
    recent_actions_same_location: list[Option] = field(default_factory=list)  # H_t^A
    candidate_history_features: Relations | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "personality": np.asarray(self.personality, dtype=float).tolist(),
            "decision_type": self.decision_type,
            "selected_location": self.selected_location,
            "recent_locations": [_option_to_dict(o) for o in self.recent_locations],
            "recent_actions_same_location": [
                _option_to_dict(o) for o in self.recent_actions_same_location
            ],
            "candidates": [_option_to_dict(o) for o in self.candidates],
            "candidate_history_features": _relations_to_dict(
                self.candidate_history_features
            ),
        }

    @staticmethod
    def _input_from_dict(d: dict[str, Any]) -> "_CaseInput":
        return _CaseInput(
            personality=np.asarray(d["personality"], dtype=float),
            decision_type=d["decision_type"],
            candidates=[_option_from_dict(o) for o in d["candidates"]],
            selected_location=d.get("selected_location"),
            recent_locations=[
                _option_from_dict(o) for o in d.get("recent_locations", [])
            ],
            recent_actions_same_location=[
                _option_from_dict(o) for o in d.get("recent_actions_same_location", [])
            ],
            candidate_history_features=_relations_from_dict(
                d.get("candidate_history_features")
            ),
        )


@dataclass
class ControlledCase(_CaseInput):
    """Scorer-generated case with a full target distribution (soft label)."""

    target_distribution: np.ndarray = field(default_factory=lambda: np.empty(0))

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["target_distribution"] = np.asarray(
            self.target_distribution, dtype=float
        ).tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ControlledCase":
        base = cls._input_from_dict(d)
        return cls(
            **vars(base),
            target_distribution=np.asarray(d["target_distribution"], dtype=float),
        )


@dataclass
class IndependentCase(_CaseInput):
    """AI-assisted, manually reviewed case with a single hard target (v1 label)."""

    target_choice: int = -1                 # index into candidates
    # provenance / review trail (kept, not used as model input)
    source: str | None = None               # e.g. prompt/model version id
    review_status: str | None = None        # e.g. "accepted" / "rejected"
    rationale: str | None = None            # AI explanation, review only
    split: str | None = None                # "train" / "val" / "test" — keep test isolated

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            target_choice=self.target_choice,
            source=self.source,
            review_status=self.review_status,
            rationale=self.rationale,
            split=self.split,
        )
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "IndependentCase":
        base = cls._input_from_dict(d)
        return cls(
            **vars(base),
            target_choice=int(d["target_choice"]),
            source=d.get("source"),
            review_status=d.get("review_status"),
            rationale=d.get("rationale"),
            split=d.get("split"),
        )
