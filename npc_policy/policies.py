"""Unified policy layer — one calling convention for all four policy variants.

A *policy* for sequence generation is any object exposing

    distribution(personality, candidates, buffer=None, level="location",
                 selected_location=None) -> np.ndarray   # probs over candidates

``DecisionController`` drives either a ``HandAuthoredScorer`` (special-cased so
the full ``ScoreTrace`` is still recorded) or any object with the shape above.
``LearnedPolicyAdapter`` gives trained RQ2 checkpoints that shape: it computes
relation features from the buffer exactly as the scorer does, then delegates to
``learned.predict_distribution``. Spec:
``docs/specs/2026-07-17-rq3-sequence-interface-design.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import DEFAULT_CONFIG, ScorerConfig
from .learned import AgnosticPolicy, NonlinearPolicy, SimplePolicy, predict_distribution
from .relations import compute_relations
from .representation import Option, Personality, RecentBuffer


def build_architecture(name: str) -> torch.nn.Module:
    """Fresh module for a checkpoint's ``payload["model"]`` name.

    Mirrors ``experiments.rq2.common.build_model`` without the seeding (the
    weights are immediately overwritten by ``load_state_dict``); duplicated here
    because ``npc_policy`` must not import from ``experiments``.
    """
    if name == "simple":
        return SimplePolicy()
    if name == "nonlinear":
        return NonlinearPolicy()
    if name == "agnostic_simple":
        return AgnosticPolicy(SimplePolicy())
    if name == "agnostic_nonlinear":
        return AgnosticPolicy(NonlinearPolicy())
    raise ValueError(f"unknown architecture {name!r}")


class LearnedPolicyAdapter:
    """A trained checkpoint behind the unified ``distribution`` convention."""

    def __init__(self, checkpoint_path: str | Path,
                 config: ScorerConfig = DEFAULT_CONFIG):
        self.checkpoint_path = Path(checkpoint_path)
        payload = torch.load(self.checkpoint_path, weights_only=False)
        self.architecture = payload["model"]
        self.model = build_architecture(self.architecture)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.config = config          # recency/similarity params for relations

    def distribution(
        self,
        personality: Personality,
        candidates: list[Option],
        buffer: RecentBuffer | None = None,
        level: str = "location",
        selected_location: Option | None = None,
    ) -> np.ndarray:
        relations = None
        if buffer is not None and not buffer.is_empty():
            relations = compute_relations(candidates, buffer, self.config)
        return predict_distribution(
            self.model, personality, candidates, level,
            relations=relations, selected_location=selected_location,
        )
