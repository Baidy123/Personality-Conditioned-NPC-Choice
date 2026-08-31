"""Core data structures: Option, Personality, and the bounded recent buffers.

An ``Option`` stores its **native** feature vector (9 for a location, 11 for an
action) plus a ``level`` tag, and keeps a stable ``id`` so exact repetition can be
detected separately from semantic similarity. The unified
twelve-dimensional learned-model vector is produced on demand by ``to_padded12``;
it is interface padding, not stored state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .schema import (
    N_MODEL_TAGS,
    model_index,
    schema_for,
    tag_index,
)


@dataclass(frozen=True)
class Option:
    """A candidate location or action, described by its native feature vector.

    ``level`` is ``'location'`` or ``'action'`` and fixes the expected length and
    tag meanings. ``features`` has values in ``[0, 1]``.
    """

    id: str
    features: np.ndarray
    level: str

    def __post_init__(self) -> None:
        tags = schema_for(self.level)  # also validates level
        f = np.asarray(self.features, dtype=float)
        if f.shape != (len(tags),):
            raise ValueError(
                f"{self.level} features must have shape ({len(tags)},), got {f.shape}"
            )
        # Out-of-range values are clamped into [0, 1] (lenient: a typo like 1.5
        # becomes 1.0 rather than raising). A wrong feature *count* still errors.
        f = np.clip(f, 0.0, 1.0)
        object.__setattr__(self, "features", f)

    def tag(self, name: str) -> float:
        """Value of a single semantic tag, resolved within this option's schema."""
        return float(self.features[tag_index(self.level, name)])

    def to_padded12(self) -> np.ndarray:
        """Unified 12-dim model vector: native values placed by tag name, rest 0."""
        out = np.zeros(N_MODEL_TAGS, dtype=float)
        for i, name in enumerate(schema_for(self.level)):
            out[model_index(name)] = self.features[i]
        return out

    @classmethod
    def from_tags(cls, level: str, id: str, **tags: float) -> "Option":
        """Build an option from named tags; unspecified tags default to 0.0."""
        names = schema_for(level)
        vec = np.zeros(len(names), dtype=float)
        for name, value in tags.items():
            vec[tag_index(level, name)] = value
        return cls(id=id, features=vec, level=level)

    @classmethod
    def location(cls, id: str, **tags: float) -> "Option":
        return cls.from_tags("location", id, **tags)

    @classmethod
    def action(cls, id: str, **tags: float) -> "Option":
        return cls.from_tags("action", id, **tags)


@dataclass(frozen=True)
class Personality:
    """An OCEAN trait vector with values in ``[-1, 1]`` (``schema.OCEAN`` order)."""

    vector: np.ndarray

    def __post_init__(self) -> None:
        v = np.asarray(self.vector, dtype=float)
        if v.shape != (5,):
            raise ValueError(f"personality must have shape (5,), got {v.shape}")
        if np.any(v < -1.0) or np.any(v > 1.0):
            raise ValueError("OCEAN values must lie in [-1, 1]")
        object.__setattr__(self, "vector", v)

    @property
    def O(self) -> float:  # noqa: E743 - matches the design's notation
        return float(self.vector[0])

    @property
    def C(self) -> float:
        return float(self.vector[1])

    @property
    def E(self) -> float:
        return float(self.vector[2])

    @property
    def A(self) -> float:
        return float(self.vector[3])

    @property
    def N(self) -> float:
        return float(self.vector[4])

    @classmethod
    def from_traits(
        cls,
        openness: float = 0.0,
        conscientiousness: float = 0.0,
        extraversion: float = 0.0,
        agreeableness: float = 0.0,
        neuroticism: float = 0.0,
    ) -> "Personality":
        return cls(
            np.array(
                [openness, conscientiousness, extraversion, agreeableness, neuroticism],
                dtype=float,
            )
        )


@dataclass
class RecentBuffer:
    """Fixed-length FIFO of recently selected options (newest pushed last).

    Used for both the recent-location buffer ``H_t^L`` and the location-local
    recent-action buffer ``H_t^A``. The action buffer must be ``clear()``-ed by the
    decision controller whenever the selected location changes
    (see ``controller.py``).
    """

    maxlen: int
    _items: deque[Option] = field(default_factory=deque, init=False)

    def __post_init__(self) -> None:
        if self.maxlen < 1:
            raise ValueError("maxlen must be >= 1")
        self._items = deque(maxlen=self.maxlen)

    def push(self, option: Option) -> None:
        self._items.append(option)

    def clear(self) -> None:
        self._items.clear()

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def recent_to_old(self) -> list[Option]:
        """Buffer contents ordered newest -> oldest (the order ``alpha_j`` expects)."""
        return list(reversed(self._items))

    def snapshot(self) -> list[Option]:
        """Copy of the contents (oldest -> newest) for state rollback."""
        return list(self._items)

    def restore(self, items: list[Option]) -> None:
        """Replace the contents with a previous :meth:`snapshot`."""
        self._items = deque(items, maxlen=self.maxlen)

    def __len__(self) -> int:
        return len(self._items)


def stack_features(options: Sequence[Option]) -> np.ndarray:
    """Stack candidate native features into an ``(m, d)`` matrix.

    All options must share the same level (same ``d``); the caller guarantees this
    by passing one candidate set or one same-type buffer.
    """
    if not options:
        return np.empty((0, 0), dtype=float)
    return np.stack([o.features for o in options], axis=0)
