"""Semantic schemas (location / action) and OCEAN personality axes.

Location and action use separate native schemas that share their first seven
features (``project_flow.md`` §1, ``[SET as v1]``):

- location (8): social, stimulation, structure, cognitive, physical, risk,
  exploration, privacy
- action (11): the shared seven, then cooperation, helping, conflict, control

A unified twelve-dimensional order (``MODEL_TAGS``) is used only at the learned-
model interface: location pads the four action-only fields with zeros, action
pads ``privacy`` with zero. The hand-authored scorer works in the native schemas.

All feature values lie in ``[0, 1]``; OCEAN trait values lie in ``[-1, 1]``.
"""

from __future__ import annotations

# --- shared first seven --------------------------------------------------------
_SHARED = (
    "social",
    "stimulation",
    "structure",
    "cognitive",
    "physical",
    "risk",
    "exploration",
)

# --- native schemas ------------------------------------------------------------
LOCATION_TAGS: tuple[str, ...] = (*_SHARED, "privacy")                       # 8
ACTION_TAGS: tuple[str, ...] = (*_SHARED, "cooperation", "helping", "conflict", "control")  # 11

# --- unified learned-model order (interface only) ------------------------------
MODEL_TAGS: tuple[str, ...] = (
    *_SHARED,
    "privacy",
    "cooperation",
    "helping",
    "conflict",
    "control",
)  # 12

# --- v1.1 base-form split (project_flow.md §2) ----------------------------------
# Intensity features use the ideal-point base form; the four action-only
# relational features keep a linear form. Layout guarantee (relied on by the
# scorer): in the native action vector the intensity block comes first and the
# relational block last.
INTENSITY_TAGS: dict[str, tuple[str, ...]] = {
    "location": LOCATION_TAGS,   # all 8
    "action": _SHARED,           # shared 7 (privacy is location-only)
}
RELATIONAL_TAGS: tuple[str, ...] = ("cooperation", "helping", "conflict", "control")


def n_intensity(level: str) -> int:
    """Number of intensity features at the start of a level's native vector."""
    try:
        return len(INTENSITY_TAGS[level])
    except KeyError:
        raise KeyError(f"level must be 'location' or 'action', got {level!r}") from None


# --- personality axes (OCEAN) --------------------------------------------------
OCEAN: tuple[str, ...] = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)

SCHEMA: dict[str, tuple[str, ...]] = {"location": LOCATION_TAGS, "action": ACTION_TAGS}

N_LOCATION_TAGS = len(LOCATION_TAGS)
N_ACTION_TAGS = len(ACTION_TAGS)
N_MODEL_TAGS = len(MODEL_TAGS)
N_TRAITS = len(OCEAN)

_TRAIT_INDEX = {name: i for i, name in enumerate(OCEAN)}
_TAG_INDEX = {
    "location": {name: i for i, name in enumerate(LOCATION_TAGS)},
    "action": {name: i for i, name in enumerate(ACTION_TAGS)},
}
_MODEL_INDEX = {name: i for i, name in enumerate(MODEL_TAGS)}


def schema_for(level: str) -> tuple[str, ...]:
    """Native tag tuple for ``'location'`` or ``'action'``."""
    try:
        return SCHEMA[level]
    except KeyError:
        raise KeyError(f"level must be 'location' or 'action', got {level!r}") from None


def tag_index(level: str, name: str) -> int:
    """Column index of a tag within a level's native feature vector / ``W^d``."""
    try:
        return _TAG_INDEX[level][name]
    except KeyError:
        raise KeyError(f"unknown tag {name!r} for level {level!r}") from None


def model_index(name: str) -> int:
    """Column index of a tag in the unified 12-dim model vector."""
    return _MODEL_INDEX[name]


def trait_index(name: str) -> int:
    """Row index of an OCEAN trait in a personality vector / in ``W``."""
    try:
        return _TRAIT_INDEX[name]
    except KeyError:
        raise KeyError(f"unknown trait {name!r}; valid traits: {OCEAN}") from None
