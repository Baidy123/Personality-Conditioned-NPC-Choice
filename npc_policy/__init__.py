"""NPC personality-conditioned choice — core representation and hand-authored scorer.

"""

from .schema import (
    LOCATION_TAGS,
    ACTION_TAGS,
    MODEL_TAGS,
    OCEAN,
    N_LOCATION_TAGS,
    N_ACTION_TAGS,
    N_MODEL_TAGS,
    N_TRAITS,
    schema_for,
    tag_index,
    model_index,
    trait_index,
)
from .config import ScorerConfig, LevelParams, DEFAULT_CONFIG
from .representation import Option, Personality, RecentBuffer
from .relations import Relations, compute_relations
from .weights import default_W_location, default_W_action
from .scorer import HandAuthoredScorer
from .controller import DecisionController, Decision
from .world import World, LocationEntry, LocalEvent, load_world, load_personalities
from .cases import ControlledCase, IndependentCase

__all__ = [
    "LOCATION_TAGS",
    "ACTION_TAGS",
    "MODEL_TAGS",
    "OCEAN",
    "N_LOCATION_TAGS",
    "N_ACTION_TAGS",
    "N_MODEL_TAGS",
    "N_TRAITS",
    "schema_for",
    "tag_index",
    "model_index",
    "trait_index",
    "ScorerConfig",
    "LevelParams",
    "DEFAULT_CONFIG",
    "Option",
    "Personality",
    "RecentBuffer",
    "Relations",
    "compute_relations",
    "default_W_location",
    "default_W_action",
    "HandAuthoredScorer",
    "DecisionController",
    "Decision",
    "World",
    "LocationEntry",
    "LocalEvent",
    "load_world",
    "load_personalities",
    "ControlledCase",
    "IndependentCase",
]
