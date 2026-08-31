"""Study 2B independent-dataset library: parse, validate, enrich, model inputs.

Raw batches are names-only JSON carrying an externally authored choice label;
this module owns every number: feature vectors come from ``data/world.json`` and
rep/sim/nov from ``npc_policy.relations``. The scorer plays no part here.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from npc_policy import DEFAULT_CONFIG, IndependentCase, RecentBuffer, World, compute_relations, load_world
from npc_policy.representation import Personality

from .common import CODE, DATA, case_input_dict

IND_DATA = DATA / "rq2_independent"
IND_RESULTS = CODE / "results" / "rq2b"
IMPORT_SEED = 20260711

TRAITS = ("O", "C", "E", "A", "N")
_TRAIT_KW = {"O": "openness", "C": "conscientiousness", "E": "extraversion",
             "A": "agreeableness", "N": "neuroticism"}

# split targets (train/val raised 2026-07-12 — the
# 2026-07-09 550/100 was a review-throughput guess, not a scientific target);
# general pool scales proportionally to these
GENERAL_TARGETS = {"train": 1200, "val": 150, "test_iid": 75}
STRUCT_TARGETS = {"test_pers": 38, "test_family": 37}
TEST_GROUPS = ("test_iid", "test_pers", "test_family")

# held-out family (amendment 2026-07-12): a test-only location replaces the
# arena holdout, so training keeps all six deployment locations. Authored in
# data/rq2b_test_world.json; never in world.json (2A/RQ1 hashes untouched).
TEST_LOCATION = "infirmary"

WD_GRID = (1e-4, 1e-3, 1e-2)        # nonlinear-family sweep, chosen on val NLL
DATA_SIZES_2B = (150, 400, 800)     # data-size curve; full-data point reuses main runs


@lru_cache(maxsize=1)
def load_2b_world() -> World:
    """Base deployment world plus the test-only location extension."""
    base = load_world(DATA / "world.json")
    ext = load_world(DATA / "rq2b_test_world.json")
    return World(entries={**base.entries, **ext.entries})


# ------------------------------------------------------------------ validate --
def validate_case(raw: dict, world: World) -> str | None:
    """Reason code for a rule violation, or ``None`` if valid."""
    p = raw.get("personality")
    if not isinstance(p, dict) or set(p) != set(TRAITS):
        return "traits_missing"
    if any(not isinstance(v, (int, float)) or not -1.0 <= v <= 1.0
           for v in p.values()):
        return "trait_range"

    dt = raw.get("decision_type")
    if dt not in ("location", "action"):
        return "bad_decision_type"

    unlocked = {o.id for o in world.resolve()}
    recent = raw.get("recent_locations") or []
    if any(name not in unlocked for name in recent):
        return "unknown_location"
    if len(recent) > DEFAULT_CONFIG.K_L:
        return "history_too_long"

    cands = raw.get("candidates") or []
    if len(set(cands)) != len(cands):
        return "duplicate_candidates"

    if dt == "location":
        if raw.get("selected_location") is not None:
            return "location_case_has_selected"
        if raw.get("recent_actions_same_location"):
            return "location_case_has_actions"
        if not 2 <= len(cands) <= len(unlocked):
            return "candidate_count"
        if any(name not in unlocked for name in cands):
            return "unknown_location"
    else:
        sel = raw.get("selected_location")
        if sel not in unlocked:
            return "missing_selected_location"
        if recent and recent[-1] != sel:
            return "selected_location_mismatch"
        native = {a.id for a in world.actions_at(sel)}
        if set(cands) != native:
            return "not_full_action_set"
        acts = raw.get("recent_actions_same_location") or []
        if len(acts) > DEFAULT_CONFIG.K_A:
            return "history_too_long"
        if any(a not in native for a in acts):
            return "action_not_native"

    if raw.get("choice") not in cands:
        return "choice_not_in_candidates"
    return None


# --------------------------------------------------------------------- parse --
def parse_raw_file(path: Path) -> tuple[str, list[tuple[int, dict]], list[dict]]:
    """One raw batch file → (source, [(array position, case)], user-rejected).

    The optional ``{"_meta": {"source": …}}`` header names the labelling model;
    elements carrying ``"review_status": "rejected"`` are the user's manual
    rejections (kept for the audit trail, never imported). Cases keep their raw
    array position so ids stay stable when the user later rejects an earlier
    element in the same file (review finding, 2026-07-11).
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path.name}: expected a JSON array")
    source = "unknown"
    cases, user_rejected = [], []
    for pos, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{path.name}: non-object array element")
        if "_meta" in item:
            meta = item["_meta"]
            if isinstance(meta, dict):
                source = str(meta.get("source", "unknown"))
            else:
                source = str(meta)
        elif item.get("review_status") == "rejected":
            user_rejected.append(item)
        else:
            cases.append((pos, item))
    return source, cases, user_rejected


# raw-format keys; anything else (e.g. hallucinated feature numbers) is ignored
# with a warning at import
KNOWN_KEYS = {"personality", "decision_type", "selected_location",
              "recent_locations", "recent_actions_same_location",
              "candidates", "choice", "reason", "review_status"}


# -------------------------------------------------------------------- enrich --
def _buffer(options, maxlen):
    buf = RecentBuffer(maxlen=maxlen)
    for o in options:                       # oldest → newest (pool convention)
        buf.push(o)
    return buf


def enrich_case(raw: dict, world: World, source: str) -> IndependentCase:
    """Names → numbers: features from the world, relations from the formula.

    Assumes ``validate_case`` returned ``None``. For an action case with empty
    ``recent_locations`` the selected location is auto-filled as the sole entry
    (2A invariant: newest recent location == selected location).
    """
    p = Personality.from_traits(
        **{_TRAIT_KW[k]: float(v) for k, v in raw["personality"].items()})
    recent_names = list(raw.get("recent_locations") or [])
    if raw["decision_type"] == "location":
        cands = [world.effective_location(n) for n in raw["candidates"]]
        history = [world.effective_location(n) for n in recent_names]
        rel = (compute_relations(cands, _buffer(history, DEFAULT_CONFIG.K_L))
               if history else None)
        return IndependentCase(
            personality=p.vector, decision_type="location", candidates=cands,
            recent_locations=history, candidate_history_features=rel,
            target_choice=raw["candidates"].index(raw["choice"]),
            source=source, review_status="accepted", rationale=raw.get("reason"),
        )
    sel = raw["selected_location"]
    if not recent_names:
        recent_names = [sel]
    by_id = {a.id: a for a in world.actions_at(sel)}
    cands = [by_id[n] for n in raw["candidates"]]
    acts = [by_id[n] for n in (raw.get("recent_actions_same_location") or [])]
    rel = (compute_relations(cands, _buffer(acts, DEFAULT_CONFIG.K_A))
           if acts else None)
    return IndependentCase(
        personality=p.vector, decision_type="action", candidates=cands,
        selected_location=sel,
        recent_locations=[world.effective_location(n) for n in recent_names],
        recent_actions_same_location=acts, candidate_history_features=rel,
        target_choice=raw["candidates"].index(raw["choice"]),
        source=source, review_status="accepted", rationale=raw.get("reason"),
    )


# ------------------------------------------------------------- split filters --
def in_pers_region(case: IndependentCase) -> bool:
    """G1 region O > 0.5 ∧ C < −0.5 (same thresholds as 2A)."""
    return case.personality[0] > 0.5 and case.personality[1] < -0.5


def touches_held_out(case: IndependentCase) -> bool:
    """Same family rule as 2A's ``g6_touches_arena`` (selected location,
    candidates, or any recent-locations entry — review finding B1, 2026-07-11)
    applied to ``TEST_LOCATION`` (amendment 2026-07-12: arena trains, the
    test-only infirmary is the held-out family)."""
    return (case.selected_location == TEST_LOCATION
            or any(o.id == TEST_LOCATION
                   for o in case.candidates + case.recent_locations))


def dedupe_key(raw: dict) -> str:
    """Canonical input+label identity for exact-duplicate dropping."""
    return json.dumps({
        "p": {k: round(float(v), 4) for k, v in sorted(raw["personality"].items())},
        "dt": raw["decision_type"],
        "sel": raw.get("selected_location"),
        "rl": raw.get("recent_locations") or [],
        "ra": raw.get("recent_actions_same_location") or [],
        "c": raw["candidates"],
        "y": raw["choice"],
    }, sort_keys=True)


# ------------------------------------------------------- model input assembly --
def independent_case_to_inputs(case: IndependentCase, ablation: str = "full") -> dict:
    """``IndependentCase`` → inputs dict plus a one-hot ``"target"``.

    With a one-hot target, ``kl_loss`` computes exactly ``-log q[choice]``
    (cross-entropy) — 2B trains through the unchanged 2A loop.
    """
    d = case_input_dict(case, ablation)
    t = np.zeros(len(case.candidates), dtype=float)
    t[case.target_choice] = 1.0
    d["target"] = t
    return d
