"""Shared utilities for the RQ2 Study 2A experiments.

Design: ``docs/specs/2026-07-10-rq2-2a-pipeline-design.md``. Pool records pair a
``ControlledCase`` with generation tags (``"gen"``: id / source / world) used only
by the split filters — tags never reach a model. All randomness in this package
derives from ``GEN_SEED`` (generation) or the run seed (training).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from npc_policy import DEFAULT_CONFIG, ControlledCase, ScorerConfig
from npc_policy.features import case_inputs
from npc_policy.representation import Personality

CODE = Path(__file__).resolve().parents[2]
DATA = CODE / "data"
RQ1_WORLDS = DATA / "rq1_cases" / "worlds"

GEN_SEED = 20260709            # dataset-generation base seed (RQ1 convention)
TRAIN_SIZE = 100_000           # per split, matched across splits [PROVISIONAL]
VAL_SIZE = 5_000
TEST_SIZE = 5_000

MODELS = ("simple", "nonlinear")
S0_MODELS = ("simple", "nonlinear", "agnostic_simple", "agnostic_nonlinear")
SEEDS = tuple(range(5))
G_SPLITS = ("G1", "G2", "G3", "G4", "G5", "G6")
ALL_SPLITS = ("S0",) + G_SPLITS
ABLATIONS = ("no_context", "location_only")   # "full" is the S0 main configuration
DATA_SIZES = (1_000, 5_000, 20_000)     # 100k point reuses the S0 main runs


def dirs(smoke: bool) -> tuple[Path, Path]:
    """(dataset dir, results dir); smoke mode is fully isolated from the full run."""
    if smoke:
        return DATA / "rq2_controlled_smoke", CODE / "results" / "rq2_smoke"
    return DATA / "rq2_controlled", CODE / "results" / "rq2"


def config_hash(config: ScorerConfig = DEFAULT_CONFIG) -> str:
    """SHA-256 of the serialised scorer config — pins the teacher's config values
    (not the scorer implementation)."""
    payload = json.dumps(dataclasses.asdict(config), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ pool I/O --
def write_pool(path: Path, records: list[tuple[ControlledCase, dict]]) -> None:
    """Write a JSONL pool atomically; buffer lists are stored oldest→newest.

    The file is written to a ``.tmp`` sibling and renamed into place, so an
    interrupted process can never leave a silently truncated pool file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for case, tags in records:
            d = case.to_dict()
            d["gen"] = tags
            f.write(json.dumps(d) + "\n")
    os.replace(tmp, path)


def read_pool(path: Path) -> list[tuple[ControlledCase, dict]]:
    """Read a JSONL pool; buffer lists are stored oldest→newest.

    A record without ``"gen"`` tags is corruption and raises ``KeyError``.
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            tags = d.pop("gen")
            records.append((ControlledCase.from_dict(d), tags))
    return records


# ------------------------------------------------------- model input assembly --
def case_to_inputs(case: ControlledCase, ablation: str = "full") -> dict:
    """``ControlledCase`` → ``features.case_inputs`` dict plus ``"target"``.

    Action cases take the selected location's features from the newest
    ``recent_locations`` entry — the controller pushes the chosen location into
    ``H_L`` before the action choice, and generation guarantees the invariant.
    ``ablation`` zeroes relation *inputs* ("no_context": both levels;
    "location_only": action cases only); targets are never touched
    (retrained-ablation design).
    """
    if ablation not in ("full", "no_context", "location_only"):
        raise ValueError(f"unknown ablation {ablation!r}")
    if case.decision_type == "location" and case.selected_location is not None:
        raise ValueError(
            "location case carries a stale selected_location "
            f"({case.selected_location!r}); location cases must have None"
        )
    selected = None
    if case.decision_type == "action":
        if (not case.recent_locations
                or case.recent_locations[-1].id != case.selected_location):
            raise ValueError(
                "action case must carry its selected location as the newest "
                "recent_locations entry"
            )
        selected = case.recent_locations[-1]
    relations = case.candidate_history_features
    if ablation == "no_context" or (ablation == "location_only" and case.decision_type == "action"):
        relations = None
    d = case_inputs(
        Personality(np.array(case.personality, dtype=float)),   # copy: no aliasing
        case.candidates, case.decision_type,
        relations=relations, selected_location=selected,
    )
    d["target"] = np.array(case.target_distribution, dtype=float)   # copy: no aliasing
    return d


# ------------------------------------------------------------------- metrics --
def kl_np(t: np.ndarray, q: np.ndarray) -> float:
    """``KL(t ‖ q)`` in nats; exact at ``t_i = 0`` (``q > 0`` from the softmax)."""
    t, q = np.asarray(t, dtype=float), np.asarray(q, dtype=float)
    pos = t > 0
    # float64 softmax can underflow to exact 0 for logit gaps > ~745; an inf
    # here would poison mean aggregation, so floor q at the smallest positive.
    q = np.maximum(q, np.finfo(float).tiny)
    return float((t[pos] * (np.log(t[pos]) - np.log(q[pos]))).sum())


def jsd_np(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (np.asarray(p, dtype=float) + np.asarray(q, dtype=float))
    return 0.5 * kl_np(p, m) + 0.5 * kl_np(q, m)


def top1_agree(t: np.ndarray, q: np.ndarray) -> bool:
    """Argmax tie-breaking is first-index, so ties resolve by candidate order."""
    return int(np.argmax(t)) == int(np.argmax(q))


# ----------------------------------------------------------------- run matrix --
@dataclass(frozen=True)
class RunSpec:
    """One training run: split × model × seed (+ ablation / data-size variants)."""

    split: str
    model: str
    seed: int
    ablation: str = "full"
    n_train: int | None = None      # None → the split's full train manifest

    @property
    def run_id(self) -> str:
        parts = [self.split, self.model]
        if self.ablation != "full":
            parts.append(f"abl_{self.ablation}")
        if self.n_train is not None:
            parts.append(f"n{self.n_train}")
        parts.append(f"s{self.seed}")
        return "__".join(parts)


def run_matrix(smoke: bool = False) -> list[RunSpec]:
    """The full 130-run matrix (design §3), or a 4-run smoke subset."""
    if smoke:
        return [
            RunSpec("S0", "simple", 0),
            RunSpec("S0", "nonlinear", 0),
            RunSpec("S0", "agnostic_simple", 0),
            RunSpec("G1", "simple", 0),
        ]
    runs = [RunSpec("S0", m, s) for m in S0_MODELS for s in SEEDS]                        # 20
    runs += [RunSpec(g, m, s) for g in G_SPLITS for m in MODELS for s in SEEDS]           # 60
    runs += [RunSpec("S0", m, s, ablation=a) for a in ABLATIONS for m in MODELS for s in SEEDS]  # 20
    runs += [RunSpec("S0", m, s, n_train=n) for n in DATA_SIZES for m in MODELS for s in SEEDS]  # 30
    return runs


def build_model(name: str, seed: int):
    """Fresh model, CPU/float64 (the model layer's native precision).

    Seeds the *global* torch RNG (``torch.manual_seed``) before construction.
    """
    import torch

    from npc_policy.learned import AgnosticPolicy, NonlinearPolicy, SimplePolicy

    torch.manual_seed(seed)
    if name == "simple":
        return SimplePolicy()
    if name == "nonlinear":
        return NonlinearPolicy()
    if name == "agnostic_simple":
        return AgnosticPolicy(SimplePolicy())
    if name == "agnostic_nonlinear":
        return AgnosticPolicy(NonlinearPolicy())
    raise ValueError(f"unknown model {name!r}")
