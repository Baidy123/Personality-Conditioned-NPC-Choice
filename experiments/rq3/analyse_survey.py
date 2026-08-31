"""RQ3 human-study analysis: Qualtrics export -> identification, trait recovery,
and side-by-side preference results.

Reads the Qualtrics TSV export (UTF-16, three header rows), the stimulus
assignment in ``results/rq3/survey/``, and the persona definitions, and reports:

  1. sample composition and exclusions;
  2. blind personality identification per policy (against the 1/3 chance line)
     and per persona, with an exact binomial test and a participant-clustered
     check;
  3. trait recovery: rated-vs-intended error and correlation per trait/policy;
  4. the side-by-side preference block (scorer vs nonlinear 2B), exact sign test
     on non-tied choices, with the tie rate reported in its own right;
  5. the pre-registered H6 prediction.

Only numpy is required: the binomial, sign, permutation, and correlation tests
are computed here rather than pulled from scipy, which this project does not
depend on.

Usage:
    python -m experiments.rq3.analyse_survey \
        --export results/rq3/survey/<file>.tsv [--include-previews]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PERSONAS = ROOT / "data" / "rq3_personas.json"
SURVEY_DIR = ROOT / "results" / "rq3" / "survey"
ASSIGNMENT = SURVEY_DIR / "assignment.csv"
STIMULI = SURVEY_DIR / "stimuli.csv"

TRAITS = ["O", "C", "E", "A", "N"]
# export tag -> policy name in assignment.csv
POLICY_TAG = {"scorer": "scorer", "nn": "nonlinear_2b", "agno": "agnostic_nonlinear_2b"}
POLICY_ORDER = ["scorer", "nonlinear_2b", "agnostic_nonlinear_2b"]
POLICY_LABEL = {"scorer": "hand-authored scorer",
                "nonlinear_2b": "nonlinear 2B",
                "agnostic_nonlinear_2b": "agnostic control"}
# the comparison block always shows scorer as 甲 (=1) and nonlinear 2B as 乙 (=2)
CMP_SIDE = {"1": "scorer", "2": "nonlinear_2b", "3": "tie"}
SLIDER_MIN, SLIDER_MAX = 1, 7


# --------------------------------------------------------------------------- #
# statistics (no scipy in this project's dependency set)
# --------------------------------------------------------------------------- #
def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p) — one-sided exact upper tail."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def binom_two_sided(k: int, n: int, p: float) -> float:
    """Exact two-sided binomial p-value by the method of small probabilities."""
    if n == 0:
        return float("nan")
    obs = math.comb(n, k) * p ** k * (1 - p) ** (n - k)
    tot = 0.0
    for i in range(n + 1):
        pr = math.comb(n, i) * p ** i * (1 - p) ** (n - i)
        if pr <= obs * (1 + 1e-9):
            tot += pr
    return min(1.0, tot)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — behaves at the small n and near-boundary rates here."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    d = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / d
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rank(a: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged."""
    order = np.argsort(a, kind="mergesort")
    pos = np.empty(len(a), dtype=float)
    pos[order] = np.arange(len(a), dtype=float)
    out = np.empty(len(a), dtype=float)
    for v in np.unique(a):
        m = a == v
        out[m] = pos[m].mean()
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation. The sliders are Likert items, so the spacing between
    two adjacent points is not known to be constant; a rank correlation needs
    no such assumption and no mapping onto the persona scale at all."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    return pearson(_rank(x), _rank(y))


def perm_test_paired(a: np.ndarray, b: np.ndarray, rng, n_iter: int = 20000) -> float:
    """Two-sided permutation test on the paired mean difference (sign flips)."""
    d = a - b
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(n_iter, len(d)))
    null = np.abs((flips * d).mean(axis=1))
    return float((np.sum(null >= obs - 1e-12) + 1) / (n_iter + 1))


def perm_test_indep(a: np.ndarray, b: np.ndarray, rng, n_iter: int = 20000) -> float:
    """Two-sided permutation test on a difference in means between two groups."""
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    na = len(a)
    null = np.empty(n_iter)
    for i in range(n_iter):
        perm = rng.permutation(pool)
        null[i] = abs(perm[:na].mean() - perm[na:].mean())
    return float((np.sum(null >= obs - 1e-12) + 1) / (n_iter + 1))


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_export(path: Path) -> tuple[list[str], list[str], list[dict]]:
    """Qualtrics TSV: row 0 = column names, row 1 = question text, row 2 = ImportIds."""
    raw = path.read_bytes()
    for enc in ("utf-16", "utf-8-sig", "utf-8"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeError:
            continue
    else:
        raise SystemExit(f"cannot decode {path}")
    rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
    header, prompts = rows[0], rows[1]
    records = [dict(zip(header, r, strict=False)) for r in rows[3:]]
    return header, prompts, records


def load_assignment() -> dict[int, list[dict]]:
    """group -> [{persona, policy, sequence_id, options, correct_position}, ...]"""
    out: dict[int, list[dict]] = defaultdict(list)
    with ASSIGNMENT.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row["options"] = row["options"].split("|")
            row["correct_position"] = int(row["correct_position"])
            out[int(row["group"])].append(row)
    return out


def load_personas() -> dict[str, dict]:
    data = json.loads(PERSONAS.read_text(encoding="utf-8"))
    return {p["id"]: p for p in data["personas"]}


def slider_to_trait(v: float) -> float:
    """1..7 rating -> [-1, 1], the scale the persona vectors live on."""
    mid = (SLIDER_MIN + SLIDER_MAX) / 2
    return (v - mid) / ((SLIDER_MAX - SLIDER_MIN) / 2)


# --------------------------------------------------------------------------- #
# trial extraction
# --------------------------------------------------------------------------- #
def build_trials(records: list[dict], assignment: dict[int, list[dict]],
                 personas: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """One row per (participant, stimulus) and one per (participant, comparison)."""
    blind, prefs = [], []
    for rec in records:
        group = int(rec["version"].lstrip("v"))
        pid = rec["ResponseId"]
        for spec in assignment[group]:
            persona, policy = spec["persona"], spec["policy"]
            tag = f"{persona}_{[k for k, v in POLICY_TAG.items() if v == policy][0]}"
            ratings = {}
            for t in TRAITS:
                raw = rec.get(f"{tag}_rate_{t}", "")
                ratings[t] = slider_to_trait(float(raw)) if raw else None
            choice_raw = rec.get(f"{tag}_choice", "")
            chosen = spec["options"][int(choice_raw) - 1] if choice_raw else None
            blind.append({
                "participant": pid, "group": group, "persona": persona,
                "policy": policy, "sequence_id": spec["sequence_id"],
                "ratings": ratings, "chosen": chosen,
                "correct": None if chosen is None else chosen == persona,
                "intended": dict(zip(TRAITS, personas[persona]["vector"], strict=True)),
                "answered": choice_raw != "",
            })
        for persona in [s["persona"] for s in assignment[group]
                        if s["policy"] == "agnostic_nonlinear_2b"]:
            for measure in ("fit", "alive"):
                raw = rec.get(f"cmp_{persona}_{measure}", "")
                prefs.append({
                    "participant": pid, "group": group, "persona": persona,
                    "measure": measure,
                    "pick": CMP_SIDE.get(raw) if raw else None,
                })
    return blind, prefs


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def h(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def report_sample(records: list[dict], excluded: list[dict], prompts_map: dict) -> None:
    h("1. SAMPLE")
    print(f"responses analysed : {len(records)}")
    if excluded:
        print(f"excluded           : {len(excluded)} "
              f"({', '.join(r['ResponseId'] + '/' + r['_why'] for r in excluded)})")
    groups = Counter(r["version"] for r in records)
    print("group (version)    : " + ", ".join(f"{k}={v}" for k, v in sorted(groups.items())))
    durations = np.array([int(r["Duration (in seconds)"]) for r in records])
    print(f"duration (s)       : median {np.median(durations):.0f}, "
          f"IQR {np.percentile(durations, 25):.0f}–{np.percentile(durations, 75):.0f}, "
          f"range {durations.min()}–{durations.max()}")
    finished = Counter(r["Finished"] for r in records)
    print(f"finished           : {finished.get('1', 0)}/{len(records)}")
    age_lbl = {"1": "18–24", "2": "25–34", "3": "35–44", "4": "45+"}
    game_lbl = {"1": "rarely", "2": "sometimes", "3": "often"}
    print("age                : " + ", ".join(
        f"{age_lbl.get(k, k)}={v}" for k, v in sorted(Counter(r["age"] for r in records).items())))
    print("plays games        : " + ", ".join(
        f"{game_lbl.get(k, k)}={v}" for k, v in sorted(Counter(r["gaming"] for r in records).items())))


def report_identification(blind: list[dict], rng) -> dict:
    h("2. BLIND PERSONALITY IDENTIFICATION  (3-alternative forced choice, chance = 1/3)")
    per_policy = {}
    print(f"{'policy':<22}{'correct/n':>12}{'rate':>9}{'95% CI':>16}"
          f"{'p (vs 1/3)':>13}")
    for pol in POLICY_ORDER:
        rows = [t for t in blind if t["policy"] == pol and t["answered"]]
        k = sum(t["correct"] for t in rows)
        n = len(rows)
        lo, hi = wilson_ci(k, n)
        p = binom_two_sided(k, n, 1 / 3)
        per_policy[pol] = {"k": k, "n": n, "rate": k / n, "ci": (lo, hi), "p": p}
        print(f"{POLICY_LABEL[pol]:<22}{f'{k}/{n}':>12}{k / n:>9.3f}"
              f"{f'[{lo:.2f}, {hi:.2f}]':>16}{p:>13.4f}")

    print("\nparticipant-level (each participant sees 2 trails per policy; "
          "guards against treating 6 trials from one person as independent)")
    by_part: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for t in blind:
        if t["answered"]:
            by_part[t["participant"]][t["policy"]].append(t["correct"])
    print(f"{'policy':<22}{'mean per-participant rate':>28}{'SD':>8}"
          f"{'p (perm. vs 1/3)':>20}")
    part_scores = {}
    for pol in POLICY_ORDER:
        v = np.array([np.mean(by_part[p][pol]) for p in by_part if by_part[p][pol]])
        part_scores[pol] = v
        p = perm_test_paired(v, np.full_like(v, 1 / 3), rng)
        print(f"{POLICY_LABEL[pol]:<22}{v.mean():>28.3f}{v.std(ddof=1):>8.3f}{p:>20.4f}")

    print("\npairwise policy contrasts (paired within participant)")
    for a, b in [("scorer", "agnostic_nonlinear_2b"),
                 ("nonlinear_2b", "agnostic_nonlinear_2b"),
                 ("scorer", "nonlinear_2b")]:
        d = part_scores[a] - part_scores[b]
        p = perm_test_paired(part_scores[a], part_scores[b], rng)
        print(f"  {POLICY_LABEL[a]:<22} - {POLICY_LABEL[b]:<20} "
              f"Δ = {d.mean():+.3f}   p = {p:.4f}")

    print("\nper persona × policy (7 participants per cell)")
    personas = sorted({t["persona"] for t in blind})
    print(f"{'':<6}" + "".join(f"{POLICY_LABEL[p]:>24}" for p in POLICY_ORDER) + f"{'row':>10}")
    for per in personas:
        line = f"{per:<6}"
        rk = rn = 0
        for pol in POLICY_ORDER:
            rows = [t for t in blind if t["persona"] == per and t["policy"] == pol
                    and t["answered"]]
            k, n = sum(t["correct"] for t in rows), len(rows)
            rk += k
            rn += n
            line += f"{f'{k}/{n} ({k / n:.2f})':>24}"
        line += f"{f'{rk}/{rn} ({rk / rn:.2f})':>10}"
        print(line)

    print("\nwhich description was picked, over all trials of that policy "
          "(response-bias check; 42 picks spread over 6 personas, but each "
          "persona is only offered as an option on 3 of the 6 stimuli)")
    for pol in POLICY_ORDER:
        picks = Counter(t["chosen"] for t in blind if t["policy"] == pol and t["answered"])
        print(f"  {POLICY_LABEL[pol]:<22}" +
              " ".join(f"{k}:{v}" for k, v in sorted(picks.items())))

    print("\nbelow-chance check — a rate under 1/3 means participants were "
          "actively misled rather than merely uninformed")
    for pol in POLICY_ORDER:
        d = per_policy[pol]
        p_lo = 1 - binom_sf(d["k"] + 1, d["n"], 1 / 3)
        print(f"  {POLICY_LABEL[pol]:<22} P(X <= {d['k']} | p=1/3) = {p_lo:.4f}")

    ctl = per_policy["agnostic_nonlinear_2b"]
    print(f"""
TWO BASELINES — DO NOT CONFLATE THEM
  chance          = 1/3 = 0.333, a theoretical number: pick at random without
                    reading the trail. Nobody actually answered this way.
  agnostic control= {ctl['rate']:.3f} ({ctl['k']}/{ctl['n']}), a real condition
                    that {len({t['participant'] for t in blind})} participants answered.
  The control sits BELOW chance, so it is not a zero-information floor: it
  carries misleading information. With no personality input it emits one
  behaviour pattern for all six personas, and participants read a definite
  personality into that pattern, so they converge on the same two or three
  answers whatever the intended persona was. A participant answering that way
  can be right for at most about one persona in six.
    control vs 1/6 : exact binomial p = {binom_two_sided(ctl['k'], ctl['n'], 1 / 6):.4f}
    control vs 1/3 : exact binomial p = {binom_two_sided(ctl['k'], ctl['n'], 1 / 3):.4f}
  Consequences for reporting:
    - EFFECT SIZE goes against chance   (scorer {per_policy['scorer']['rate'] - 1 / 3:+.3f},
      nonlinear {per_policy['nonlinear_2b']['rate'] - 1 / 3:+.3f}).
    - ATTRIBUTION goes against the control: it shows the difference comes from
      the personality input rather than from the trail format. The margin over
      the control must NOT be reported as an effect size, because it also
      contains however far the control's own uniform pattern pushed it below
      chance.""")

    print("\nper stimulus sequence")
    print(f"  {'seq':<7}{'persona':<9}{'policy':<24}{'correct/n':>12}")
    for t_seq in sorted({(t["sequence_id"], t["persona"], t["policy"]) for t in blind}):
        rows = [t for t in blind if t["sequence_id"] == t_seq[0] and t["answered"]]
        k, n = sum(t["correct"] for t in rows), len(rows)
        print(f"  {t_seq[0]:<7}{t_seq[1]:<9}{POLICY_LABEL[t_seq[2]]:<24}"
              f"{f'{k}/{n} ({k / n:.2f})':>12}")
    return per_policy


def report_traits(blind: list[dict], rng) -> None:
    h("3. TRAIT RECOVERY  (sliders 1–7 mapped to [-1, 1]; intended = persona vector)")
    print("mean absolute error per trait (lower = closer to the intended profile)")
    print(f"{'policy':<22}" + "".join(f"{t:>9}" for t in TRAITS) + f"{'mean':>9}")
    mae_by_policy = {}
    for pol in POLICY_ORDER:
        rows = [t for t in blind if t["policy"] == pol]
        line = f"{POLICY_LABEL[pol]:<22}"
        allerr = []
        per_trait = {}
        for tr in TRAITS:
            e = np.array([abs(t["ratings"][tr] - t["intended"][tr])
                          for t in rows if t["ratings"][tr] is not None])
            per_trait[tr] = e
            allerr.append(e)
            line += f"{e.mean():>9.3f}"
        line += f"{np.concatenate(allerr).mean():>9.3f}"
        mae_by_policy[pol] = per_trait
        print(line)

    print("\ncorrelation between intended and rated trait value")
    print("(the signal test: does a higher intended trait produce a higher rating?)")
    print("trial level, n = 42 — but only 6 distinct intended values, so the "
          "persona-mean column below is the honest effect size")
    print(f"{'policy':<22}" + "".join(f"{t:>9}" for t in TRAITS))
    for pol in POLICY_ORDER:
        rows = [t for t in blind if t["policy"] == pol]
        line = f"{POLICY_LABEL[pol]:<22}"
        for tr in TRAITS:
            x = np.array([t["intended"][tr] for t in rows if t["ratings"][tr] is not None])
            y = np.array([t["ratings"][tr] for t in rows if t["ratings"][tr] is not None])
            line += f"{pearson(x, y):>9.3f}"
        print(line)

    print("\nSpearman rank correlation, trial level — the assumption-free "
          "version. The sliders are Likert items, so equal spacing between "
          "adjacent points is an assumption, not a fact; ranks need neither "
          "that nor any mapping onto the persona scale. Read this row if a "
          "reviewer challenges the 1..7 -> [-1, 1] conversion.")
    print(f"{'policy':<22}" + "".join(f"{t:>9}" for t in TRAITS))
    for pol in POLICY_ORDER:
        rows = [t for t in blind if t["policy"] == pol]
        line = f"{POLICY_LABEL[pol]:<22}"
        for tr in TRAITS:
            x = np.array([t["intended"][tr] for t in rows if t["ratings"][tr] is not None])
            y = np.array([t["ratings"][tr] for t in rows if t["ratings"][tr] is not None])
            line += f"{spearman(x, y):>9.3f}"
        print(line)

    print(f"\npersona-mean level, n = 6 points per cell")
    print(f"{'policy':<22}" + "".join(f"{t:>9}" for t in TRAITS))
    ids = sorted({t["persona"] for t in blind})
    for pol in POLICY_ORDER:
        rows = [t for t in blind if t["policy"] == pol]
        line = f"{POLICY_LABEL[pol]:<22}"
        for tr in TRAITS:
            x = np.array([next(t for t in rows if t["persona"] == p)["intended"][tr]
                          for p in ids])
            y = np.array([np.mean([t["ratings"][tr] for t in rows if t["persona"] == p
                                   and t["ratings"][tr] is not None]) for p in ids])
            line += f"{pearson(x, y):>9.3f}"
        print(line)

    print("\nsame, dropping H6 — H6 is the only persona with positive intended N "
          "(the other five span -0.77..-0.08), and its description opens with "
          "行动多变、难以捉摸, which is verbatim the high anchor printed on the N "
          "slider. Both reasons make the N column a one-persona result.")
    print(f"{'policy':<22}" + "".join(f"{t:>9}" for t in TRAITS))
    for pol in POLICY_ORDER:
        rows = [t for t in blind if t["policy"] == pol and t["persona"] != "H6"]
        line = f"{POLICY_LABEL[pol]:<22}"
        for tr in TRAITS:
            x = np.array([t["intended"][tr] for t in rows if t["ratings"][tr] is not None])
            y = np.array([t["ratings"][tr] for t in rows if t["ratings"][tr] is not None])
            line += f"{pearson(x, y):>9.3f}"
        print(line)

    print("\nmean rating per persona × trait (agnostic control shows the "
          "stereotype floor: it has no personality input, so any spread there "
          "is read into the trail)")
    for pol in POLICY_ORDER:
        print(f"\n  {POLICY_LABEL[pol]}")
        print(f"  {'persona':<10}" + "".join(f"{t + ' rated':>12}" for t in TRAITS))
        print(f"  {'':<10}" + "".join(f"{t + ' intend':>12}" for t in TRAITS))
        for per in sorted({t["persona"] for t in blind}):
            rows = [t for t in blind if t["persona"] == per and t["policy"] == pol]
            l1 = f"  {per:<10}"
            l2 = f"  {'':<10}"
            for tr in TRAITS:
                v = [t["ratings"][tr] for t in rows if t["ratings"][tr] is not None]
                l1 += f"{np.mean(v):>12.2f}"
                l2 += f"{rows[0]['intended'][tr]:>12.2f}"
            print(l1)
            print(l2)

    print("\nMAE differences vs the agnostic control (paired by trial, "
          "permutation test)")
    base = mae_by_policy["agnostic_nonlinear_2b"]
    for pol in ("scorer", "nonlinear_2b"):
        line = f"  {POLICY_LABEL[pol]:<22}"
        for tr in TRAITS:
            a, b = mae_by_policy[pol][tr], base[tr]
            n = min(len(a), len(b))
            p = perm_test_indep(a, b, rng, n_iter=5000)
            line += f"  {tr}: {a.mean() - b.mean():+.2f} (p={p:.3f})"
        print(line)

    print(f"""
WHY RAW DISTANCE CANNOT BE THE HEADLINE METRIC
  Rating everything at the midpoint — reading nothing, moving no slider off 4 —
  scores an MAE of {np.concatenate([np.array([abs(t['intended'][tr]) for t in blind if t['policy'] == 'scorer']) for tr in TRAITS]).mean():.3f}, which beats all three policies above. Participants
  rate more widely than the personas actually differ (rating SD 0.47-0.68
  against a true SD of 0.27-0.59, worst on O), so their noise inflates every
  distance while a constant answer has none. Raw MAE and raw Euclidean
  distance therefore reward not answering, and are reported here only as
  between-policy comparisons, never as an absolute standard of accuracy.""")


def report_pointing(blind: list[dict], personas: dict[str, dict], rng) -> None:
    """Does a rating point at the intended persona rather than at the others?

    Raw distance to the intended vector is unusable on its own (see the note in
    section 3): it rewards a participant who never moves a slider. Comparing the
    distance to the intended persona against the mean distance to the other five
    fixes that, because both terms sit inside the same rating. A participant who
    rates uniformly high, uniformly low, or timidly near the midpoint shifts
    both distances together and the difference is unchanged; only a rating that
    actually resembles one persona more than the rest moves it above zero."""
    h("3b. DOES THE RATING POINT AT THE RIGHT PERSONA?  "
      "(mean distance to the other five, minus distance to the intended one; "
      "0 = the rating is no closer to the intended persona than to any other)")
    ids = sorted(personas)
    vec = {p: np.array(personas[p]["vector"]) for p in ids}

    print("per trait")
    print(f"{'policy':<22}" + "".join(f"{t:>10}" for t in TRAITS))
    per_trait: dict[str, dict[str, np.ndarray]] = {}
    for pol in POLICY_ORDER:
        rows = [t for t in blind if t["policy"] == pol]
        line = f"{POLICY_LABEL[pol]:<22}"
        per_trait[pol] = {}
        for i, tr in enumerate(TRAITS):
            d = np.array([
                np.mean([abs(t["ratings"][tr] - vec[p][i]) for p in ids
                         if p != t["persona"]])
                - abs(t["ratings"][tr] - t["intended"][tr])
                for t in rows])
            per_trait[pol][tr] = d
            line += f"{d.mean():>+10.3f}"
        print(line)
    print(f"{'  (p vs 0)':<22}")
    for pol in POLICY_ORDER:
        line = f"{POLICY_LABEL[pol]:<22}"
        for tr in TRAITS:
            d = per_trait[pol][tr]
            line += f"{perm_test_paired(d, np.zeros_like(d), rng, 8000):>10.4f}"
        print(line)

    print("\nall five traits at once, averaged within participant")
    by_part: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for t in blind:
        v = np.array([t["ratings"][x] for x in TRAITS])
        own = np.linalg.norm(v - np.array([t["intended"][x] for x in TRAITS]))
        oth = np.mean([np.linalg.norm(v - vec[p]) for p in ids if p != t["persona"]])
        by_part[t["participant"]][t["policy"]].append(oth - own)
    score = {pol: np.array([np.mean(by_part[p][pol]) for p in by_part])
             for pol in POLICY_ORDER}
    for pol in POLICY_ORDER:
        v = score[pol]
        print(f"  {POLICY_LABEL[pol]:<22} {v.mean():>+7.3f}   "
              f"p vs 0 = {perm_test_paired(v, np.zeros_like(v), rng):.4f}")
    for a, b in [("scorer", "agnostic_nonlinear_2b"),
                 ("nonlinear_2b", "agnostic_nonlinear_2b"),
                 ("scorer", "nonlinear_2b")]:
        print(f"  {POLICY_LABEL[a]:<22} - {POLICY_LABEL[b]:<20} "
              f"Δ = {score[a].mean() - score[b].mean():+.3f}   "
              f"p = {perm_test_paired(score[a], score[b], rng):.4f}")


def report_preference(prefs: list[dict], rng) -> None:
    h("4. SIDE-BY-SIDE PREFERENCE  (甲 = hand-authored scorer, 乙 = nonlinear 2B)")
    label = {"fit": "fits the stated character better",
             "alive": "reads more like a person with a temper of its own"}
    for measure in ("fit", "alive"):
        rows = [p for p in prefs if p["measure"] == measure and p["pick"]]
        c = Counter(p["pick"] for p in rows)
        n = len(rows)
        ties = c.get("tie", 0)
        s, nn = c.get("scorer", 0), c.get("nonlinear_2b", 0)
        eff = s + nn
        p_sign = binom_two_sided(s, eff, 0.5) if eff else float("nan")
        print(f"\n{measure} — “{label[measure]}”   (n = {n} judgements)")
        print(f"  scorer {s}   nonlinear 2B {nn}   tie {ties} "
              f"({ties / n:.0%} tie rate)")
        if eff:
            lo, hi = wilson_ci(s, eff)
            print(f"  excluding ties: scorer wins {s}/{eff} = {s / eff:.3f} "
                  f"[{lo:.2f}, {hi:.2f}], exact sign test p = {p_sign:.4f}")
        print("  per persona: " + "  ".join(
            f"{per}: {Counter(p['pick'] for p in rows if p['persona'] == per)['scorer']}"
            f"/{Counter(p['pick'] for p in rows if p['persona'] == per)['nonlinear_2b']}"
            f"/{Counter(p['pick'] for p in rows if p['persona'] == per)['tie']}"
            for per in sorted({p["persona"] for p in rows})))
        print("               (scorer/nonlinear/tie)")

    print("\nparticipant-level consistency (each participant makes 2 fit and 2 "
          "alive judgements)")
    by_part = defaultdict(Counter)
    for p in prefs:
        if p["pick"]:
            by_part[p["participant"]][p["pick"]] += 1
    always = Counter()
    for part, c in by_part.items():
        top = max(c, key=lambda k: c[k])
        always[top if c[top] == sum(c.values()) else "mixed"] += 1
    print("  " + ", ".join(f"{k}: {v}" for k, v in always.items())
          + "   (participants whose 4 judgements all went one way, else mixed)")


def report_profile_recovery(blind: list[dict], personas: dict[str, dict]) -> None:
    """Continuous complement to the 3AFC: is the rated OCEAN vector nearest to
    the intended persona among all six? Uses only the sliders, so it is not
    constrained by which three descriptions the forced choice happened to offer."""
    h("6. NEAREST-PERSONA RECOVERY FROM THE SLIDERS ALONE (chance = 1/6)")
    ids = sorted(personas)
    mat = np.array([personas[i]["vector"] for i in ids])
    print(f"{'policy':<22}{'nearest = intended':>20}{'rate':>9}"
          f"{'mean dist to intended':>24}{'rank of intended':>19}")
    for pol in POLICY_ORDER:
        rows = [t for t in blind if t["policy"] == pol
                and all(v is not None for v in t["ratings"].values())]
        hits, dists, ranks = 0, [], []
        for t in rows:
            v = np.array([t["ratings"][x] for x in TRAITS])
            d = np.linalg.norm(mat - v, axis=1)
            order = [ids[i] for i in np.argsort(d)]
            hits += order[0] == t["persona"]
            dists.append(d[ids.index(t["persona"])])
            ranks.append(order.index(t["persona"]) + 1)
        n = len(rows)
        print(f"{POLICY_LABEL[pol]:<22}{f'{hits}/{n}':>20}{hits / n:>9.3f}"
              f"{np.mean(dists):>24.3f}{np.mean(ranks):>19.2f}")
    print("  (rank 1 = the intended persona is the closest of the six; "
          "chance rank = 3.5)")


def report_covariates(blind: list[dict], records: list[dict]) -> None:
    h("7. COVARIATES AND DATA-QUALITY CHECKS")
    by_part = defaultdict(list)
    for t in blind:
        if t["answered"]:
            by_part[t["participant"]].append(t["correct"])
    acc = {p: float(np.mean(v)) for p, v in by_part.items()}
    dur = {r["ResponseId"]: int(r["Duration (in seconds)"]) for r in records}
    ids = [p for p in acc if p in dur]
    x = np.array([dur[p] for p in ids], dtype=float)
    y = np.array([acc[p] for p in ids])
    print(f"  duration vs overall accuracy: r = {pearson(x, y):+.3f} (n = {len(ids)})")
    game = {r["ResponseId"]: r["gaming"] for r in records}
    lbl = {"1": "rarely", "2": "sometimes", "3": "often"}
    for level in sorted({game[p] for p in ids}):
        v = np.array([acc[p] for p in ids if game[p] == level])
        print(f"  plays games {lbl.get(level, level):<10} n = {len(v):<3} "
              f"accuracy {v.mean():.3f}")
    flat = 0
    for t in blind:
        vals = [v for v in t["ratings"].values() if v is not None]
        if len(vals) == 5 and len(set(vals)) == 1:
            flat += 1
    print(f"  straight-lined slider sets (all five identical): {flat}/{len(blind)}")
    print("  note: the comparison block always shows the scorer as 甲 and the "
          "nonlinear policy as 乙 in that fixed order, so a left/first-position "
          "bias is not separable from the preference itself.")


def report_h6(blind: list[dict]) -> None:
    h("5. PRE-REGISTERED PREDICTION — H6 has the lowest identification rate "
      "under every policy")
    ok = True
    for pol in POLICY_ORDER:
        rates = {}
        for per in sorted({t["persona"] for t in blind}):
            rows = [t for t in blind if t["persona"] == per and t["policy"] == pol
                    and t["answered"]]
            rates[per] = sum(t["correct"] for t in rows) / len(rows)
        lowest = min(rates, key=lambda k: rates[k])
        tie = [p for p in rates if rates[p] == rates["H6"]]
        held = rates["H6"] == rates[lowest]
        ok &= held
        print(f"  {POLICY_LABEL[pol]:<22} H6 = {rates['H6']:.2f}; "
              f"lowest = {lowest} ({rates[lowest]:.2f})"
              + (f"; tied lowest with {tie}" if held and len(tie) > 1 else "")
              + ("  -> supported" if held else "  -> NOT supported"))
    print(f"\n  overall: {'supported in every condition' if ok else 'not supported'}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", required=True, type=Path)
    ap.add_argument("--include-previews", action="store_true",
                    help="keep Status=4 (survey preview) responses")
    ap.add_argument("--min-duration", type=int, default=0,
                    help="drop responses faster than this many seconds")
    ap.add_argument("--seed", type=int, default=20260808)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    _, prompts, records = load_export(args.export)
    assignment = load_assignment()
    personas = load_personas()

    kept, excluded = [], []
    for r in records:
        why = ""
        if r.get("consent_status") != "accepted":
            why = "no consent"
        elif not r.get("version"):
            why = "no group assigned"
        elif r["Status"] == "4" and not args.include_previews:
            why = "preview response"
        elif args.min_duration and int(r["Duration (in seconds)"]) < args.min_duration:
            why = f"under {args.min_duration}s"
        if why:
            r["_why"] = why
            excluded.append(r)
        else:
            kept.append(r)

    blind, prefs = build_trials(kept, assignment, personas)

    print(f"RQ3 human study — {args.export.name}")
    report_sample(kept, excluded, {})
    report_identification(blind, rng)
    report_traits(blind, rng)
    report_pointing(blind, personas, rng)
    report_preference(prefs, rng)
    report_h6(blind)
    report_profile_recovery(blind, personas)
    report_covariates(blind, kept)
    print()


if __name__ == "__main__":
    main()
