"""E1–E4 structural diagnostic on the trained S0 students (design §5).

A ``StudentTraceAdapter`` duck-types the scorer interface the RQ1 pipeline uses
(``trace(personality, candidates, buffer=…, level=…)``), so trained students run
on the frozen RQ1 matched cases without modifying ``experiments/rq1``. Action
choices need the selected location's context, which that interface does not
carry — the adapter holds ``current_location``, set by the local trajectory
runner after each location decision.

Pre-registered expectation (research spec §8): the simple model fails to track
the N-temperature channel (E1 N-sweep entropy curve); the nonlinear model
follows it.

Run from ``code/``:  python -m experiments.rq2.run_e_diag [--smoke]
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.rq1.common import (
    LOC_COLORS,
    TRAIT_COLORS,
    TRAIT_SHORT,
    TRAITS,
    TRAJ_TEMPERATURE,
    entropy,
    load_cases,
    mantel,
    pairwise_jsd,
    personality_of,
    run_trajectory,
    setup_style,
    spearman,
    trajectory_metrics,
    world_for,
    write_csv,
)
from npc_policy import (
    DEFAULT_CONFIG,
    DecisionController,
    HandAuthoredScorer,
    load_personalities,
)
from npc_policy.learned import predict_distribution
from npc_policy.relations import Relations, compute_relations

from .common import DATA, dirs
from .run_2a import load_student

FAMILIES = ("simple", "nonlinear")
STUDENT_STYLE = {"simple": ("#0072B2", "--"), "nonlinear": ("#E69F00", ":")}


class StudentTraceAdapter:
    """Makes a learned policy a drop-in for ``HandAuthoredScorer`` in the
    controller and the matched-case analyses. Only ``P_rule`` (and the relations
    actually used) is meaningful on the returned trace object."""

    def __init__(self, model, config=DEFAULT_CONFIG):
        self.model = model
        self.config = config
        self.current_location = None      # set by the trajectory runner

    def trace(self, personality, candidates, buffer=None, relations=None,
              level="location"):
        if relations is None and buffer is not None and not buffer.is_empty():
            relations = compute_relations(candidates, buffer, self.config)
        selected = self.current_location if level == "action" else None
        P = predict_distribution(self.model, personality, candidates, level,
                                 relations=relations, selected_location=selected)
        if relations is None:
            m = len(candidates)
            relations = Relations(np.zeros(m), np.zeros(m), np.zeros(m))
        return SimpleNamespace(P_rule=P, relations=relations, base=None,
                               P_base=None, mu=None, gamma=None, q=None, T_N=None)

    def distribution(self, personality, candidates, buffer=None, relations=None,
                     level="location"):
        return self.trace(personality, candidates, buffer, relations, level).P_rule


def student_trajectory(adapter: StudentTraceAdapter, world, personality, seed: int,
                       rounds: int = 50, memory: str = "full"):
    """RQ1 ``run_trajectory`` semantics with the location-context hook added."""
    if memory not in ("full", "location_only", "none"):
        raise ValueError(f"unknown memory condition {memory!r}")
    ctrl = DecisionController(
        adapter, config=adapter.config, mode="sample",
        rng=np.random.default_rng(seed), selection_temperature=TRAJ_TEMPERATURE,
    )
    visits, acts = [], []
    for _ in range(rounds):
        d = ctrl.choose_location(personality, world.resolve())
        adapter.current_location = world.effective_location(d.option.id)
        a = ctrl.choose_action(personality, world.actions_at(d.option.id))
        visits.append(d.option.id)
        acts.append(a.option.id)
        if memory == "none":
            ctrl.H_L.clear()
            ctrl.H_A.clear()
        elif memory == "location_only":
            ctrl.H_A.clear()
    return visits, acts


def discover_students(results_dir) -> dict[str, list]:
    """S0 main-run students per family (whatever seeds exist — works for smoke)."""
    out: dict[str, list] = {}
    for fam in FAMILIES:
        models = []
        for f in sorted((results_dir / "runs").glob(f"S0__{fam}__s?.json")):
            models.append(load_student(results_dir, f.stem))
        if models:
            out[fam] = models
    return out


def _seed_mean(models, personality, candidates, level) -> np.ndarray:
    return np.mean([predict_distribution(m, personality, candidates, level)
                    for m in models], axis=0)


def _metric_means(runs: list[dict]) -> list[str]:
    keys = ("visit_entropy", "max_share", "distinct", "repeat_rate",
            "action_repeat_rate")
    return [f"{float(np.mean([r.get(k, 0.0) for r in runs])):.4f}" for k in keys]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="E1-E4 diagnostic on trained students")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    setup_style()
    _, results_dir = dirs(args.smoke)
    out = results_dir / "e_diag"
    out.mkdir(parents=True, exist_ok=True)
    students = discover_students(results_dir)
    if not students:
        raise SystemExit(f"no S0 student runs under {results_dir / 'runs'} - train first")
    scorer = HandAuthoredScorer()
    cases = load_cases()
    sweep = cases["profiles"]["sweep"]
    values = sorted({e["value"] for e in sweep})
    world = world_for("full")
    locs = world.resolve()
    n_profiles = 40 if args.smoke else len(cases["profiles"]["random"])
    traj_seeds = range(1) if args.smoke else range(2)
    conditions = ("full",) if args.smoke else ("full", "location_only", "none")

    def sweep_p(trait, v):
        return personality_of(next(e for e in sweep
                                   if e["trait"] == trait and e["value"] == v))

    # ---- E1 overlay: teacher vs student curves + N entropy ---------------------
    ent_rows = []
    for fam, models in students.items():
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
        color, dash = STUDENT_STYLE[fam]
        for ax, trait in zip(axes.flat, TRAITS):
            P_t = np.stack([scorer.distribution(sweep_p(trait, v), locs,
                                                level="location") for v in values])
            P_s = np.stack([_seed_mean(models, sweep_p(trait, v), locs, "location")
                            for v in values])
            for k, o in enumerate(locs):
                ax.plot(values, P_t[:, k], color=LOC_COLORS[o.id], label=o.id)
                ax.plot(values, P_s[:, k], color=LOC_COLORS[o.id], linestyle="--",
                        alpha=0.75)
            ax.set_title(f"{TRAIT_SHORT[trait]} (teacher solid / student dashed)")
            ax.set_ylabel("P")
        ax = axes.flat[5]
        ent_t = [entropy(scorer.distribution(sweep_p("neuroticism", v), locs,
                                             level="location")) for v in values]
        ent_s = [entropy(_seed_mean(models, sweep_p("neuroticism", v), locs,
                                    "location")) for v in values]
        ax.plot(values, ent_t, color=TRAIT_COLORS["neuroticism"], label="teacher")
        ax.plot(values, ent_s, color=color, linestyle=dash, label=fam)
        ax.set_title("N temperature: entropy of P (pre-registered check)")
        ax.set_xlabel("N value")
        ax.set_ylabel("entropy (nats)")
        ax.legend(frameon=False)
        fig.suptitle(f"E1 overlay — teacher vs {fam} (S0 students, seed mean)", y=1.0)
        fig.tight_layout()
        fig.savefig(out / f"e1_overlay_{fam}.png", bbox_inches="tight")
        plt.close(fig)
        for v, a, b in zip(values, ent_t, ent_s):
            ent_rows.append([fam, v, f"{a:.6f}", f"{b:.6f}"])
    write_csv(out / "e1_n_entropy.csv",
              ["family", "N", "teacher_entropy", "student_entropy"], ent_rows)

    # ---- E2: profile-distinguishability correlation ----------------------------
    profiles = [personality_of(e) for e in cases["profiles"]["random"][:n_profiles]]
    P_t = np.stack([scorer.distribution(p, locs, level="location") for p in profiles])
    D_t = pairwise_jsd(P_t)
    iu = np.triu_indices(len(profiles), k=1)
    e2_rows = []
    for fam, models in students.items():
        P_s = np.stack([_seed_mean(models, p, locs, "location") for p in profiles])
        D_s = pairwise_jsd(P_s)
        rho, pval = mantel(D_t, D_s, n_perm=199 if args.smoke else 999)
        e2_rows.append([fam, f"{spearman(D_t[iu], D_s[iu]):.4f}",
                        f"{rho:.4f}", f"{pval:.4f}"])
        print(f"E2 {fam}: mantel rho {rho:.3f} (p {pval:.3f})")
    write_csv(out / "e2_correlation.csv",
              ["family", "spearman_upper", "mantel_rho", "mantel_p"], e2_rows)

    # ---- E3/E4: trajectory statistics ------------------------------------------
    named = load_personalities(DATA / "personalities.json")
    rows = []
    for prof_name, p in named.items():
        for cond in conditions:
            runs = [trajectory_metrics(*run_trajectory(scorer, world, p, seed=s,
                                                       memory=cond))
                    for s in range(10)]
            rows.append(["teacher", prof_name, cond] + _metric_means(runs))
            for fam, models in students.items():
                runs = []
                for mi, m in enumerate(models):
                    adapter = StudentTraceAdapter(m)
                    for s in traj_seeds:
                        runs.append(trajectory_metrics(*student_trajectory(
                            adapter, world, p, seed=1000 * mi + s, memory=cond)))
                rows.append([fam, prof_name, cond] + _metric_means(runs))
    header = ["policy", "profile", "memory", "visit_entropy", "max_share",
              "distinct", "repeat_rate", "action_repeat_rate"]
    write_csv(out / "e3_e4_traj_stats.csv", header, rows)

    # small comparison figure: visit entropy per profile, full memory
    fig, ax = plt.subplots(figsize=(10, 4.5))
    names = list(named)
    x = np.arange(len(names))
    series = ["teacher"] + list(students)
    for i, pol in enumerate(series):
        ys = [float(r[3]) for r in rows if r[0] == pol and r[2] == "full"
              and r[1] in names]
        ax.bar(x + (i - (len(series) - 1) / 2) * 0.26, ys, width=0.24,
               label=pol,
               color={"teacher": "#444444", "simple": "#0072B2",
                      "nonlinear": "#E69F00"}[pol])
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_ylabel("visit entropy (nats)")
    ax.set_title("E3 — trajectory diversity, teacher vs students (full memory)")
    ax.legend(frameon=False)
    fig.savefig(out / "e3_visit_entropy.png", bbox_inches="tight")
    plt.close(fig)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
