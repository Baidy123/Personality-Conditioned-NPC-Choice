"""Annotator test-retest consistency — the practical ceiling on 2B agreement.

The same scenario can draw different but equally plausible choices from the
annotator on different occasions. That irreducible variability caps what *any*
system can score against these labels, so a raw 68% means little until the
ceiling is known.

Two steps:

  export  →  re-annotate the SAME cases in N fresh chat sessions  →  score

    python -m experiments.rq2.run_label_consistency export
    #  paste results/rq2b/consistency/prompts.md into N *new* chats (no shared
    #  context, or the model copies its own previous answers) and save each
    #  reply as results/rq2b/consistency/replies_1.json, _2.json, ...
    python -m experiments.rq2.run_label_consistency score

The scored quantity is majority-vote agreement: for each case, the share of
repeat annotations that match the modal answer, averaged over cases. It is the
same quantity a policy's top-1 accuracy is measured against, so the two are
directly comparable. Cases are drawn from the *test* groups only, so the
ceiling describes the set the models are actually scored on.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from npc_policy import IndependentCase

from .common import read_pool
from .independent import IND_DATA, IND_RESULTS, TEST_GROUPS

OUT = IND_RESULTS / "consistency"
N_CASES = 30
SAMPLE_SEED = 20260712

LOC_BLURB = {
    "tavern": "a lively drinking house: noisy, social, little privacy, the odd brawl",
    "library": "quiet halls of books: solitary study, order, deep thought",
    "chapel": "a calm sanctuary: ritual, reflection, quiet confession",
    "market": "a bustling trade square: haggling, browsing, deal-making",
    "forest": "wild ground beyond the walls: roaming, foraging, solitude",
    "arena": "a fighting ground: combat bouts, training, spectacle, betting",
    "infirmary": "the village healing house: sickbeds, herbs, quiet duty",
}
TRAITS = ("O", "C", "E", "A", "N")


def sample_cases(data_dir: Path) -> list[tuple[str, IndependentCase]]:
    """A fixed random sample of test cases (seeded — the same 30 every export)."""
    pool = [(t["id"], c) for c, t in
            read_pool(data_dir / "cases.jsonl", case_cls=IndependentCase)
            if t["group"] in TEST_GROUPS]
    rng = np.random.default_rng(SAMPLE_SEED)
    idx = rng.choice(len(pool), size=min(N_CASES, len(pool)), replace=False)
    return [pool[int(i)] for i in sorted(idx)]


def render(cid: str, case: IndependentCase) -> str:
    """One case as the annotator sees it — same information as the guide's format."""
    p = ", ".join(f"{t} {v:+.1f}" for t, v in zip(TRAITS, case.personality))
    lines = [f"### Case {cid}", f"- Personality: {p}"]
    if case.decision_type == "location":
        hist = [o.id for o in case.recent_locations]
        lines.append(f"- Recently visited (oldest first): "
                     f"{', '.join(hist) if hist else '(nothing recorded)'}")
        lines.append("- It may now go to:")
        for i, o in enumerate(case.candidates):
            lines.append(f"    {chr(65 + i)}. {o.id} — {LOC_BLURB.get(o.id, '')}")
        lines.append("- Where does it go? Answer with the letter only.")
    else:
        acts = [o.id for o in case.recent_actions_same_location]
        lines.append(f"- It is at the {case.selected_location} "
                     f"({LOC_BLURB.get(case.selected_location, '')})")
        lines.append(f"- Just did there (oldest first): "
                     f"{', '.join(acts) if acts else '(only just arrived)'}")
        lines.append("- It may now:")
        for i, o in enumerate(case.candidates):
            lines.append(f"    {chr(65 + i)}. {o.id}")
        lines.append("- What does it do? Answer with the letter only.")
    return "\n".join(lines)


def do_export(data_dir: Path) -> None:
    cases = sample_cases(data_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    body = "\n\n".join(render(cid, c) for cid, c in cases)
    prompt = f"""# Re-annotation batch (annotator consistency check)

You are judging what a fantasy-village NPC would do. Stay in character; weigh
comfort, boredom, and curiosity against the personality. There is no "correct"
answer — give the choice this specific NPC would most plausibly make.

Personality traits run from -1 to +1: O openness (curiosity, novelty seeking),
C conscientiousness (order, routine), E extraversion (sociability, stimulation),
A agreeableness (warmth, cooperation), N neuroticism (anxiety, volatility).

Answer every case. Reply with a JSON object only — no prose:

    {{"<case id>": "<letter>", ...}}

{body}
"""
    (OUT / "prompts.md").write_text(prompt, encoding="utf-8")
    (OUT / "cases.json").write_text(
        json.dumps({cid: [o.id for o in c.candidates] for cid, c in cases}, indent=2),
        encoding="utf-8")
    print(f"exported {len(cases)} cases → {OUT / 'prompts.md'}\n"
          f"paste it into N *fresh* chat sessions (no shared context) and save each\n"
          f"reply as {OUT / 'replies_1.json'}, replies_2.json, ...  then run:\n"
          f"  python -m experiments.rq2.run_label_consistency score")


def do_score() -> None:
    cand = json.loads((OUT / "cases.json").read_text(encoding="utf-8"))
    files = sorted(OUT.glob("replies_*.json"))
    if len(files) < 2:
        raise SystemExit(f"need >= 2 reply files in {OUT}, found {len(files)}")
    reps = []
    for f in files:
        raw = json.loads(f.read_text(encoding="utf-8"))
        reps.append({str(k): str(v).strip().upper()[:1] for k, v in raw.items()})

    rows, unanimous = [], 0
    for cid, options in cand.items():
        votes = [r[cid] for r in reps if cid in r]
        if len(votes) < 2:
            continue
        counts = Counter(votes)
        modal, n_modal = counts.most_common(1)[0]
        agree = n_modal / len(votes)          # share matching the modal answer
        unanimous += int(len(counts) == 1)
        rows.append({"case_id": cid, "n_votes": len(votes),
                     "modal": modal, "agreement": agree,
                     "answers": "".join(votes),
                     "n_candidates": len(options)})

    agr = float(np.mean([r["agreement"] for r in rows]))
    chance = float(np.mean([1.0 / r["n_candidates"] for r in rows]))
    from experiments.rq1.common import write_csv
    write_csv(OUT / "consistency.csv", list(rows[0].keys()),
              [list(r.values()) for r in rows])

    print(f"{len(rows)} cases x {len(files)} independent annotation rounds\n")
    print(f"  majority-vote agreement (the ceiling) : {agr:.1%}")
    print(f"  fully unanimous cases                 : {unanimous}/{len(rows)} "
          f"({unanimous / len(rows):.0%})")
    print(f"  chance level for these cases          : {chance:.1%}")
    print(f"\nRead as: no system can be expected to exceed ~{agr:.0%} agreement with "
          f"these labels;\ncompare against the best learned policy in main_table.csv.")
    print(f"written: {OUT / 'consistency.csv'}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Annotator test-retest consistency")
    ap.add_argument("step", choices=("export", "score"))
    ap.add_argument("--data", type=Path, default=IND_DATA)
    args = ap.parse_args(argv)
    do_export(args.data) if args.step == "export" else do_score()


if __name__ == "__main__":
    main()
