"""Turn the generated stimuli into ready-to-paste Study-3 survey material.

Consumes ``data/rq3_sequences/study/`` (18 sequences + manifest) and produces
everything the Qualtrics build needs: the three Latin-square group scripts with
trails rendered in Chinese, the three-alternative options with their correct
answers, the preference-comparison pairs, and a machine-readable assignment
table for analysis.

Design constraints this script enforces, rather than leaving to the survey
builder:

- **One trail per persona per participant.** Two trails of one persona resemble
  each other whatever policy produced them; judging both would compress the
  difference the study measures. The Latin square rotates policy across groups
  so each persona is still seen under all three.
- **Distractors are fixed per persona, not per trial.** If a persona's two
  distractors were redrawn for every participant, its identification difficulty
  would vary with the draw, and difficulty would be confounded with policy.
  Drawn once with a fixed seed, the same three options accompany a persona in
  every group and under every policy.
- **Option order is emitted fixed and randomised by Qualtrics**, so the correct
  answer does not sit in a constant position.
- **Preference pairs contrast scorer against the learned policy only.** The
  agnostic control exists to establish the chance floor for blind
  identification; asking which of two trails is "better" when one comes from a
  policy that cannot read personality answers nothing.
- **Both compared trails are new to the participant.** A group's comparison
  personas are the ones it met under the agnostic control during blind
  identification, so neither the scorer nor the learned trail has been seen
  before. Comparing a trail the participant already judged against one they
  have not would put familiarity on one side of the choice.

Outputs (results/rq3/survey/):
  group1.md, group2.md, group3.md   paste-ready blocks, one per Latin-square group
  assignment.csv                    persona x group -> policy, sequence id, correct answer
  stimuli.csv                       all 18 trails as rendered text

Run from ``code/``:  python -m experiments.rq3.build_survey
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[2]
DATA = CODE / "data"
SEQ_DIR = DATA / "rq3_sequences" / "study"
OUT = CODE / "results" / "rq3" / "survey"

PERSONAS = DATA / "rq3_personas.json"
LABELS = DATA / "rq3_labels_zh.json"
LABELS_EN = DATA / "rq3_labels_en.json"   # English mirror for appendix/ethics record

# Policy order defines the Latin square: group g gives persona i the policy at
# index (i + g) % 3. Renaming these changes nothing but the manifest lookup.
POLICIES = ("scorer", "nonlinear_2b", "agnostic_nonlinear_2b")
POLICY_ZH = {"scorer": "手写公式", "nonlinear_2b": "神经网络",
             "agnostic_nonlinear_2b": "无性格对照"}
N_GROUPS = 3
N_DISTRACTORS = 2
DISTRACTOR_SEED = 20260804
PREFERENCE_POLICIES = ("scorer", "nonlinear_2b")


def preference_personas(g: int, n_personas: int) -> list[int]:
    """Persona indices this group compares: those it blind-judged under the
    agnostic control, so both compared trails are unseen. The Latin square
    gives exactly two per group and covers all six across the three."""
    agnostic = POLICIES.index("agnostic_nonlinear_2b")
    return [i for i in range(n_personas) if (i + g) % N_GROUPS == agnostic]


def render(seq: dict, labels: dict) -> str:
    L, A = labels["locations"], labels["actions"]
    return " → ".join(f"{L.get(s['location'], s['location'])}"
                      f"（{A.get(s['action'], s['action'])}）" for s in seq["steps"])


def render_en(seq: dict, labels_en: dict) -> str:
    L, A = labels_en["locations"], labels_en["actions"]
    return " → ".join(f"{L.get(s['location'], s['location'])}"
                      f" ({A.get(s['action'], s['action'])})" for s in seq["steps"])


def load_stimuli() -> dict[tuple[str, str], dict]:
    """(persona, policy) -> sequence dict, via the manifest."""
    manifest = SEQ_DIR / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(
            f"{manifest} not found — run:\n"
            "  python -m experiments.rq3.gen_sequences --config configs/rq3_study.json")
    out = {}
    with open(manifest, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seq = json.loads((SEQ_DIR / row["file"]).read_text(encoding="utf-8"))
            out[(row["personality"], row["policy"])] = seq
    return out


def draw_distractors(personas: list[dict]) -> dict[str, list[str]]:
    """Two fixed distractor ids per persona, drawn once."""
    rng = np.random.default_rng(DISTRACTOR_SEED)
    ids = [p["id"] for p in personas]
    return {pid: sorted(rng.choice([q for q in ids if q != pid],
                                   size=N_DISTRACTORS, replace=False).tolist())
            for pid in ids}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    personas = json.loads(PERSONAS.read_text(encoding="utf-8"))["personas"]
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    labels_en = json.loads(LABELS_EN.read_text(encoding="utf-8"))
    stimuli = load_stimuli()
    by_id = {p["id"]: p for p in personas}
    distractors = draw_distractors(personas)
    OUT.mkdir(parents=True, exist_ok=True)

    missing = [(p["id"], pol) for p in personas for pol in POLICIES
               if (p["id"], pol) not in stimuli]
    if missing:
        raise SystemExit(f"missing stimuli: {missing}")

    # ---- assignment table ----------------------------------------------------
    rows = []
    for g in range(N_GROUPS):
        for i, p in enumerate(personas):
            policy = POLICIES[(i + g) % N_GROUPS]
            seq = stimuli[(p["id"], policy)]
            opts = sorted([p["id"], *distractors[p["id"]]])
            rows.append({
                "group": g + 1, "persona": p["id"], "policy": policy,
                "sequence_id": seq["meta"]["sequence_id"],
                "seed": seq["meta"]["seed"],
                "options": "|".join(opts),
                "correct": p["id"],
                "correct_position": opts.index(p["id"]) + 1,
            })
    with open(OUT / "assignment.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- all stimuli as text (zh = participant-facing, en = record) ----------
    with open(OUT / "stimuli.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence_id", "persona", "policy", "seed", "trail_zh", "trail_en"])
        for p in personas:
            for pol in POLICIES:
                seq = stimuli[(p["id"], pol)]
                w.writerow([seq["meta"]["sequence_id"], p["id"], pol,
                            seq["meta"]["seed"], render(seq, labels),
                            render_en(seq, labels_en)])

    # ---- per-group paste-ready scripts ---------------------------------------
    for g in range(N_GROUPS):
        lines = [
            f"# 第 {g + 1} 组问卷材料",
            "",
            f"每位受试者看 6 条轨迹，每个角色各一条。策略分配见下表（**受试者不可见**）。",
            "",
            "| 角色 | 策略 | 序列 |",
            "|---|---|---|",
        ]
        for i, p in enumerate(personas):
            policy = POLICIES[(i + g) % N_GROUPS]
            lines.append(f"| {p['id']} | {POLICY_ZH[policy]} | "
                         f"{stimuli[(p['id'], policy)]['meta']['sequence_id']} |")
        lines += ["", "---", "", "## 板块一 · 盲猜（6 题，顺序随机）", ""]

        for i, p in enumerate(personas):
            policy = POLICIES[(i + g) % N_GROUPS]
            seq = stimuli[(p["id"], policy)]
            opts = sorted([p["id"], *distractors[p["id"]]])
            lines += [
                f"### 角色 {i + 1}",
                "",
                f"<!-- {seq['meta']['sequence_id']} | persona={p['id']} | "
                f"system={policy} | seed={seq['meta']['seed']} -->",
                "",
                "[此处嵌入视频]",
                "",
                render(seq, labels),
                "",
                "**问 1：下面哪段描述最像这个角色？**（选项顺序由 Qualtrics 随机）",
                "",
            ]
            for q in opts:
                mark = "  ← 正确答案" if q == p["id"] else ""
                lines.append(f"- {by_id[q]['description_zh']}{mark}")
            lines += [
                "",
                "**问 2：给这个角色打分**（1–7 滑条）",
                "",
                "- 封闭保守 ←→ 开放好奇",
                "- 散漫随性 ←→ 自律有条理",
                "- 安静内向 ←→ 活跃外向",
                "- 冷淡强硬 ←→ 温和随和",
                "- 沉稳平静 ←→ 焦虑易怒",
                "",
            ]

        lines += ["---", "",
                  "## 板块二 · 配对比较（2 题，必须放在板块一全部完成之后）", ""]
        for n, idx in enumerate(preference_personas(g, len(personas)), 1):
            p = personas[idx]
            a, b = (stimuli[(p["id"], pol)] for pol in PREFERENCE_POLICIES)
            lines += [
                f"### 配对 {n}",
                "",
                f"<!-- persona={p['id']} | 甲={PREFERENCE_POLICIES[0]}"
                f"({a['meta']['sequence_id']}) | 乙={PREFERENCE_POLICIES[1]}"
                f"({b['meta']['sequence_id']}) | 甲乙左右位置由 Qualtrics 随机 -->",
                "",
                f"> 这个角色的设定是：**{p['description_zh']}**",
                "",
                f"**甲**：{render(a, labels)}",
                "",
                f"**乙**：{render(b, labels)}",
                "",
                "- 问 1：哪一段更符合上面的设定？　○ 甲　○ 乙　○ 两个差不多",
                "- 问 2：哪一段更像一个有自己脾气的活人？　○ 甲　○ 乙　○ 两个差不多",
                "",
            ]
        (OUT / f"group{g + 1}.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- English mirrors (dissertation appendix / ethics record) -------------
    # Same structure and assignment as the Chinese files; not shown to
    # participants. Kept as separate files so the participant-facing material
    # stays single-language.
    sliders_en = [
        "- Closed / conventional ←→ Open / curious",
        "- Careless / impulsive ←→ Organised / disciplined",
        "- Quiet / introverted ←→ Active / extraverted",
        "- Cold / harsh ←→ Warm / agreeable",
        "- Calm / stable ←→ Anxious / volatile",
    ]
    for g in range(N_GROUPS):
        lines = [
            f"# Group {g + 1} survey material (English record)",
            "",
            "Participants see the Chinese version; this mirror documents it.",
            "",
            "| Persona | Policy | Sequence |",
            "|---|---|---|",
        ]
        for i, p in enumerate(personas):
            policy = POLICIES[(i + g) % N_GROUPS]
            lines.append(f"| {p['id']} | {policy} | "
                         f"{stimuli[(p['id'], policy)]['meta']['sequence_id']} |")
        lines += ["", "---", "", "## Block 1 · Blind identification "
                  "(6 trials, order randomised)", ""]
        for i, p in enumerate(personas):
            policy = POLICIES[(i + g) % N_GROUPS]
            seq = stimuli[(p["id"], policy)]
            opts = sorted([p["id"], *distractors[p["id"]]])
            lines += [
                f"### Character {i + 1}",
                "",
                f"<!-- {seq['meta']['sequence_id']} | persona={p['id']} | "
                f"system={policy} | seed={seq['meta']['seed']} -->",
                "",
                "[video embedded here]",
                "",
                render_en(seq, labels_en),
                "",
                "**Q1: Which description best fits this character?** "
                "(choice order randomised by Qualtrics)",
                "",
            ]
            for q in opts:
                mark = "  ← correct" if q == p["id"] else ""
                lines.append(f"- {by_id[q]['description_en']}{mark}")
            lines += ["", "**Q2: Rate this character** (1–7 sliders)", "",
                      *sliders_en, ""]
        lines += ["---", "", "## Block 2 · Paired comparison "
                  "(2 trials, strictly after Block 1)", ""]
        for n, idx in enumerate(preference_personas(g, len(personas)), 1):
            p = personas[idx]
            a, b = (stimuli[(p["id"], pol)] for pol in PREFERENCE_POLICIES)
            lines += [
                f"### Pair {n}",
                "",
                f"<!-- persona={p['id']} | A={PREFERENCE_POLICIES[0]}"
                f"({a['meta']['sequence_id']}) | B={PREFERENCE_POLICIES[1]}"
                f"({b['meta']['sequence_id']}) | left/right randomised -->",
                "",
                f"> This character is meant to be: **{p['description_en']}**",
                "",
                f"**A**: {render_en(a, labels_en)}",
                "",
                f"**B**: {render_en(b, labels_en)}",
                "",
                "- Q1: Which sequence better fits the description? "
                "○ A ○ B ○ About the same",
                "- Q2: Which reads more like a person with a character of "
                "their own? ○ A ○ B ○ About the same",
                "",
            ]
        (OUT / f"group{g + 1}_en.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- console -------------------------------------------------------------
    print("distractor sets (fixed across groups and policies):")
    for p in personas:
        print(f"  {p['id']}  vs  {', '.join(distractors[p['id']])}")
    print("\nLatin square (persona -> policy):")
    print("        " + "".join(f"{p['id']:>10}" for p in personas))
    for g in range(N_GROUPS):
        print(f"  组{g + 1}  " + "".join(
            f"{POLICY_ZH[POLICIES[(i + g) % N_GROUPS]]:>10}"
            for i in range(len(personas))))
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["policy"]] = counts.get(r["policy"], 0) + 1
    print(f"\ncells: {len(rows)} (= {N_GROUPS} groups x {len(personas)} personas); "
          f"per policy {counts}")
    print(f"\nwritten to {OUT}:")
    for f in ("group1.md", "group2.md", "group3.md", "assignment.csv", "stimuli.csv"):
        print(f"  {f}")


if __name__ == "__main__":
    main()
