"""Emit the Study-3 survey as a Qualtrics ``.qsf`` file ready to import.

Qualtrics builds a survey from clicks; this script builds the same survey from
the stimulus files, so the questionnaire cannot drift from the sequences it is
supposed to show. Import with *Create new project -> From a file -> .qsf*.

What the generated survey does, and why each piece is there:

- **Group assignment is a Survey Flow randomiser** with *Evenly Present
  Elements*, not three separate survey links. Participants take one link; the
  flow deals them into group 1, 2 or 3 in balanced rotation. The group number
  is the one embedded field the survey records.
- **Stimuli are identified by question name, not by embedded data.** Each
  (persona, policy) pair is a fixed question whose export tag is e.g.
  ``H1_scorer_choice``, so the exported column header already says which
  stimulus it is. Only presentation *order* is randomised, and Qualtrics
  exports that separately as the block randomiser's display order. Setting
  persona/policy as embedded data per block would re-record, by hand, what the
  column names already carry.
- **Blind identification precedes preference comparison** because the
  comparison block reveals that several systems exist.
- **Trait sliders sit above the forced choice** so the candidate descriptions
  do not anchor the ratings. They share a page: the participant can see the
  descriptions before rating, which weakens but does not remove the anchoring;
  splitting them across a page break was considered and not adopted.
- **Choice order is randomised per question**, so the correct description does
  not sit in a fixed position.

Video: pass ``--video-base URL`` to embed ``<video>`` tags pointing at
``URL/<sequence_id>.mp4``. Without it each stimulus carries a visible
placeholder, and the survey is complete apart from the clips — build it now,
upload the recordings to the Qualtrics file library later, and re-import.

Run from ``code/``:
    python -m experiments.rq3.build_qsf --probe      # 2-question format check
    python -m experiments.rq3.build_qsf              # the real thing
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiments.rq3.build_survey import (
    DATA, N_GROUPS, OUT, POLICIES, PREFERENCE_POLICIES, draw_distractors,
    load_stimuli, preference_personas, render,
)

SURVEY_NAME = "NPC 行为与性格判断"
# Qualtrics ids are a prefix plus a fixed-width alphanumeric body; short or
# obviously fabricated ones are a plausible reason for an import to be refused.
SURVEY_ID = "SV_0rq3study000001"
RESPONSE_SET = "RS_0rq3study000001"
POLICY_TAG = {"scorer": "scorer", "nonlinear_2b": "nn",
              "agnostic_nonlinear_2b": "agno"}

TRAITS = [
    ("O", "封闭保守 ←→ 开放好奇"),
    ("C", "散漫随性 ←→ 自律有条理"),
    ("E", "安静内向 ←→ 活跃外向"),
    ("A", "冷淡强硬 ←→ 温和随和"),
    ("N", "沉稳平静 ←→ 焦虑易怒"),
]

CONSENT = """<p>这是曼彻斯特大学机器人学硕士论文的一项研究。你会看到几段游戏角色的行为记录，
需要判断这些角色分别是什么样的人。全程约十分钟。</p>
<p>研究不收集任何能识别你身份的信息，只保存你的作答。你可以在任何时候关闭页面退出，
已作答的部分不会被使用。</p>"""

HOWTO = """<p>下面每一页显示一个游戏角色连续十次的行动记录。这个世界里有酒馆、集市、
教堂、图书馆、树林、竞技场几个地方，角色每次自己决定去哪里、做什么。</p>
<p>每页先给这个角色的性格打分，再从三段描述里挑出最像他的一段。凭直觉即可，没有标准答案。</p>"""

COMPARE_INTRO = """<p>接下来的两页换一种做法：先告诉你这个角色的设定，
再给你两段由<strong>不同系统</strong>生成的行为记录。请判断哪一段更符合设定。</p>"""

THANKS = "<p>问卷到此结束，感谢参与。</p>"


class Builder:
    """Accumulates questions and blocks, then assembles the .qsf envelope."""

    def __init__(self, survey_id: str = SURVEY_ID):
        self.sid = survey_id
        self.questions: list[dict] = []
        self.blocks: list[dict] = []
        self._qid = 0
        self._bid = 0
        self._fid = 0

    # -- ids -----------------------------------------------------------------
    def next_qid(self) -> str:
        self._qid += 1
        return f"QID{self._qid}"

    def next_bid(self) -> str:
        self._bid += 1
        return f"BL_{self._bid}"

    def next_fid(self) -> str:
        self._fid += 1
        return f"FL_{self._fid}"

    # -- questions -----------------------------------------------------------
    def _add(self, payload: dict) -> str:
        self.questions.append({
            "SurveyID": self.sid,
            "Element": "SQ",
            "PrimaryAttribute": payload["QuestionID"],
            "SecondaryAttribute": payload["QuestionText"][:100],
            "TertiaryAttribute": None,
            "Payload": payload,
        })
        return payload["QuestionID"]

    def text(self, tag: str, html: str) -> str:
        """Descriptive text — shows content, collects nothing."""
        qid = self.next_qid()
        return self._add({
            "QuestionText": html,
            "DataExportTag": tag,
            "QuestionType": "DB",
            "Selector": "TB",
            "Configuration": {"QuestionDescriptionOption": "UseText"},
            "QuestionDescription": tag,
            "ChoiceOrder": [],
            "Validation": {"Settings": {"Type": "None"}},
            "GradingData": [],
            "Language": [],
            "NextChoiceId": 1,
            "NextAnswerId": 1,
            "QuestionID": qid,
        })

    def choice(self, tag: str, text: str, options: list[str],
               randomise: bool = True, force: bool = True) -> str:
        qid = self.next_qid()
        payload = {
            "QuestionText": text,
            "DataExportTag": tag,
            "QuestionType": "MC",
            "Selector": "SAVR",
            "SubSelector": "TX",
            "Configuration": {"QuestionDescriptionOption": "UseText"},
            "QuestionDescription": tag,
            "Choices": {str(i): {"Display": o} for i, o in enumerate(options, 1)},
            "ChoiceOrder": [str(i) for i in range(1, len(options) + 1)],
            "Validation": {"Settings": {
                "ForceResponse": "ON" if force else "OFF",
                "ForceResponseType": "ON" if force else "OFF",
                "Type": "None"}},
            "GradingData": [],
            "Language": [],
            "NextChoiceId": len(options) + 1,
            "NextAnswerId": 1,
            "QuestionID": qid,
        }
        if randomise:
            payload["Randomization"] = {
                "Advanced": None, "TotalRandSubset": "", "Type": "All"}
        return self._add(payload)

    def sliders(self, tag: str, text: str, rows: list[str]) -> str:
        """One 1-7 slider per row; SnapToGrid makes the export integers."""
        qid = self.next_qid()
        return self._add({
            "QuestionText": text,
            "DataExportTag": tag,
            "QuestionType": "Slider",
            "Selector": "HSLIDER",
            "Configuration": {
                "QuestionDescriptionOption": "UseText",
                "CSSliderMin": 1,
                "CSSliderMax": 7,
                "GridLines": 7,
                "SnapToGrid": True,
                "CustomStart": False,
                "NotApplicable": False,
                "MobileFirst": True,
                "NumDecimals": "0",
                "ShowValue": True,
            },
            "Choices": {str(i): {"Display": r} for i, r in enumerate(rows, 1)},
            "ChoiceOrder": [str(i) for i in range(1, len(rows) + 1)],
            "Validation": {"Settings": {
                "ForceResponse": "ON", "ForceResponseType": "ON", "Type": "None"}},
            "GradingData": [],
            "Language": [],
            "Labels": [],
            "NextChoiceId": len(rows) + 1,
            "NextAnswerId": 1,
            "QuestionID": qid,
        })

    # -- blocks --------------------------------------------------------------
    def block(self, name: str, qids: list[str]) -> str:
        bid = self.next_bid()
        self.blocks.append({
            "Type": "Standard",
            "SubType": "",
            "Description": name,
            "ID": bid,
            "BlockElements": [{"Type": "Question", "QuestionID": q} for q in qids],
            "Options": {"BlockLocking": "false", "RandomizeQuestions": "false",
                        "BlockVisibility": "Collapsed"},
        })
        return bid

    # -- flow helpers --------------------------------------------------------
    def f_block(self, bid: str) -> dict:
        return {"Type": "Block", "ID": bid, "FlowID": self.next_fid(),
                "Autofill": []}

    def f_embedded(self, field: str, value: str) -> dict:
        return {"Type": "EmbeddedData", "FlowID": self.next_fid(),
                "EmbeddedData": [{"Description": field, "Type": "Custom",
                                  "Field": field, "VariableType": "String",
                                  "DataVisibility": [], "AnalyzeText": False,
                                  "Value": value}]}

    def f_randomiser(self, flows: list[dict], subset: int,
                     even: bool) -> dict:
        return {"Type": "BlockRandomizer", "FlowID": self.next_fid(),
                "SubSet": subset, "ByGroup": False, "EvenPresentation": even,
                "Flow": flows}

    def f_group(self, name: str, flows: list[dict]) -> dict:
        return {"Type": "Group", "FlowID": self.next_fid(),
                "Description": name, "Flow": flows}

    def f_end_if_choice(self, qid: str, choice: int, why: str) -> dict:
        """Terminate the survey when ``qid`` has ``choice`` selected."""
        loc = f"q://{qid}/SelectableChoice/{choice}"
        return {
            "Type": "Branch", "FlowID": self.next_fid(), "Description": why,
            "BranchLogic": {"0": {"0": {
                "LogicType": "Question", "QuestionID": qid,
                "QuestionIsInLoop": "no", "ChoiceLocator": loc,
                "Operator": "Selected", "QuestionIDFromLocator": qid,
                "LeftOperand": loc, "Type": "Expression", "Description": why},
                "Type": "If"}, "Type": "BooleanExpression"},
            "Flow": [{"Type": "EndSurvey", "FlowID": self.next_fid()}],
        }

    # -- envelope ------------------------------------------------------------
    def qsf(self, flow: list[dict]) -> dict:
        """Assemble the import envelope.

        Qualtrics rejects the whole file with one generic message when any
        skeleton element is missing, so SCO/PROJ/STAT/RS are emitted with
        minimal payloads even though nothing here uses scoring or statistics,
        and a Trash block is included because the editor expects one to exist.
        Owner/brand ids are left null: the import assigns the importing
        account's own, and a fabricated value is a value it can fail to match.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        blocks = self.blocks + [{
            "Type": "Trash", "SubType": "", "Description": "Trash / Unused Questions",
            "ID": self.next_bid(), "BlockElements": []}]
        elements = [
            {"SurveyID": self.sid, "Element": "BL",
             "PrimaryAttribute": "Survey Blocks", "SecondaryAttribute": None,
             "TertiaryAttribute": None, "Payload": blocks},
            {"SurveyID": self.sid, "Element": "FL",
             "PrimaryAttribute": "Survey Flow", "SecondaryAttribute": None,
             "TertiaryAttribute": None,
             "Payload": {"Type": "Root", "FlowID": "FL_0", "Flow": flow,
                         "Properties": {"Count": self._fid + 1,
                                        "RemovedFieldsets": []}}},
            {"SurveyID": self.sid, "Element": "SO",
             "PrimaryAttribute": "Survey Options", "SecondaryAttribute": None,
             "TertiaryAttribute": None,
             "Payload": {
                 "BackButton": "false",          # no revising an earlier trail
                 "SaveAndContinue": "true",
                 "SurveyProtection": "PublicSurvey",
                 "BallotBoxStuffingPrevention": "false",
                 "NoIndex": "Yes",
                 "SecureResponseFiles": "true",
                 "SurveyExpiration": "None",
                 "SurveyTermination": "DefaultMessage",
                 "Header": "", "Footer": "",
                 "ProgressBarDisplay": "Text",
                 "PartialData": "+1 week",
                 "ValidationMessage": "", "PreviousButton": "",
                 "NextButton": "", "SurveyTitle": SURVEY_NAME,
                 "SurveyLanguage": "ZH-S",
                 "SurveyStartDate": "0000-00-00 00:00:00",
                 "SurveyExpirationDate": "0000-00-00 00:00:00"}},
            {"SurveyID": self.sid, "Element": "SCO",
             "PrimaryAttribute": "Scoring", "SecondaryAttribute": None,
             "TertiaryAttribute": None,
             "Payload": {"ScoringCategories": [], "ScoringCategoryGroups": [],
                         "ScoringSummaryCategory": None,
                         "ScoringSummaryAfterQuestions": 0,
                         "ScoringSummaryAfterSurvey": 0,
                         "DefaultScoringCategory": None,
                         "AutoScoringCategory": None}},
            {"SurveyID": self.sid, "Element": "PROJ",
             "PrimaryAttribute": "CORE", "SecondaryAttribute": None,
             "TertiaryAttribute": "1.1.0",
             "Payload": {"ProjectCategory": "CORE", "SchemaVersion": "1.1.0"}},
            {"SurveyID": self.sid, "Element": "STAT",
             "PrimaryAttribute": "Survey Statistics", "SecondaryAttribute": None,
             "TertiaryAttribute": None,
             "Payload": {"MobileCompatible": True, "ID": "Survey Statistics"}},
            {"SurveyID": self.sid, "Element": "QC",
             "PrimaryAttribute": "Survey Question Count",
             "SecondaryAttribute": str(len(self.questions)),
             "TertiaryAttribute": None, "Payload": None},
            {"SurveyID": self.sid, "Element": "RS",
             "PrimaryAttribute": RESPONSE_SET, "SecondaryAttribute": "Default Response Set",
             "TertiaryAttribute": None, "Payload": None},
        ] + self.questions
        return {
            "SurveyEntry": {
                "SurveyID": self.sid,
                "SurveyName": SURVEY_NAME,
                "SurveyDescription": None,
                "SurveyOwnerID": None,
                "SurveyBrandID": None,
                "DivisionID": None,
                "SurveyLanguage": "ZH-S",
                "SurveyActiveResponseSet": RESPONSE_SET,
                "SurveyStatus": "Inactive",
                "SurveyStartDate": "0000-00-00 00:00:00",
                "SurveyExpirationDate": "0000-00-00 00:00:00",
                "SurveyCreationDate": now,
                "CreatorID": None,
                "LastModified": now,
                "LastAccessed": "0000-00-00 00:00:00",
                "LastActivated": "0000-00-00 00:00:00",
                "Deleted": None,
            },
            "SurveyElements": elements,
        }


def to_txt(b: Builder) -> str:
    """Render the same survey in Qualtrics' Advanced Format text import.

    This is the fallback route when a ``.qsf`` will not import. The text format
    is documented and forgiving, but carries only blocks and questions — no
    Survey Flow, so group assignment, block order randomisation and the consent
    branch are rebuilt by hand afterwards (block names begin ``G1_``/``G2_``/
    ``G3_`` to make that mechanical). It also has no slider type, so the trait
    question arrives as a 1-7 matrix; the data are identical and the type can be
    switched in the editor afterwards.
    """
    payloads = {q["Payload"]["QuestionID"]: q["Payload"] for q in b.questions}
    out = ["[[AdvancedFormat]]", ""]
    for blk in b.blocks:
        if blk["Type"] == "Trash":
            continue
        out += [f"[[Block:{blk['Description']}]]", ""]
        for el in blk["BlockElements"]:
            p = payloads[el["QuestionID"]]
            # the text format ends a field at the next [[ marker, so any
            # embedded newline inside HTML would split the question text
            text = " ".join(p["QuestionText"].split())
            tag = f"[[ID:{p['DataExportTag']}]]"   # follows the type marker
            if p["QuestionType"] == "DB":
                out += ["[[Question:DB]]", tag, text, ""]
            elif p["QuestionType"] == "MC":
                out += ["[[Question:MC:SingleAnswer]]", tag, text, "[[Choices]]"]
                out += [p["Choices"][k]["Display"] for k in p["ChoiceOrder"]]
                out.append("")
            elif p["QuestionType"] == "Slider":
                out += ["[[Question:Matrix]]", tag, text, "[[Choices]]"]
                out += [p["Choices"][k]["Display"] for k in p["ChoiceOrder"]]
                out += ["[[Answers]]", *[str(i) for i in range(1, 8)], ""]
            else:
                raise ValueError(f"no text-format mapping: {p['QuestionType']}")
    return "\n".join(out) + "\n"


def trail_html(trail: str, seq_id: str, video_base: str | None) -> str:
    """Video (or a placeholder) above the same sequence as text."""
    if video_base:
        clip = (f'<video controls preload="metadata" '
                f'style="max-width:100%;margin-bottom:1em" '
                f'src="{video_base.rstrip("/")}/{seq_id}.mp4"></video>')
    else:
        clip = ('<div style="padding:2em;border:2px dashed #bbb;color:#888;'
                'text-align:center;margin-bottom:1em">'
                f'[视频占位 {seq_id} — 录制后用 --video-base 重新生成]</div>')
    return (clip + '<p style="font-size:1.15em;line-height:2">'
            + trail + '</p>')


def build(video_base: str | None, probe: bool) -> Builder:
    personas = json.loads((DATA / "rq3_personas.json").read_text(
        encoding="utf-8"))["personas"]
    labels = json.loads((DATA / "rq3_labels_zh.json").read_text(encoding="utf-8"))
    stimuli = load_stimuli()
    by_id = {p["id"]: p for p in personas}
    distractors = draw_distractors(personas)
    b = Builder()

    # ---- fixed opening -------------------------------------------------------
    consent_qid = b.choice("consent", "你是否同意参与？", ["同意", "不同意"],
                           randomise=False)
    intro = b.block("引言", [b.text("consent_text", CONSENT), consent_qid])
    howto = b.block("说明", [b.text("howto", HOWTO)])
    # declining ends the survey rather than skipping questions, so a refusal
    # leaves no partial stimulus responses behind
    decline = b.f_end_if_choice(consent_qid, 2, "不同意参与")

    def stimulus_block(pid: str, policy: str, g: int) -> str:
        """One trail: clip + text, trait sliders, then the forced choice.

        The block is named with its group so that a hand-built Survey Flow (the
        text-import route, which carries no flow) can be assembled by reading
        block names alone. Question export tags stay group-free, since a
        stimulus is one stimulus whichever group meets it."""
        seq = stimuli[(pid, policy)]
        tag = f"{pid}_{POLICY_TAG[policy]}"
        opts = sorted([pid, *distractors[pid]])
        return b.block(f"G{g + 1}_{tag}", [
            b.text(f"{tag}_trail",
                   trail_html(render(seq, labels), seq["meta"]["sequence_id"],
                              video_base)),
            b.sliders(f"{tag}_rate", "你觉得这个角色大概是怎样的人？",
                      [d for _, d in TRAITS]),
            b.choice(f"{tag}_choice", "下面哪段描述最像这个角色？",
                     [by_id[q]["description_zh"] for q in opts]),
        ])

    if probe:
        # smallest file that exercises every question type and the randomiser
        p0 = personas[0]["id"]
        blk = stimulus_block(p0, POLICIES[0], 0)
        end = b.block("结束", [b.text("thanks", THANKS)])
        flow = [b.f_block(intro), decline, b.f_block(howto),
                b.f_randomiser([b.f_block(blk)], 1, True), b.f_block(end)]
        return b, flow

    # ---- three groups --------------------------------------------------------
    group_flows = []
    for g in range(N_GROUPS):
        blind = [b.f_block(stimulus_block(p["id"], POLICIES[(i + g) % N_GROUPS], g))
                 for i, p in enumerate(personas)]

        cmp_flows = []
        for idx in preference_personas(g, len(personas)):
            p = personas[idx]
            a, c = (stimuli[(p["id"], pol)] for pol in PREFERENCE_POLICIES)
            tag = f"cmp_{p['id']}"
            cmp_flows.append(b.f_block(b.block(f"G{g + 1}_{tag}", [
                b.text(f"{tag}_stim",
                       f'<p>这个角色的设定是：<strong>{p["description_zh"]}</strong></p>'
                       f'<hr><p><strong>甲</strong></p>'
                       + trail_html(render(a, labels), a["meta"]["sequence_id"],
                                    video_base)
                       + f'<hr><p><strong>乙</strong></p>'
                       + trail_html(render(c, labels), c["meta"]["sequence_id"],
                                    video_base)),
                # "about the same" is offered rather than forcing a side: a
                # forced binary converts indifference into a coin flip, which
                # at n = 21 is indistinguishable from a real preference. Ties
                # are excluded from the preference test and reported as a rate
                # in their own right — a high tie rate is the finding that the
                # policies are not tellable apart.
                b.choice(f"{tag}_fit", "哪一段更符合上面的设定？",
                         ["甲", "乙", "两个差不多"], randomise=False),
                b.choice(f"{tag}_alive", "哪一段更像一个有自己脾气的活人？",
                         ["甲", "乙", "两个差不多"], randomise=False),
            ])))

        cmp_intro = b.block(f"比较说明_{g + 1}", [b.text(f"cmp_intro_{g + 1}",
                                                     COMPARE_INTRO)])
        group_flows.append(b.f_group(f"第 {g + 1} 组", [
            b.f_embedded("group", str(g + 1)),
            b.f_randomiser(blind, len(blind), False),   # order only
            b.f_block(cmp_intro),
            b.f_randomiser(cmp_flows, len(cmp_flows), False),
        ]))

    demo = b.block("背景", [
        b.choice("age", "你的年龄段？",
                 ["18–24", "25–34", "35–44", "45 及以上"], randomise=False),
        b.choice("gaming", "你平时玩电子游戏吗？",
                 ["几乎不玩", "偶尔玩", "经常玩"], randomise=False),
    ])
    end = b.block("结束", [b.text("thanks", THANKS)])

    flow = [b.f_block(intro), decline, b.f_block(howto),
            b.f_randomiser(group_flows, 1, True),   # even -> balanced groups
            b.f_block(demo), b.f_block(end)]
    return b, flow


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video-base", default=None,
                    help="URL prefix; clips are <base>/<sequence_id>.mp4")
    ap.add_argument("--probe", action="store_true",
                    help="emit a 1-stimulus file to verify the format imports")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    b, flow = build(args.video_base, args.probe)
    stem = "rq3_probe" if args.probe else "rq3_survey"
    OUT.mkdir(parents=True, exist_ok=True)

    # text is written before qsf: qsf() appends the Trash block, so calling it
    # first would leak that block into the text output
    txt_path = OUT / f"{stem}.txt"
    txt_path.write_text(to_txt(b), encoding="utf-8")

    qsf_path = Path(args.out) if args.out else OUT / f"{stem}.qsf"
    qsf_path.parent.mkdir(parents=True, exist_ok=True)
    qsf_path.write_text(json.dumps(b.qsf(flow), ensure_ascii=False, indent=1),
                        encoding="utf-8")

    print(f"blocks    {len(b.blocks)}")
    print(f"questions {len(b.questions)}")
    print(f"video     {args.video_base or 'placeholder (re-run with --video-base)'}")
    print(f"written   {qsf_path}")
    print(f"          {txt_path}   (fallback: text import, flow rebuilt by hand)")


if __name__ == "__main__":
    main()
