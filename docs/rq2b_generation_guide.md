# NPC decision-case generation guide (Study 2B)

You are generating decision cases for a fantasy-village NPC simulation. Each case
describes an NPC (a personality), a situation (where it has been recently, what it
can do now), and your judgement of what this NPC would choose. Judge as a
thoughtful human observer: stay in character, weigh comfort, boredom, and
curiosity against the personality. There is no "correct" answer; give the choice
this specific NPC would most plausibly make.

## Output format

Reply with a JSON array only — no prose around it. First element is a metadata
header; every other element is one case:

```json
[
  {"_meta": {"source": "<your model name and version>"}},
  {
    "personality": {"O": 0.7, "C": -0.4, "E": 0.6, "A": 0.1, "N": -0.2},
    "decision_type": "location",
    "recent_locations": ["market", "market", "tavern"],
    "candidates": ["tavern", "library", "arena"],
    "choice": "tavern",
    "reason": "one short sentence"
  },
  {
    "personality": {"O": -0.8, "C": 0.9, "E": -0.5, "A": 0.3, "N": 0.4},
    "decision_type": "action",
    "selected_location": "library",
    "recent_locations": ["chapel", "library"],
    "recent_actions_same_location": ["read"],
    "candidates": ["read", "research", "discuss"],
    "choice": "read",
    "reason": "one short sentence"
  }
]
```

## Personality

Five traits, each a number in [-1, 1] (0 = average):
O openness (curiosity, novelty seeking), C conscientiousness (order, routine),
E extraversion (sociability, stimulation), A agreeableness (warmth, cooperation),
N neuroticism (anxiety, volatility). You invent the personality for each case.

## World vocabulary (use these names only; never invent new ones)

Locations:
- `tavern` — a lively drinking house: noisy, social, little privacy, the odd brawl
- `library` — quiet halls of books: solitary study, order, deep thought
- `chapel` — a calm sanctuary: ritual, reflection, quiet confession
- `market` — a bustling trade square: haggling, browsing, deal-making
- `forest` — wild ground beyond the walls: roaming, foraging, solitude
- `arena` — a fighting ground: combat bouts, training, spectacle, betting
- `infirmary` — the village healing house: sickbeds, herbs, quiet duty
  **(test-only: appears ONLY in Held-out location batches)**

Actions per location (an action case's `candidates` must be this full list):
- tavern: `chat` (friendly talk), `drink` (loosen up alone or in company),
  `brawl` (start or join a fist-fight)
- library: `read` (quiet reading), `research` (dig into a problem),
  `discuss` (debate ideas with another scholar)
- chapel: `pray` (formal ritual), `meditate` (silent stillness),
  `confess` (unburden to the priest)
- market: `haggle` (push for a better price), `browse` (wander the stalls),
  `trade` (do business with a partner)
- forest: `explore` (push into unknown ground), `forage` (gather food),
  `rest` (settle down and recover)
- arena: `fight` (real bout), `spar` (practice bout), `drill` (disciplined
  training), `coach` (train someone else), `spectate` (watch the bouts),
  `bet` (wager on outcomes)
- infirmary: `tend` (care for the sick), `assist` (help the healer with tasks),
  `vigil` (keep quiet night watch at a bedside)

## Rules

1. `decision_type` is `"location"` or `"action"`; aim for half of each per batch.
2. Location cases: `candidates` = 2–6 distinct location names;
   `recent_locations` = 0–3 names, oldest first (the NPC's last visits);
   no `selected_location`, no `recent_actions_same_location`.
3. Action cases: `selected_location` = where the NPC is; `candidates` = that
   location's full action list; `recent_locations` = 0–3 names, oldest first,
   and its last entry must equal `selected_location` (it may be just that one
   name, or empty); `recent_actions_same_location` = 0–3 action names of
   that same location (what it just did there — empty if it only just arrived).
4. `choice` must be one of `candidates`. Numbers only in `personality`; never
   output feature values or probabilities.
5. Vary personalities: within a batch, each trait should appear high (> 0.3),
   middling, and low (< -0.3) in at least ~20% of cases each. About 20% of
   cases should have empty history. Vary candidate subsets and history patterns
   (repeats, alternations, fresh arrivals).
6. Consider the history: an NPC that has repeated one place or act may be bored
   of it (or, if it loves routine, comforted by it) — let the personality decide.

## Batch types (the requester tells you which one)

- **General batch**: rules above, plus: never use `infirmary` (or its actions)
  and never give personalities with O > 0.5 together with C < -0.5. All six
  other locations, including `arena`, are fine.
- **Personality batch (test)**: every case's personality has O > 0.5 AND
  C < -0.5; no infirmary content.
- **Held-out location batch (test)**: every case involves `infirmary` — as an
  action case at the infirmary, or a location case with `infirmary` among the
  candidates.

Produce the number of cases the requester asks for (default 50).
