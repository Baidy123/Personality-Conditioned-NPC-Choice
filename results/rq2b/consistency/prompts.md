# Re-annotation batch (annotator consistency check)

You are judging what a fantasy-village NPC would do. Stay in character; weigh
comfort, boredom, and curiosity against the personality. There is no "correct"
answer — give the choice this specific NPC would most plausibly make.

Personality traits run from -1 to +1: O openness (curiosity, novelty seeking),
C conscientiousness (order, routine), E extraversion (sociability, stimulation),
A agreeableness (warmth, cooperation), N neuroticism (anxiety, volatility).

Answer every case. Reply with a JSON object only — no prose:

    {"<case id>": "<letter>", ...}

### Case general_01#75
- Personality: O +0.6, C -0.5, E +0.9, A -0.9, N +0.8
- Recently visited (oldest first): forest, chapel
- It may now go to:
    A. chapel — a calm sanctuary: ritual, reflection, quiet confession
    B. market — a bustling trade square: haggling, browsing, deal-making
    C. forest — wild ground beyond the walls: roaming, foraging, solitude
- Where does it go? Answer with the letter only.

### Case general_01#93
- Personality: O +0.4, C -0.7, E +0.2, A -0.5, N +0.0
- It is at the chapel (a calm sanctuary: ritual, reflection, quiet confession)
- Just did there (oldest first): (only just arrived)
- It may now:
    A. pray
    B. meditate
    C. confess
- What does it do? Answer with the letter only.

### Case general_01#94
- Personality: O -0.2, C -0.7, E -0.7, A +0.6, N +0.4
- It is at the arena (a fighting ground: combat bouts, training, spectacle, betting)
- Just did there (oldest first): drill, drill, spar
- It may now:
    A. fight
    B. spar
    C. drill
    D. coach
    E. spectate
    F. bet
- What does it do? Answer with the letter only.

### Case general_02#32
- Personality: O -0.2, C +0.0, E +0.0, A -0.2, N +0.2
- It is at the market (a bustling trade square: haggling, browsing, deal-making)
- Just did there (oldest first): browse, haggle, haggle
- It may now:
    A. haggle
    B. browse
    C. trade
- What does it do? Answer with the letter only.

### Case general_02#35
- Personality: O -0.5, C -0.7, E -0.7, A +0.6, N +0.6
- Recently visited (oldest first): library
- It may now go to:
    A. tavern — a lively drinking house: noisy, social, little privacy, the odd brawl
    B. chapel — a calm sanctuary: ritual, reflection, quiet confession
    C. forest — wild ground beyond the walls: roaming, foraging, solitude
    D. library — quiet halls of books: solitary study, order, deep thought
    E. market — a bustling trade square: haggling, browsing, deal-making
    F. arena — a fighting ground: combat bouts, training, spectacle, betting
- Where does it go? Answer with the letter only.

### Case general_05#94
- Personality: O -0.5, C -0.2, E +0.6, A -0.2, N -0.5
- Recently visited (oldest first): chapel, forest, forest
- It may now go to:
    A. arena — a fighting ground: combat bouts, training, spectacle, betting
    B. forest — wild ground beyond the walls: roaming, foraging, solitude
- Where does it go? Answer with the letter only.

### Case general_06#50
- Personality: O -0.5, C -0.2, E +0.4, A +0.2, N +0.6
- Recently visited (oldest first): arena
- It may now go to:
    A. forest — wild ground beyond the walls: roaming, foraging, solitude
    B. tavern — a lively drinking house: noisy, social, little privacy, the odd brawl
    C. market — a bustling trade square: haggling, browsing, deal-making
    D. arena — a fighting ground: combat bouts, training, spectacle, betting
    E. library — quiet halls of books: solitary study, order, deep thought
- Where does it go? Answer with the letter only.

### Case general_06#67
- Personality: O +0.6, C +0.4, E -0.9, A +0.0, N -0.5
- Recently visited (oldest first): tavern
- It may now go to:
    A. chapel — a calm sanctuary: ritual, reflection, quiet confession
    B. tavern — a lively drinking house: noisy, social, little privacy, the odd brawl
- Where does it go? Answer with the letter only.

### Case general_07#80
- Personality: O +0.6, C -0.5, E -0.2, A -0.2, N +0.8
- Recently visited (oldest first): arena
- It may now go to:
    A. arena — a fighting ground: combat bouts, training, spectacle, betting
    B. tavern — a lively drinking house: noisy, social, little privacy, the odd brawl
    C. chapel — a calm sanctuary: ritual, reflection, quiet confession
    D. library — quiet halls of books: solitary study, order, deep thought
    E. market — a bustling trade square: haggling, browsing, deal-making
    F. forest — wild ground beyond the walls: roaming, foraging, solitude
- Where does it go? Answer with the letter only.

### Case general_09#47
- Personality: O -0.2, C +0.0, E -0.7, A +0.0, N +0.4
- Recently visited (oldest first): forest, tavern
- It may now go to:
    A. forest — wild ground beyond the walls: roaming, foraging, solitude
    B. library — quiet halls of books: solitary study, order, deep thought
    C. tavern — a lively drinking house: noisy, social, little privacy, the odd brawl
    D. market — a bustling trade square: haggling, browsing, deal-making
- Where does it go? Answer with the letter only.

### Case general_09#84
- Personality: O -0.9, C +0.4, E +0.6, A -0.2, N +0.9
- Recently visited (oldest first): arena
- It may now go to:
    A. arena — a fighting ground: combat bouts, training, spectacle, betting
    B. library — quiet halls of books: solitary study, order, deep thought
- Where does it go? Answer with the letter only.

### Case general_09#100
- Personality: O -0.7, C +0.0, E -0.5, A -0.9, N -0.9
- Recently visited (oldest first): arena, arena
- It may now go to:
    A. chapel — a calm sanctuary: ritual, reflection, quiet confession
    B. library — quiet halls of books: solitary study, order, deep thought
    C. market — a bustling trade square: haggling, browsing, deal-making
    D. arena — a fighting ground: combat bouts, training, spectacle, betting
    E. forest — wild ground beyond the walls: roaming, foraging, solitude
    F. tavern — a lively drinking house: noisy, social, little privacy, the odd brawl
- Where does it go? Answer with the letter only.

### Case general_10#88
- Personality: O +0.8, C +0.2, E +0.9, A +0.6, N +0.8
- Recently visited (oldest first): library
- It may now go to:
    A. tavern — a lively drinking house: noisy, social, little privacy, the odd brawl
    B. forest — wild ground beyond the walls: roaming, foraging, solitude
    C. market — a bustling trade square: haggling, browsing, deal-making
    D. chapel — a calm sanctuary: ritual, reflection, quiet confession
    E. library — quiet halls of books: solitary study, order, deep thought
- Where does it go? Answer with the letter only.

### Case general_11#6
- Personality: O -0.5, C +0.2, E +0.4, A +0.9, N +0.6
- It is at the tavern (a lively drinking house: noisy, social, little privacy, the odd brawl)
- Just did there (oldest first): brawl
- It may now:
    A. chat
    B. drink
    C. brawl
- What does it do? Answer with the letter only.

### Case general_11#68
- Personality: O -0.2, C +0.8, E +0.9, A -0.2, N +0.0
- It is at the market (a bustling trade square: haggling, browsing, deal-making)
- Just did there (oldest first): haggle
- It may now:
    A. haggle
    B. browse
    C. trade
- What does it do? Answer with the letter only.

### Case general_11#93
- Personality: O -0.7, C +0.6, E -0.5, A -0.5, N -0.5
- Recently visited (oldest first): arena
- It may now go to:
    A. tavern — a lively drinking house: noisy, social, little privacy, the odd brawl
    B. chapel — a calm sanctuary: ritual, reflection, quiet confession
    C. library — quiet halls of books: solitary study, order, deep thought
    D. forest — wild ground beyond the walls: roaming, foraging, solitude
    E. market — a bustling trade square: haggling, browsing, deal-making
- Where does it go? Answer with the letter only.

### Case general_12#61
- Personality: O +0.2, C +0.6, E -0.5, A +0.9, N +0.0
- It is at the tavern (a lively drinking house: noisy, social, little privacy, the odd brawl)
- Just did there (oldest first): (only just arrived)
- It may now:
    A. chat
    B. drink
    C. brawl
- What does it do? Answer with the letter only.

### Case general_12#75
- Personality: O -0.9, C +0.4, E -0.2, A +0.8, N -0.5
- Recently visited (oldest first): (nothing recorded)
- It may now go to:
    A. chapel — a calm sanctuary: ritual, reflection, quiet confession
    B. forest — wild ground beyond the walls: roaming, foraging, solitude
- Where does it go? Answer with the letter only.

### Case general_13#48
- Personality: O +0.3, C +0.9, E +0.2, A -0.9, N +0.6
- It is at the tavern (a lively drinking house: noisy, social, little privacy, the odd brawl)
- Just did there (oldest first): drink, chat, brawl
- It may now:
    A. chat
    B. drink
    C. brawl
- What does it do? Answer with the letter only.

### Case general_14#53
- Personality: O -0.2, C +0.9, E +0.6, A -0.5, N +0.8
- Recently visited (oldest first): library, arena, chapel
- It may now go to:
    A. market — a bustling trade square: haggling, browsing, deal-making
    B. library — quiet halls of books: solitary study, order, deep thought
    C. chapel — a calm sanctuary: ritual, reflection, quiet confession
- Where does it go? Answer with the letter only.

### Case general_14#86
- Personality: O -0.5, C -0.9, E -0.9, A +0.6, N +0.0
- It is at the tavern (a lively drinking house: noisy, social, little privacy, the odd brawl)
- Just did there (oldest first): drink, chat
- It may now:
    A. chat
    B. drink
    C. brawl
- What does it do? Answer with the letter only.

### Case heldout_location_01#9
- Personality: O +0.1, C -0.7, E -0.8, A -0.8, N -0.5
- Recently visited (oldest first): chapel, library, chapel
- It may now go to:
    A. infirmary — the village healing house: sickbeds, herbs, quiet duty
    B. forest — wild ground beyond the walls: roaming, foraging, solitude
    C. tavern — a lively drinking house: noisy, social, little privacy, the odd brawl
- Where does it go? Answer with the letter only.

### Case heldout_location_01#17
- Personality: O -0.7, C +0.6, E -0.2, A +0.6, N +0.8
- Recently visited (oldest first): chapel, infirmary, chapel
- It may now go to:
    A. infirmary — the village healing house: sickbeds, herbs, quiet duty
    B. tavern — a lively drinking house: noisy, social, little privacy, the odd brawl
    C. arena — a fighting ground: combat bouts, training, spectacle, betting
- Where does it go? Answer with the letter only.

### Case heldout_location_01#22
- Personality: O +0.0, C +0.0, E +0.0, A -0.8, N +0.8
- It is at the infirmary (the village healing house: sickbeds, herbs, quiet duty)
- Just did there (oldest first): (only just arrived)
- It may now:
    A. tend
    B. assist
    C. vigil
- What does it do? Answer with the letter only.

### Case heldout_location_01#35
- Personality: O +0.6, C +0.4, E +0.1, A -0.2, N -0.8
- It is at the infirmary (the village healing house: sickbeds, herbs, quiet duty)
- Just did there (oldest first): assist, assist, vigil
- It may now:
    A. tend
    B. assist
    C. vigil
- What does it do? Answer with the letter only.

### Case personality_01#12
- Personality: O +0.9, C -0.6, E +0.9, A -0.9, N +0.2
- Recently visited (oldest first): (nothing recorded)
- It may now go to:
    A. tavern — a lively drinking house: noisy, social, little privacy, the odd brawl
    B. arena — a fighting ground: combat bouts, training, spectacle, betting
    C. library — quiet halls of books: solitary study, order, deep thought
- Where does it go? Answer with the letter only.

### Case personality_01#14
- Personality: O +0.8, C -0.7, E -0.7, A +0.8, N +0.0
- Recently visited (oldest first): market, chapel
- It may now go to:
    A. tavern — a lively drinking house: noisy, social, little privacy, the odd brawl
    B. forest — wild ground beyond the walls: roaming, foraging, solitude
    C. library — quiet halls of books: solitary study, order, deep thought
- Where does it go? Answer with the letter only.

### Case personality_01#30
- Personality: O +0.7, C -0.9, E +0.8, A -0.5, N -0.8
- It is at the forest (wild ground beyond the walls: roaming, foraging, solitude)
- Just did there (oldest first): explore, forage
- It may now:
    A. explore
    B. forage
    C. rest
- What does it do? Answer with the letter only.

### Case personality_01#35
- Personality: O +0.9, C -0.8, E -0.8, A -0.8, N -0.6
- It is at the market (a bustling trade square: haggling, browsing, deal-making)
- Just did there (oldest first): haggle, trade, browse
- It may now:
    A. haggle
    B. browse
    C. trade
- What does it do? Answer with the letter only.

### Case personality_01#36
- Personality: O +0.6, C -0.9, E +0.0, A +0.0, N +0.8
- It is at the forest (wild ground beyond the walls: roaming, foraging, solitude)
- Just did there (oldest first): (only just arrived)
- It may now:
    A. explore
    B. forage
    C. rest
- What does it do? Answer with the letter only.
