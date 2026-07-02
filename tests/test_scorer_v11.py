"""v1.1 scorer behaviour (spec §5: the six required checks)."""
import numpy as np

from npc_policy.config import ScorerConfig
from npc_policy.representation import Option, Personality, RecentBuffer
from npc_policy.scorer import HandAuthoredScorer, softmax
from npc_policy.schema import trait_index
from npc_policy.weights import default_C_location, default_W_location

LOCS = [
    Option.location("tavern", social=0.9, stimulation=0.8),
    Option.location("library", cognitive=0.8, structure=0.7, privacy=0.6),
    Option.location("forest", exploration=0.9, privacy=0.5, risk=0.3),
]


def entropy(p: np.ndarray) -> float:
    return float(-(p * np.log(p + 1e-12)).sum())


def test_1_bilinear_regression_matches_v1_base():
    # base_form="bilinear", empty buffer, N=0  =>  P_rule == softmax(p^T W o / tau_0)
    cfg = ScorerConfig(base_form="bilinear")
    scorer = HandAuthoredScorer(config=cfg)
    p = Personality.from_traits(openness=0.5, extraversion=0.5)  # N = 0
    dist = scorer.distribution(p, LOCS, level="location")
    base = np.stack([o.features for o in LOCS]) @ (p.vector @ default_W_location())
    assert np.allclose(dist, softmax(base, cfg.location.tau_0))


def test_2_order_equivariance():
    scorer = HandAuthoredScorer()
    p = Personality.from_traits(openness=0.4, neuroticism=0.3)
    buf = RecentBuffer(maxlen=3)
    buf.push(LOCS[0])
    d = scorer.distribution(p, LOCS, buffer=buf, level="location")
    perm = [2, 0, 1]
    d_perm = scorer.distribution(p, [LOCS[i] for i in perm], buffer=buf, level="location")
    assert np.allclose(d_perm, d[perm])


def test_3_empty_buffer_no_memory_adjustment():
    scorer = HandAuthoredScorer()
    p = Personality.from_traits(openness=0.8, conscientiousness=0.6, neuroticism=0.5)
    tr = scorer.trace(p, LOCS, level="location")
    assert np.allclose(tr.relations.rep, 0.0) and np.allclose(tr.relations.sim, 0.0)
    expected = softmax(tr.base / scorer.config.location.tau_0, tr.T_N)
    assert np.allclose(tr.P_rule, expected)


def test_4_bidirectional_N_entropy():
    # zero N's base-table row so only the temperature channel varies with N
    C = default_C_location().copy()
    C[trait_index("neuroticism"), :] = 0.0
    scorer = HandAuthoredScorer(C_location=C)

    def h(n: float) -> float:
        p = Personality.from_traits(extraversion=0.5, neuroticism=n)
        return entropy(scorer.distribution(p, LOCS, level="location"))

    assert h(-1.0) < h(0.0) < h(+1.0)


def test_5_ideal_point_mid_preference():
    # mid-E personality: mu_social = 0.5 + 0.45 * 0.3 = 0.635 -> middle option wins
    cands = [Option.location(f"s{v}", social=v) for v in (0.2, 0.65, 0.9)]
    p = Personality.from_traits(extraversion=0.3)
    dist = HandAuthoredScorer().distribution(p, cands, level="location")
    assert int(np.argmax(dist)) == 1


def test_6_gamma_direction_routine_vs_novelty():
    scorer = HandAuthoredScorer()
    market = Option.location("market", social=0.7, stimulation=0.7, exploration=0.5)
    cands = [market, LOCS[1]]                 # market vs library
    buf = RecentBuffer(maxlen=3)
    buf.push(LOCS[0])                          # just visited the tavern (similar to market)

    def p_market(p: Personality) -> tuple[float, float]:
        without = scorer.distribution(p, cands, level="location")[0]
        with_buf = scorer.distribution(p, cands, buffer=buf, level="location")[0]
        return without, with_buf

    before, after = p_market(Personality.from_traits(conscientiousness=1.0))
    assert after > before                      # high C: familiarity attracts
    before, after = p_market(Personality.from_traits(openness=1.0))
    assert after < before                      # high O: familiarity repels
