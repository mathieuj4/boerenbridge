import random
from boerenbridge import *
from agents import RandomAgent, HeuristicAgent

cfg = Config()
cfg.validate()
assert len(cfg.deck()) == 36 and min(r for _, r in cfg.deck()) == 6
assert len(round_schedule(8)) == 9 and len(round_schedule(10)) == 11
assert [n for n, _ in round_schedule(3)] == [1, 2, 3, 3]
assert [t for _, t in round_schedule(3)] == [True, True, True, False]

# scoring
assert score_round(3, 3, cfg) == 19 and score_round(0, 0, cfg) == 10
assert score_round(3, 5, cfg) == -6 and score_round(0, 4, cfg) == -12

# trick resolution
S, H, C = "S", "H", "C"
assert trick_winner([(0, (S, 10)), (1, (S, 12)), (2, (S, 7))], None) == 1
assert trick_winner([(0, (S, 10)), (1, (H, 6)), (2, (S, 14))], H) == 1      # trump beats ace
assert trick_winner([(0, (S, 10)), (1, (C, 14)), (2, (S, 7))], H) == 0      # off-suit non-trump loses
assert trick_winner([(0, (H, 8)), (1, (H, 9)), (2, (S, 14))], H) == 1
assert legal_cards([(S, 9), (H, 7)], [(1, (S, 6))]) == [(S, 9)]
assert len(legal_cards([(H, 9), (H, 7)], [(1, (S, 6))])) == 2

# full-match invariants across many random and heuristic matches
rng = random.Random(1)
hook_bound = 0
for m in range(300):
    agents = [RandomAgent(rng), HeuristicAgent(), RandomAgent(rng)] if m % 2 else [HeuristicAgent()] * 3
    scores, results = play_match(rng, agents, cfg, first_dealer=m % 3)
    assert len(results) == 11
    for r in results:
        assert sum(r.won) == r.n_cards
        if r.hook_forbidden is not None:
            hook_bound += 1
            assert sum(r.bids) != r.n_cards          # hook rule never violated
        elif r.n_cards >= cfg.hook_min_cards:
            pass                                      # forbidden value fell outside 0..n
        assert all(len(h) == r.n_cards for h in r.hands)
        assert r.points == [score_round(r.bids[s], r.won[s], cfg) for s in range(3)]
    assert sum(scores) == sum(sum(r.points) for r in results)
print(f"engine OK: 300 matches, hook rule held in all {hook_bound} bound rounds")
