"""Tests for the learned bots (tuned heuristic, bid model, planner, value search)."""
import random, numpy as np
from boerenbridge import Config, play_match
from agents import HeuristicAgent
from agents2 import VersionA, VersionB, TunedAgent, BidNetAgent
from tuned import Tuned, theta0
from fast import deal, base_bidder, base_move
from planner import future_probs, pmf_bernoulli

# 1. theta0 reproduces the stock heuristic exactly
T = Tuned(theta0()); rng = random.Random(1)
for _ in range(2000):
    nc = rng.randint(1, 10); hands, tc, tr = deal(rng, nc, rng.random() < .9)
    f = rng.choice([None, rng.randint(0, nc)])
    assert T.bid(hands[0], tr, nc, f) == base_bidder(hands[0], tr, nc, f, [])
print("ok tuned==heuristic at theta0")

# 2. probabilities / pmf sanity
U = [c for c in range(0, 64) if (c & 15) in range(6, 15) and (c >> 4) < 4]
ps = future_probs([3 * 16 + 14, 0 * 16 + 6], U, 5, 3, 2)
assert 0 <= ps[1] <= ps[0] <= 1, ps
d = pmf_bernoulli(ps); assert abs(sum(d) - 1) < 1e-9
assert future_probs([3 * 16 + 14], [], 0, 3, 2) == [1.0]
print("ok probabilities")

# 3. full matches (engine validates every bid/card, incl. the hook rule) for every learned bot
cfg = Config()
for cls in (TunedAgent, BidNetAgent, VersionA, VersionB):
    for m in range(5):
        ags = [cls(seed=s) if cls in (VersionA, VersionB) else cls(s) for s in range(3)]
        sc, res = play_match(random.Random(m), ags, cfg, first_dealer=m % 3)
        assert len(res) == 11 and sum(r.n_cards for r in res) == 65
print("ok full matches")

# 4. determinism for a fixed seed
def run(cls):
    ags = [cls(seed=s) for s in range(3)]
    return play_match(random.Random(7), ags, cfg, first_dealer=0)[0]
assert run(VersionA) == run(VersionA) and run(VersionB) == run(VersionB)
print("ok determinism")

# 5. mixed tables with the stock heuristic and with the old search bot still run
from ismcts import ISMCTSAgent
sc, _ = play_match(random.Random(3), [VersionA(seed=0), VersionB(seed=1), HeuristicAgent()], cfg)
sc, _ = play_match(random.Random(3), [VersionB(seed=0), ISMCTSAgent(play_iters=20, bid_iters=60, seed=1), HeuristicAgent()], cfg)
print("ok mixed tables")

# 6. 4 players are refused cleanly
try:
    play_match(random.Random(0), [VersionA(seed=s) for s in range(4)], Config(n_players=4, max_hand=8))
    raise SystemExit("4p should be refused")
except ValueError as e:
    print("ok 4-player refused:", e)
print("ALL NEW TESTS PASSED")
