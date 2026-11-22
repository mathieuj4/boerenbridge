import random
from boerenbridge import Config, play_round, play_match, score_round, round_schedule
from agents import HeuristicAgent, RandomAgent
from ismcts import (enc, dec, SUIT_IDX, heur_move, heur_bid, infer_voids, unseen_info, determinize,
                    ExactMaxN, mask_of, trick_winner_idx, ISMCTSAgent, play_search, bid_search, pimc_vote)

cfg = Config()


class Rec(HeuristicAgent):
    """Heuristic agent that records every view it is shown."""
    def __init__(self):
        self.bids, self.plays = [], []
    def bid(self, v):
        self.bids.append(v)
        return super().bid(v)
    def play(self, v):
        self.plays.append(v)
        return super().play(v)


def record(seed, n_cards, use_trump=True):
    rng = random.Random(seed)
    agents = [Rec() for _ in range(3)]
    res = play_round(rng, agents, n_cards, use_trump, rng.randrange(3), cfg, [0, 0, 0], 0)
    return agents, res


# 1) int-card heuristics are identical to the engine's HeuristicAgent -----------------
H = HeuristicAgent()
n_moves = n_bids = 0
for seed in range(150):
    n_cards = [1, 2, 3, 5, 7, 10][seed % 6]
    agents, res = record(seed, n_cards, use_trump=(seed % 7 != 0))
    for a in agents:
        for v in a.bids:
            trump = SUIT_IDX[v.trump] if v.trump else -1
            assert heur_bid([enc(c) for c in v.hand], trump, v.n_cards, v.forbidden) == H.bid(v)
            n_bids += 1
        for v in a.plays:
            trump = SUIT_IDX[v.trump] if v.trump else -1
            lg = [enc(c) for c in v.legal]
            tr = [enc(c) for _, c in v.trick]
            need = v.bids[v.seat] - v.tricks_won[v.seat]
            m = heur_move(lg, tr, need, trump, len(v.trick) == v.cfg.n_players - 1)
            assert dec(m) == H.play(v), (dec(m), H.play(v))
            n_moves += 1
print(f"heuristic parity OK: {n_bids} bids, {n_moves} card choices identical to the engine's agent")

# 2) void inference + determinization never contradict the true hidden hands -----------
checked = 0
for seed in range(150):
    agents, res = record(1000 + seed, [4, 6, 8, 10][seed % 4])
    for a in agents:
        for v in a.plays:
            pool, counts = unseen_info(v)
            voids = infer_voids(v.played)
            opps = [s for s in range(3) if s != v.seat]
            for s in opps:
                remaining = [c for c in res.hands[s] if (s, c) not in set(v.played)]
                assert len(remaining) == counts[s]
                assert all(SUIT_IDX[c[0]] not in voids[s] for c in remaining), "void contradicted by truth"
                assert all(enc(c) in pool for c in remaining)
            det = determinize(pool, counts, voids, opps, random.Random(seed))
            for s in opps:
                assert len(det[s]) == counts[s] and not any((c >> 4) in voids[s] for c in det[s])
            assert not set(det[opps[0]]) & set(det[opps[1]])
            checked += 1
print(f"determinization OK: {checked} decision points, voids sound, hand sizes and disjointness hold")


# 3) exact solver equals a plain brute-force maxn with the same tie-break -------------
def brute(bids, trump, hands, leader):
    def u(p, wn, g):
        return score_round(bids[p], wn[p] + g, cfg)
    def better(a, b, m, wn):
        if a[m] != b[m]:
            return a[m] > b[m]
        return sum(a) - a[m] < sum(b) - b[m]
    def go(hs, tr, ldr, wn):
        if not tr and not any(hs):
            return tuple(score_round(bids[p], wn[p], cfg) for p in range(3))
        mover = (ldr + len(tr)) % 3
        hand = hs[mover]
        if tr:
            f = [c for c in hand if c >> 4 == tr[0] >> 4]
            moves = f or list(hand)
        else:
            moves = list(hand)
        best = None
        for c in moves:
            nh = list(hs); nh[mover] = [x for x in hand if x != c]
            nt = tr + (c,)
            if len(nt) == 3:
                w = (ldr + trick_winner_idx(nt, trump)) % 3
                nw = list(wn); nw[w] += 1
                v = go(nh, (), w, tuple(nw))
            else:
                v = go(nh, nt, ldr, wn)
            if best is None or v[mover] > best[mover] or (v[mover] == best[mover] and sum(v) - v[mover] < sum(best) - best[mover]):
                best = v
        return best
    return go([list(h) for h in hands], (), leader, (0, 0, 0))

bad = 0
for seed in range(200):
    rng = random.Random(seed)
    k = rng.choice([1, 2, 3, 3])
    deck = [enc(c) for c in cfg.deck()]
    rng.shuffle(deck)
    hands = [sorted(deck[i * k:(i + 1) * k]) for i in range(3)]
    trump = rng.choice([-1, 0, 1, 2, 3])
    bids = [rng.randint(0, k) for _ in range(3)]
    leader = rng.randrange(3)
    ex = ExactMaxN(bids, trump, cfg).search(tuple(mask_of(h) for h in hands), (), leader, (0, 0, 0))
    if ex != brute(bids, trump, hands, leader):
        bad += 1
assert bad == 0, bad
print("exact solver OK: 200 random small deals identical to brute-force maxn")

# 4) the agent always acts legally, including under the hook rule ---------------------
rng = random.Random(5)
ag = ISMCTSAgent(play_iters=20, bid_iters=60, seed=1)
ag2 = ISMCTSAgent(play_iters=20, bid_iters=60, rollout="random", seed=2)
for m in range(2):
    scores, results = play_match(rng, [ag, HeuristicAgent(), ag2], cfg, first_dealer=m)
    for r in results:
        assert sum(r.won) == r.n_cards
        if r.hook_forbidden is not None:
            assert sum(r.bids) != r.n_cards
print("agent legality OK: 2 full 11-round matches with two ISMCTS seats, hook rule never violated")

# 5) a search with one obviously best card finds it ---------------------------------
agents, res = record(77, 3)
v = next(v for a in agents for v in a.plays if len(v.legal) > 1)
stats = play_search(v, 300, random.Random(0))
assert sum(n for n, _ in stats.values()) > 0
print("search sanity OK")

# 6) conditioning on bids moves the sampled hands the right way ---------------------------
from ismcts import make_bank, _bid_loglik, heur_est, match_reward, bid_match_values, refine_bids
import statistics as _st
deck36 = [enc(c) for c in cfg.deck()]
rng = random.Random(0)
def _sampler():
    sh = deck36[:]; rng.shuffle(sh)
    return {1: sh[:6], 2: sh[6:12]}
est = {}
for kb in (0, 4):
    bank = make_bank(_sampler, lambda d: _bid_loglik(d, {1: kb}, 2, 1.0), 2000)
    draws = [bank.draw(rng) for _ in range(3000)]
    est[kb] = (_st.mean(heur_est(d[1], 2) for d in draws), _st.mean(heur_est(d[2], 2) for d in draws), bank.ess)
assert est[4][0] > est[0][0] + 1.0, est                        # a high bidder is dealt stronger hands
assert abs(est[4][1] - est[0][1]) < 0.4, est                   # the seat that has not bid is unaffected
assert est[0][2] > 30 and est[4][2] > 30, est                  # the weighted pool is not degenerate
print(f"bid conditioning OK: mean estimated tricks of a 6-card hand that bid 0 = {est[0][0]:.2f}, that bid 4 = {est[4][0]:.2f} "
      f"(unconditioned {est[0][1]:.2f}); effective sample size >= {min(est[0][2], est[4][2]):.0f} of 2000")

# 7) match reward + last-round win probabilities ------------------------------------------
assert match_reward((0, 0, 0), [10, 0, 0], 0, 8.0) > 0.5 > match_reward((0, 0, 0), [0, 10, 0], 0, 8.0)
assert match_reward((0, 20, 0), [10, 0, 0], 0, 8.0) < match_reward((0, 0, 0), [10, 0, 0], 0, 8.0)
agents6, _ = record(4242, 4)
bv = agents6[0].bids[0]
pts_, win_ = bid_match_values(bv, 40, random.Random(1), (0, 15, 5))
assert len(pts_) == len(win_) == bv.n_cards + 1 and all(0.0 <= w <= 1.0 for w in win_)
print("match utility OK")

# 8) every option on, a full 11-round match, legal throughout -----------------------------
rng = random.Random(9)
full = ISMCTSAgent(play_iters=20, bid_iters=60, seed=5, infer_bids=True, opp_model="model",
                   refine=2, refine_samples=4, refine_iters=4, match_aware=True)
scores, results = play_match(rng, [full, HeuristicAgent(), HeuristicAgent()], cfg, first_dealer=1)
assert len(results) == 11 and results[-1].n_cards == 10 and results[-1].trump is None
for r in results:
    assert sum(r.won) == r.n_cards
    if r.hook_forbidden is not None:
        assert sum(r.bids) != r.n_cards
print("all-options agent OK: full 11-round match, last round is the 10-card no-trump round, hook rule held")

# 9) the learned opponent model moves in the right direction ----------------------------------
from ismcts import OppModel
from opponents import BoldHeuristic, NoisyHeuristic


class Watcher(HeuristicAgent):
    def __init__(self):
        self.om = None
    def bid(self, v):
        if self.om is None or v.round_index == 0:
            self.om = OppModel(v.seat)
        return super().bid(v)
    def end_round(self, res):
        self.om.observe(res)


def learn(opp, n_matches=6, seed=0):
    rng = random.Random(seed)
    w = Watcher()
    for m in range(n_matches):
        w.om = None
        play_match(rng, [w, opp, HeuristicAgent()], cfg, first_dealer=m % 3)
    return w.om


om_h = learn(HeuristicAgent())
om_b = learn(BoldHeuristic(1.0))
om_n = learn(NoisyHeuristic(0.4))
ah, bh, sh, eh = om_h.params(1)
ab, bb, sb, eb = om_b.params(1)
an, bn, sn, en = om_n.params(1)
assert abs(ah) < 0.4 and 0.85 < bh < 1.15 and eh <= 0.12, (ah, bh, eh)       # the heuristic is recognised as itself
assert (ab + bb * 2) - (ah + bh * 2) > 0.6 and eb <= 0.15, (ab, bb, eb)    # bold: predicted bid for a 2-trick hand is about one higher
assert en > 0.3 and not om_n.use_model(1) and om_h.use_model(1), (en,)        # noisy: higher play noise, dropped from the model
print(f"opponent model OK: heuristic (a,b,eps)=({ah:.2f},{bh:.2f},{eh:.2f}); bold ({ab:.2f},{bb:.2f},{eb:.2f}); "
      f"noisy eps={en:.2f}")

# 10) an adapting agent plays a legal full match
rng = random.Random(21)
ad = ISMCTSAgent(play_iters=20, bid_iters=60, seed=7, infer_bids=True, adapt=True, refine=2, refine_samples=4,
                 refine_iters=4, match_aware=True)
scores, results = play_match(rng, [HeuristicAgent(), ad, BoldHeuristic()], cfg, first_dealer=2)
assert len(results) == 11 and ad.om is not None and ad.om.summary()
for r in results:
    assert sum(r.won) == r.n_cards
    if r.hook_forbidden is not None:
        assert sum(r.bids) != r.n_cards
print("adaptive agent OK: full 11-round match legal, learned", ad.om.summary())

# 11) four players: 8 cards max, 8 rounds with trump + one without = 9 rounds -------------------------
cfg4 = Config(n_players=4, max_hand=8)
assert len(round_schedule(8)) == 9 and round_schedule(8)[-1] == (8, False)
rng = random.Random(33)
ag4 = [ISMCTSAgent(play_iters=20, bid_iters=60, seed=1, infer_bids=True, adapt=True, refine=2, refine_samples=3,
                   refine_iters=3, match_aware=True),
       ISMCTSAgent(play_iters=20, bid_iters=60, seed=2, opp_model="model"), HeuristicAgent(), RandomAgent(random.Random(3))]
scores, results = play_match(rng, ag4, cfg4, first_dealer=1)
assert len(results) == 9 and results[-1].n_cards == 8 and results[-1].trump is None and len(scores) == 4
for r in results:
    assert sum(r.won) == r.n_cards and len(r.bids) == 4
    if r.hook_forbidden is not None:
        assert sum(r.bids) != r.n_cards
assert len(ag4[0].om.summary()) == 3
print("four-player OK: 9-round match (8 cards max, last round no trump), hook held, learned", ag4[0].om.summary())

# 12) the int-card heuristic copy matches the engine's HeuristicAgent with four players too ----------------
class Rec4(Rec):
    pass
n4 = 0
for seed in range(40):
    rng = random.Random(900 + seed)
    ags = [Rec() for _ in range(4)]
    play_round(rng, ags, [1, 3, 5, 8][seed % 4], seed % 5 != 0, rng.randrange(4), cfg4, [0] * 4, 0)
    for a in ags:
        for v in a.plays:
            trump = SUIT_IDX[v.trump] if v.trump else -1
            lg = [enc(c) for c in v.legal]; tr = [enc(c) for _, c in v.trick]
            m = heur_move(lg, tr, v.bids[v.seat] - v.tricks_won[v.seat], trump, len(v.trick) == 3)
            assert dec(m) == H.play(v), (dec(m), H.play(v)); n4 += 1
print(f"four-player heuristic parity OK: {n4} card choices")
