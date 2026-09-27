"""Ordinal strength: a relative request's strength picks among the offered options by rank.

People don't agree on what "way warmer" means in degrees, and our data sources didn't either
(the smart-home generator, the LLM worlds and the intensity ladder all used different
amounts), so the model had to memorize exact amounts. Instead, the harness offers a few
coarse, well-separated values in the asked direction, and the strength picks by rank among
them, with "close enough" counted as right (several right answers share the target):

  small  ("a bit warmer")        the smallest move offered
  plain  ("warmer")              the middle one(s); either of two
  big    ("way warmer")          the upper half
  max    ("as hot as it goes")   the most extreme move offered

With 71, 73, 76 offered from 70: "a bit" 71, "warmer" 73, "way warmer" 76, "all the way" 76.
Offered values are simple: fractions of the room left in that direction (about 15%, 35%,
60% or all of it), rounded to whole numbers, to 5s on wide ranges, or to 0.05 on 0..1.
"""
import math

FRACTIONS = [0.15, 0.35, 0.6, 1.0]
STRENGTHS = ("small", "plain", "big", "max")


def nice(v, lo, hi, whole):
    span = hi - lo
    if whole or span >= 10:
        step = 5 if span >= 50 else 1
    else:
        step = 0.05 * span
    v = round(v / step) * step
    v = min(hi, max(lo, v))
    return float(round(v)) if (whole or span >= 10) and step >= 1 else round(v, 2)


def offer(cur, lo, hi, up, rng, whole=False):
    """2-4 coarse values that move from cur toward hi (up) or lo, well apart, sorted by the
    size of the move. Fewer if the device can't move that far. [] at the limit."""
    room = (hi - cur) if up else (cur - lo)
    if room <= 1e-9:
        return []
    n = rng.choice([2, 3, 3, 4, 4])
    fracs = sorted(rng.sample(FRACTIONS, n))        # the end (1.0) is often left out: "max" must still work
    gap = 0.08 * (hi - lo)
    out = []
    for f in fracs:
        v = nice(cur + (1 if up else -1) * f * room, lo, hi, whole)
        if abs(v - cur) >= 0.04 * (hi - lo) and all(abs(v - o) >= gap for o in out):
            out.append(v)
    return sorted(out, key=lambda v: abs(v - cur))


IDEAL = {"small": 0.15, "plain": 0.35, "big": 0.6, "max": 1.0}      # share of the room left


def weights(moves, cur, lo, hi, up, strength):
    """Graded right answers {index: weight}: of the acceptable options (right()), the one
    nearest the strength's ideal amount weighs 1, the others 0.5; everything else 0."""
    ok = right(len(moves), strength)
    if not ok:
        return {}
    room = max((hi - cur) if up else (cur - lo), 1e-9)
    best = min(ok, key=lambda i: abs(abs(moves[i] - cur) / room - IDEAL[strength]))
    return {i: 1.0 if i == best else 0.5 for i in ok}


def right(n, strength):
    """Indices (0 = smallest move) that count as right for a strength, among n offered moves."""
    if n == 0:
        return []
    if strength == "small":
        return [0]
    if strength == "max":
        return [n - 1]
    if strength == "big":
        return list(range(math.ceil(n / 2), n)) if n > 1 else [0]
    return list(range(1, n - 1)) if n >= 3 else list(range(n))      # plain
