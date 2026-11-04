"""Right action, right direction: credit and scoring that don't hinge on the exact amount.

A request like "less jets" on a 0..3 pump at 1 has one way down (0), a full move. Graded
targets (dink/gen/ordinal.py) give an off-size move 0, the same as the wrong device, so the
model learns to prefer a moderate move on the wrong action over the only move on the right
one. Here:

  same_direction(s)      the options on the answer's action that move the same way from its
                         current value (meta["needed"] names the state field it sets)
  credit_direction(s)    training targets: those options get at least `floor` of the best
                         option's weight. Left alone: requests that say a number, and
                         requests that name a size ("a little", "a lot", "all the way":
                         classes ending in _small, _big, _max), where the amount is the point
  thin_options(s, rng)   augmentation: one action (right or distractor) keeps only some of
                         its options, so "offered once" or "lopsided" is no tell
"""
import copy
import re

from .numbers import extract


def _best(s):
    i = s.label
    return i if i < len(s.tools) else None


def current(s, name):
    """Current value of the state field the answer's action sets, if the scenario shows it."""
    for need in s.meta.get("needed", []):
        for f in s.state:
            if f.name == need and f.arg:
                return f.value
    return None


def _sign(x):
    return (x > 1e-9) - (x < -1e-9)


def same_direction(s):
    """Indices of options on the best option's action that move the same way from the current
    value. Empty when there is no such reading (no current value, a plain action, nothing)."""
    b = _best(s)
    if b is None or not s.tools[b].arg:
        return []
    best = s.tools[b]
    cur = current(s, best.name)
    if cur is None or _sign(best.value - cur) == 0:
        return []
    d = _sign(best.value - cur)
    return [i for i, t in enumerate(s.tools) if t.name == best.name and t.arg and _sign(t.value - cur) == d]


SIZED = re.compile(r"_(small|big|max)$")


def credit_direction(s, floor=0.1):
    """Same-direction options get at least floor x the best weight; renormalized."""
    if floor <= 0 or extract(s.prompt) or SIZED.search(s.meta.get("cls") or ""):
        return s
    idx = same_direction(s)
    top = max(s.target)
    if not idx or all(s.target[i] >= floor * top for i in idx):
        return s
    s = copy.deepcopy(s)
    for i in idx:
        s.target[i] = max(s.target[i], floor * top)
    z = sum(s.target)
    s.target = [t / z for t in s.target]
    return s


def right_direction(s, pick):
    """The pick is the best answer, or another option on its action in the same direction."""
    return s.target[pick] >= max(s.target) - 1e-6 or pick in same_direction(s)


def thin_options(s, rng):
    """Keep a random non-empty subset of one action's options (the best option always stays)."""
    by = {}
    for i, t in enumerate(s.tools):
        by.setdefault(t.name, []).append(i)
    multi = [n for n, ix in by.items() if len(ix) > 1]
    if not multi:
        return s
    idx = by[rng.choice(multi)]
    top = max(s.target)
    must = [i for i in idx if s.target[i] >= top - 1e-9]
    rest = [i for i in idx if i not in must]
    keep_n = rng.randint(0 if must else 1, len(rest))       # the action keeps at least one option
    drop = set(rng.sample(rest, len(rest) - keep_n))
    if not drop or len(s.tools) - len(drop) < 2:
        return s
    s = copy.deepcopy(s)
    keep = [i for i in range(len(s.tools)) if i not in drop]
    s.tools = [s.tools[i] for i in keep]
    target = [s.target[i] for i in keep] + [s.target[-1]]
    z = sum(target)                                          # > 0: the best option is kept
    s.target = [t / z for t in target]
    return s
