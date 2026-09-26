"""Stage 2: answer-preserving changes, applied each time a scenario is drawn for training.

Every function takes a Scenario and a random.Random and returns a new Scenario whose right
answer is unchanged (tool order changes carry the target along). They are generic, not
smart-home specific, so any future scenario generator can reuse them.
"""
import copy
import random

from ..data.scenario import PROMPT_BYTES, Scenario
from .names import parse

FILLERS = ["uh", "um", "ok", "hey", "so", "like", "yeah", "please", "alright"]
ASR_SWAPS = {  # plausible speech-recognition confusions
    "too": "to", "to": "too", "four": "for", "for": "four", "two": "to", "right": "write",
    "it's": "its", "i'm": "im", "you": "u", "the": "a", "a": "the", "off": "of", "in": "and",
    "lights": "light", "light": "lights", "fan": "fam", "heat": "he", "lock": "look",
}


def _clone(s: Scenario) -> Scenario:
    return copy.deepcopy(s)


def _fits(prompt: str) -> bool:
    return 0 < len(prompt.encode()) <= PROMPT_BYTES


def permute_tools(s: Scenario, rng: random.Random) -> Scenario:
    s = _clone(s)
    order = list(range(len(s.tools)))
    rng.shuffle(order)
    s.tools = [s.tools[i] for i in order]
    s.target = [s.target[i] for i in order] + [s.target[-1]]
    return s


def permute_state(s: Scenario, rng: random.Random) -> Scenario:
    s = _clone(s)
    rng.shuffle(s.state)
    return s


def drop_state(s: Scenario, rng: random.Random, p: float = 0.3) -> Scenario:
    """Remove state fields the answer does not depend on (meta["needed"] and occupancy stay)."""
    s = _clone(s)
    keep = set(s.meta.get("needed", []))
    s.state = [f for f in s.state if f.name in keep or parse(f.name)[1] == "occ" or rng.random() > p]
    return s


def reorder_chunks(s: Scenario, rng: random.Random) -> Scenario:
    """Move the movable prompt pieces: "in the kitchen, make it warmer", "make it warmer please"."""
    chunks = s.meta.get("chunks")
    if not chunks or len(chunks) < 2:
        return s
    core = [c for c in chunks if c[1] == "core"]
    movable = [c for c in chunks if c[1] != "core"]
    parts = core[:]
    for c in movable:
        pos = rng.choice([0, len(parts)])
        text = c[0] + ("," if pos == 0 and c[1] == "loc" else "")
        parts.insert(pos, [text, c[1]])
    prompt = " ".join(t for t, _ in parts)
    if not _fits(prompt):
        return s
    s = _clone(s)
    s.prompt = prompt
    return s


def fillers(s: Scenario, rng: random.Random) -> Scenario:
    words = s.prompt.split()
    words.insert(rng.randint(0, len(words)), rng.choice(FILLERS))
    prompt = " ".join(words)
    if not _fits(prompt):
        return s
    s = _clone(s)
    s.prompt = prompt
    return s


def typos(s: Scenario, rng: random.Random, rate: float = 0.03) -> Scenario:
    """Swap, drop or double characters at `rate` per character."""
    out = list(s.prompt)
    i = 0
    while i < len(out):
        if out[i] != " " and rng.random() < rate:
            op = rng.randint(0, 2)
            if op == 0 and i + 1 < len(out):
                out[i], out[i + 1] = out[i + 1], out[i]
            elif op == 1 and len(out) > 3:
                del out[i]
                continue
            else:
                out.insert(i, out[i])
                i += 1
        i += 1
    prompt = "".join(out).strip()
    if not _fits(prompt):
        return s
    s = _clone(s)
    s.prompt = prompt
    return s


def case_space(s: Scenario, rng: random.Random) -> Scenario:
    r = rng.random()
    prompt = s.prompt
    if r < 0.2:
        prompt = prompt.capitalize()
    elif r < 0.3:
        prompt = prompt.upper()
    if rng.random() < 0.2:
        prompt = prompt + rng.choice([".", "!", "?", "..."])
    if rng.random() < 0.1:
        prompt = prompt.replace(" ", "  ", 1)
    if not _fits(prompt):
        return s
    s = _clone(s)
    s.prompt = prompt
    return s


def asr_noise(s: Scenario, rng: random.Random, rate: float = 0.15) -> Scenario:
    """What a speech recognizer does: confusable words, dropped small words, no punctuation."""
    words = []
    for w in s.prompt.lower().replace(",", "").split():
        r = rng.random()
        if r < rate and w in ASR_SWAPS:
            words.append(ASR_SWAPS[w])
        elif r < rate * 0.5 and w in {"the", "a", "please", "it"}:
            continue
        else:
            words.append(w)
    prompt = " ".join(words)
    if not _fits(prompt):
        return s
    s = _clone(s)
    s.prompt = prompt
    return s


# (function, probability) -- the default training recipe
RECIPE = [
    (permute_tools, 1.0),
    (permute_state, 1.0),
    (drop_state, 0.3),
    (reorder_chunks, 0.3),
    (fillers, 0.15),
    (case_space, 0.3),
    (typos, 0.2),
    (asr_noise, 0.2),
]


def augment(s: Scenario, rng: random.Random, recipe=RECIPE) -> Scenario:
    for fn, p in recipe:
        if rng.random() < p:
            s = fn(s, rng)
    return s.check()
