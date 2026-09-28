"""A device harness: turns a gadget (its state and actions) and a request into Dink's options,
without knowing the answer, and runs Dink in two passes.

Dink picks among at most 8 offered options, and a gadget can have more actions and values
than that. The harness decides what to offer, the way firmware on a device would:

  pass 1  which action, and which way: every action once; a valued action gets one step each
          way from its current value (about a third of the room), the range ends when there's
          no current value, and any number the request says that fits its range. Over 8
          options, the actions whose names best match the request (Dink's own text encoder)
          are kept.
  pass 2  how much: if pass 1 picked a valued action, Dink runs again on that action alone,
          offering a ladder in both directions (about 15 / 35 / 60 / 100% of the room left)
          plus any number said. The best of those is the answer.

Gadget format (the LLM worlds' format): state = [{name, value, lo, hi}], actions =
[{name, lo, hi, sets}] with lo = hi = None for actions without a value.
"""
import torch
import torch.nn.functional as F

from .data.collate import collate
from .data.numbers import extract
from .data.scenario import MAX_SLOTS, Field, Scenario
from .gen.ordinal import FRACTIONS, nice
from .model.text import to_bytes


TIE = 0.15          # pass-1 top two, different actions, closer than this: unsure


def _current(gadget, a):
    st = {f["name"]: f["value"] for f in gadget["state"]}
    return st.get(a.get("sets")) if a.get("sets") else None


def _whole(gadget, a, cur):
    return all(float(x).is_integer() for x in (a["lo"], a["hi"])) and (cur is None or float(cur).is_integer())


def said_numbers(prompt, a):
    """Numbers in the request that fit this action's range (levels become values)."""
    out = []
    for v, kind in extract(prompt):
        if kind == "level":
            v = a["lo"] + v * (a["hi"] - a["lo"])
        if a["lo"] <= v <= a["hi"]:
            out.append(round(v, 4))
    return out


def ladder(gadget, a, prompt, fractions=FRACTIONS):
    """Candidate values for one valued action: steps both ways from the current value, plus
    numbers said. Without a current value: evenly across the range."""
    cur = _current(gadget, a)
    whole = _whole(gadget, a, cur)
    said = said_numbers(prompt, a)
    vals = list(said)                            # numbers the user said come first: never trimmed
    if cur is None:
        vals = [nice(a["lo"] + f * (a["hi"] - a["lo"]), a["lo"], a["hi"], whole) for f in (0, 0.25, 0.5, 0.75, 1)]
    else:
        for up in (True, False):
            room = (a["hi"] - cur) if up else (cur - a["lo"])
            for f in fractions:
                if room > 1e-9:
                    vals.append(nice(cur + (1 if up else -1) * f * room, a["lo"], a["hi"], whole))
    gap = 0.04 * (a["hi"] - a["lo"])
    out = []
    for v in vals:
        if v in said or ((cur is None or abs(v - cur) > 1e-9) and all(abs(v - o) >= gap for o in out)):
            if v not in out:
                out.append(v)
    return out


def relevance(model, prompt, names):
    """Cosine between the request and each action name, by Dink's own text encoder."""
    with torch.no_grad():
        dev = next(model.parameters()).device
        p = model.text(to_bytes([prompt]).to(dev))[:, 0]
        n = model.text(to_bytes(names, 40).to(dev))[:, 0]
        return F.cosine_similarity(p, n).tolist()


def score(model, state, prompt, tools):
    s = Scenario(state, prompt, tools, [0.0] * len(tools) + [1.0])
    x = collate([s])
    dev = next(model.parameters()).device
    with torch.no_grad():
        p = model({k: v.to(dev) for k, v in x.items()}).softmax(-1)[0].cpu()
    return p[:len(tools)].tolist() + [p[MAX_SLOTS].item()]


def fields(gadget):
    return [Field(f["name"], f["value"], f["lo"], f["hi"]) for f in gadget["state"]][:MAX_SLOTS]


def pass1_tools(model, gadget, prompt):
    groups = []                                   # (action, [Field]) per action
    for a in gadget["actions"]:
        if a["lo"] is None:
            groups.append((a, [Field(a["name"], 0, 0, 0, arg=False)]))
            continue
        cur = _current(gadget, a)
        whole = _whole(gadget, a, cur)
        vals = said_numbers(prompt, a)
        if cur is None:
            vals += [a["lo"], a["hi"]]
        else:
            for up in (True, False):
                room = (a["hi"] - cur) if up else (cur - a["lo"])
                if room > 1e-9:
                    vals.append(nice(cur + (1 if up else -1) * 0.35 * room, a["lo"], a["hi"], whole))
        vals = list(dict.fromkeys(v for v in vals if cur is None or abs(v - cur) > 1e-9))
        groups.append((a, [Field(a["name"], v, a["lo"], a["hi"]) for v in vals]))
    if sum(len(g) for _, g in groups) > MAX_SLOTS:  # keep the actions whose names match the request best
        rel = relevance(model, prompt, [a["name"].replace("_", " ") for a, _ in groups])
        order = sorted(range(len(groups)), key=lambda i: -rel[i])
        kept, n = [], 0
        for i in order:
            if n + len(groups[i][1]) <= MAX_SLOTS:
                kept.append(groups[i])
                n += len(groups[i][1])
        groups = kept
    return [t for _, g in groups for t in g]


def decide(model, gadget, prompt):
    """-> dict(pass1=[(Field, p)], nothing1, pick=Field|None, pass2=[(Field, p)] or None, p)."""
    state = fields(gadget)
    tools = pass1_tools(model, gadget, prompt)
    probs = score(model, state, prompt, tools)
    best = max(range(len(probs)), key=probs.__getitem__)
    out = {"pass1": list(zip(tools, probs[:-1])), "nothing1": probs[-1], "pass2": None}
    if best == len(tools):
        out.update(pick=None, p=probs[-1])
        return out
    pick = tools[best]
    out.update(pick=pick, p=probs[best], unsure=False)
    # Near-tie between two different actions: flag it (a device could ask "the pH or the
    # sanitizer?") and break the tie by how well each action's name matches the request.
    order = sorted(range(len(tools)), key=lambda i: -probs[i])
    if len(order) > 1 and order[1] < len(tools) and tools[order[1]].name != pick.name \
            and probs[order[0]] - probs[order[1]] < TIE:
        rel = relevance(model, prompt, [tools[order[0]].name.replace("_", " "), tools[order[1]].name.replace("_", " ")])
        if rel[1] > rel[0]:
            best = order[1]
            pick = tools[best]
        out.update(pick=pick, p=probs[best], unsure=True)
    a = next((x for x in gadget["actions"] if x["name"] == pick.name), None)
    if a is None or a["lo"] is None or pick.value in said_numbers(prompt, a):
        return out                               # no amount to refine: a plain action, or the number said
    vals = ladder(gadget, a, prompt)[:MAX_SLOTS]
    if len(vals) > 1:
        t2 = [Field(a["name"], v, a["lo"], a["hi"]) for v in vals]
        p2 = score(model, state, prompt, t2)
        b2 = max(range(len(t2)), key=p2.__getitem__)      # the amount; "nothing" was already ruled out
        out.update(pass2=list(zip(t2, p2[:-1])), pick=t2[b2], p=p2[b2])
    return out
