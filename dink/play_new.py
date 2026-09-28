"""Play against a brand-new gadget: the LLM invents one, Dink answers through a harness.

  uv run python -m dink.play_new "an oil refinery"          # seed idea (optional)
  uv run python -m dink.play_new                            # a random area
  uv run python -m dink.play_new --load data/llm/play_new/<file>.json   # replay a saved gadget, no LLM

The LLM stages are the world collector's own (dink/llm/worlds.py): a setting from the idea,
then the gadget's state, actions and 12 labeled requests, then an independent second labeling
with rewordings and flips. Dink never sees the answers: dink/harness.py builds its options
from the gadget's actions and the request alone, in two passes (which action and which way,
then how much). Each request is scored against the LLM's label:

  RIGHT  same action, and the same amount (or "nothing relevant" on both sides)
  CLOSE  same action and direction, an amount within one strength step (a bit / plain / a lot
         / all the way), or a number the request didn't say
  WRONG  anything else

Rows where the two LLM labelings disagree are marked "?": those are questions about the
generation prompt, not about Dink. After the table, type your own requests (-i is the
default; --no-interactive skips it). Gadgets are saved to data/llm/play_new/.
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

from .harness import decide
from .play import load_model

SAVE = Path("data/llm/play_new")


def level(frac):
    return 0 if frac < 0.25 else 1 if frac < 0.5 else 2 if frac < 0.85 else 3


# the same "close enough" as training (dink/gen/ordinal.py): strength -> acceptable pick levels
ACCEPT = {"small": {0, 1}, "plain": {0, 1, 2}, "big": {1, 2, 3}, "max": {2, 3}}
LEVEL_OF = {"small": 0, "plain": 1, "big": 2, "max": 3}


def at_limit(g, u):
    """The gadget as this request sees it: a limit request's field sits at its lo / hi."""
    from .gen.worlds import state_for
    if not u.get("limit"):
        return g
    st = state_for(g, u)
    return dict(g, state=[dict(f, value=st[f["name"]]["value"]) for f in g["state"]])


def verdict(gadget, label_action, label_value, res, strength=None):
    pick = res["pick"]
    if label_action == "none":
        return "RIGHT" if pick is None else "WRONG"
    if pick is None or pick.name != label_action:
        return "WRONG"
    a = next(x for x in gadget["actions"] if x["name"] == label_action)
    if a["lo"] is None or label_value is None:
        return "RIGHT"
    if abs(pick.value - label_value) < 1e-6:
        return "RIGHT"
    cur = {f["name"]: f["value"] for f in gadget["state"]}.get(a.get("sets"))
    if cur is None:
        return "CLOSE" if abs(pick.value - label_value) <= 0.15 * (a["hi"] - a["lo"]) else "WRONG"
    if (pick.value - cur) * (label_value - cur) <= 0:
        return "WRONG"
    up = label_value > cur
    room = max((a["hi"] - cur) if up else (cur - a["lo"]), 1e-9)
    got = level(abs(pick.value - cur) / room)
    if strength:                                   # a movement request: judge by strength, not the LLM's number
        return "RIGHT" if got == LEVEL_OF[strength] else "CLOSE" if got in ACCEPT[strength] else "WRONG"
    return "CLOSE" if abs(got - level(abs(label_value - cur) / room)) <= 1 else "WRONG"


def show_gadget(g):
    print(f"\n  {g['setting']} ({g['area']})\n  {g['about']}")
    print("  state  : " + "  ".join(f"{f['name']}={f['value']:g} [{f['lo']:g}..{f['hi']:g}]" for f in g["state"]))
    def rng(a):
        return "" if a["lo"] is None else f"{a['lo']:g}..{a['hi']:g}"
    print("  actions: " + "  ".join(f"{a['name']}({rng(a)})" for a in g["actions"]))


def fmt_label(action, value, g=None):
    if action == "none":
        return "(nothing)"
    a = next((x for x in (g or {}).get("actions", []) if x["name"] == action), None)
    if a is not None and a["lo"] is None:
        return f"{action}()"
    return f"{action}{'' if value is None else '=' + format(value, 'g')}"


def generate(idea, seed):
    from .llm.openrouter import Client
    from .llm.worlds import AREAS, SETTINGS_SCHEMA, SETTINGS_SYSTEM, SETTINGS_USER, ask, build
    c = Client.parasail(max_requests=12, max_spend_per_day=10.0)
    c.reasoning = "off"
    area = idea or random.Random(seed).choice(AREAS)
    got = ask(c, SETTINGS_USER.format(area=area, n=3, exclude="(none yet)"), SETTINGS_SYSTEM,
              max_tokens=2000, temperature=1.0, response_format=SETTINGS_SCHEMA)
    settings = [s for s in got.get("settings") or [] if isinstance(s, dict) and s.get("name")]
    if not settings:
        raise SystemExit("the LLM returned no setting; try again or rephrase the idea")
    setting = random.Random(seed).choice(settings)
    rec = build(c, setting, area, c.model)
    if not rec or "failed" in rec:
        raise SystemExit(f"the LLM's gadget failed the checks: {rec.get('failed') if rec else rec}")
    print(f"  ({c.summary()})")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("idea", nargs="*", help='a seed idea, e.g. "an oil refinery"')
    ap.add_argument("--load", default=None, help="a saved gadget (skips the LLM)")
    ap.add_argument("--run", default="runs/dink/all_v13")
    ap.add_argument("--all", action="store_true", help="also score the rewordings and flips, not just the 12 requests")
    ap.add_argument("--no-interactive", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    model = load_model(a.run, "cpu")
    if a.load:
        g = json.loads(Path(a.load).read_text())
    else:
        idea = " ".join(a.idea).strip()
        print(f"  asking the LLM for a gadget{' about ' + repr(idea) if idea else ''} ...")
        g = generate(idea, a.seed if a.seed is not None else int(time.time()))
        SAVE.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "_", g["setting"].lower()).strip("_")[:50]
        path = SAVE / f"{time.strftime('%Y%m%d-%H%M%S')}_{slug}.json"
        path.write_text(json.dumps(g, indent=1))
        print(f"  saved {path}")
    show_gadget(g)

    rows = [u for u in g["utterances"] if a.all or u["source"] == "request"]
    tally = {"RIGHT": 0, "CLOSE": 0, "WRONG": 0}
    print(f"\n  {'kind':14s} {'request':44s} {'LLM answer':30s} {'Dink (harness, 2 passes)':34s} verdict")
    from .gen.worlds import amount_strength
    for u in rows:
        gu = at_limit(g, u)
        res = decide(model, gu, u["say"])
        strength = amount_strength(gu, u)[0] if u["action"] != "none" else None
        v = verdict(gu, u["action"], u.get("value"), res, strength)
        pick = "(nothing)" if res["pick"] is None else f"{res['pick'].name}={res['pick'].value:g}" \
            if res["pick"].arg else f"{res['pick'].name}()"
        if res.get("unsure"):
            pick = "~" + pick                      # a near-tie in pass 1, broken by name match
        doubt = "" if u.get("agree", True) else " ?"
        rep = u.get("replay")
        second = f"  [2nd: {fmt_label(rep['action'], rep.get('value'))}]" if doubt and rep else ""
        tally[v] += 1
        print(f"  {u['kind'][:14]:14s} {u['say'][:44]:44s} {fmt_label(u['action'], u.get('value'), g)[:30]:30s} "
              f"{pick[:28]:28s} {res['p']:5.0%} {v}{doubt}{second}")
    n = len(rows)
    print(f"\n  {tally['RIGHT']}/{n} right, {tally['CLOSE']} close, {tally['WRONG']} wrong"
          f"   (? = the two LLM labelings disagree: a question for the prompt, not for Dink;"
          f" ~ = a near-tie, a device would ask)")
    if a.no_interactive:
        return
    print("\n  your own requests (empty line or :quit to stop; :why shows both passes)")
    last = None
    while True:
        try:
            line = input("dink> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line or line in (":q", ":quit"):
            return
        if line == ":why" and last:
            for name, key in (("pass 1", "pass1"), ("pass 2", "pass2")):
                if last[key]:
                    print(f"  {name}:")
                    for t, p in sorted(last[key], key=lambda x: -x[1])[:8]:
                        print(f"     {p:6.1%}  {t.name}{'=' + format(t.value, 'g') if t.arg else '()'}")
            print(f"  pass 1 'nothing relevant': {last['nothing1']:.1%}")
            continue
        last = decide(model, g, line)
        pick = last["pick"]
        print(f"  -> {'(nothing relevant)' if pick is None else pick.name + ('=' + format(pick.value, 'g') if pick.arg else '()')}"
              f"   {last['p']:.0%}")


if __name__ == "__main__":
    main()
