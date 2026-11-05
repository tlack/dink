"""Direction probes: many small made-up gadgets, one request each, run through the harness the
way a device would (dink/harness.py). Cases: dink/evals/direction_cases.jsonl, written by the
LLM (dink/llm/probes.py) and reviewed by hand.

Scored for "most correct in the right direction":

  dir   the right action (or one setting the same field), moved the right way (or nothing, for questions; or nothing when the
        value is already at the end it was asked to move toward)
  full  dir, and the amount fits, judged by rank among the options offered on that action in
        that direction, never by exact value:
          small  not the farthest option that way, unless it's the only one
                 (on 1..3 at 3, "a little lower" is 2, not 1)
          big    not the nearest option that way, unless it's the only one
          end    the range end
          number the value said
          any    any option that way
  lone  the raw model, pass 1 options with the right action offered once (its other-way
        option dropped): an action offered once must still be picked

Misses by kind: action (wrong device), way (right device, wrong way), idle (did nothing),
acted (acted on a question), amount (right way, amount off).

  uv run python -m dink.evals.direction --run models/dink-v16 runs/dink/<name> [--misses]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

CASES = Path(__file__).with_name("direction_cases.jsonl")


def load(path=CASES):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _sign(x):
    return (x > 1e-9) - (x < -1e-9)


def facts(c):
    """(action dict or None, current value, wanted sign, room that way)."""
    a = next((x for x in c["actions"] if x["name"] == c["action"]), None)
    if a is None or a["lo"] is None:
        return a, None, 0, 0
    cur = next(f["value"] for f in c["state"] if f["name"] == a["sets"])
    want = {"up": 1, "down": -1}.get(c["dir"], 0)
    if c["amount"] == "number" and c["number"] is not None:
        want = _sign(c["number"] - cur)
    room = (a["hi"] - cur) if want > 0 else (cur - a["lo"]) if want < 0 else 0
    return a, cur, want, room


def same_action(c, a, name):
    """The right action, or another that sets the same field (volume_up / volume_down)."""
    if name == a["name"]:
        return True
    b = next((x for x in c["actions"] if x["name"] == name), None)
    return b is not None and a["sets"] is not None and b["sets"] == a["sets"]


def judge(c, out):
    """-> (dir ok, full ok, what was picked, kind of miss or None)."""
    d, f, said = _judge(c, out)
    if f:
        return d, f, said, None
    pick = out["pick"]
    a, cur, want, _ = facts(c)
    kind = ("acted" if c["action"] == "none" else "idle" if pick is None else
            "action" if a is None or not same_action(c, a, pick.name) else "amount" if d else "way")
    return d, f, said, kind


def _judge(c, out):
    pick = out["pick"]
    said = "(nothing)" if pick is None else pick.name if not pick.arg else f"{pick.name}={pick.value:g}"
    if c["action"] == "none":
        return pick is None, pick is None, said
    a, cur, want, room = facts(c)
    if pick is None:
        ok = a is not None and a["lo"] is not None and room <= 1e-9
        return ok, ok, said
    if not same_action(c, a, pick.name):
        return False, False, said
    if a["lo"] is None:
        return True, True, said
    span = a["hi"] - a["lo"]
    if c["amount"] == "number":
        ok = abs(pick.value - c["number"]) <= 0.02 * span
        return ok or (want != 0 and _sign(pick.value - cur) == want), ok, said
    ok = want != 0 and _sign(pick.value - cur) == want
    offered = [f for f, _ in out["pass2"]] if out["pass2"] else [f for f, _ in out["pass1"] if f.name == pick.name]
    same = sorted({f.value for f in offered if _sign(f.value - cur) == want}, key=lambda v: abs(v - cur))
    amt = c["amount"]
    if amt == "small":
        full = ok and (len(same) <= 1 or pick.value != same[-1])
    elif amt == "big":
        full = ok and (len(same) <= 1 or pick.value != same[0])
    elif amt == "end":
        full = ok and abs(pick.value - (a["hi"] if want > 0 else a["lo"])) <= 1e-6 * span
    else:
        full = ok
    return ok, full, said


def lone(model, c):
    """Raw model on pass-1 options, the right action offered once (the right way). None: n/a."""
    from ..data.scenario import Field
    from ..harness import fields, pass1_tools, score
    a, cur, want, room = facts(c)
    if a is None or a["lo"] is None or want == 0 or room <= 1e-9:
        return None
    tools = pass1_tools(model, c, c["say"])
    keep = [t for t in tools if t.name != a["name"] or _sign(t.value - cur) == want]
    if not any(t.name == a["name"] for t in keep):
        return None
    if len(keep) < 2:
        keep.append(Field("do_nothing_scene", 0, 0, 0, arg=False))
    p = score(model, fields(c), c["say"], keep)
    best = max(range(len(p)), key=p.__getitem__)
    return best < len(keep) and keep[best].name == a["name"]


def run(model, cases):
    from ..harness import decide
    rows = []
    for c in cases:
        d, f, said, kind = judge(c, decide(model, c, c["say"]))
        rows.append({"theme": c["theme"], "say": c["say"], "dir": d, "full": f, "said": said, "miss": kind,
                     "lone": lone(model, c), "want": want_text(c)})
    return rows


def want_text(c):
    if c["action"] == "none":
        return "(nothing)"
    if c["dir"] == "none" and c["amount"] != "number":
        return c["action"]
    return f"{c['action']} {c['dir'] if c['amount'] != 'number' else '='}{c['number'] if c['amount'] == 'number' else ' ' + c['amount']}"


def summary(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["theme"]].append(r)
    by["ALL"] = rows
    out = {}
    for t, rs in by.items():
        ln = [r["lone"] for r in rs if r["lone"] is not None]
        out[t] = {"n": len(rs), "dir": sum(r["dir"] for r in rs) / len(rs),
                  "full": sum(r["full"] for r in rs) / len(rs),
                  "lone": sum(ln) / len(ln) if ln else None, "lone_n": len(ln)}
    return out


def main():
    from ..play import load_model
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", default=["models/dink-v16"])
    ap.add_argument("--cases", default=str(CASES))
    ap.add_argument("--misses", action="store_true", help="list every miss")
    ap.add_argument("--save", help="write per-case results (JSON) here")
    a = ap.parse_args()
    cases = load(a.cases)
    res = {}
    for r in a.run:
        rows = run(load_model(r, "cpu"), cases)
        res[r] = {"summary": summary(rows), "rows": rows}
    themes = list(res[a.run[0]]["summary"])
    print(f"{len(cases)} cases.  dir / full / lone per run")
    print(f"{'theme':12s} {'n':>4s}  " + "  ".join(f"{Path(r).name[:17]:>17s}" for r in a.run))
    for t in themes:
        cells = []
        for r in a.run:
            s = res[r]["summary"][t]
            ln = "  - " if s["lone"] is None else f"{s['lone']:.2f}"
            cells.append(f"{s['dir']:.2f}/{s['full']:.2f}/{ln}")
        print(f"{t:12s} {res[a.run[0]]['summary'][t]['n']:4d}  " + "  ".join(f"{c:>17s}" for c in cells))
    kinds = ("action", "way", "idle", "acted", "amount")
    print("\nmisses by kind: " + ", ".join(kinds))
    for r in a.run:
        rows = res[r]["rows"]
        print(f"{Path(r).name:17s} " + "  ".join(f"{k} {sum(x['miss'] == k for x in rows):3d}" for k in kinds))
    if a.misses:
        for r in a.run:
            print(f"\n{r} misses :")
            for x in res[r]["rows"]:
                if not x["full"]:
                    print(f"  {x['miss']:6s} {x['theme']:11s} {x['say']!r:46s} -> {x['said']:34s} want {x['want']}")
    if a.save:
        Path(a.save).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
