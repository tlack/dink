"""The scenario format shared by generators, augmentations and the encoder.

A scenario is one decision: a structured state, a text prompt, up to 8 tools, and the
right answer as a probability distribution over the tools plus "nothing relevant"
(the last entry). Most answers are one-hot; ambiguous requests spread probability over
the plausible tools.
"""
from dataclasses import asdict, dataclass, field
import json

NAME_BYTES = 40     # state field and tool names (dink/gen/names.py: up to 35 in the smart home)
PROMPT_BYTES = 64
MAX_SLOTS = 8       # state fields, and separately tools


@dataclass
class Field:
    """A state field or a tool: a name (up to NAME_BYTES) and a value within a declared range.
    Tools without an argument (scenes) have arg=False and value 0."""
    name: str
    value: float
    lo: float
    hi: float
    arg: bool = True

    def norm(self) -> float:
        if not self.arg or self.hi <= self.lo:
            return 0.0
        return min(1.0, max(0.0, (self.value - self.lo) / (self.hi - self.lo)))


@dataclass
class Scenario:
    state: list          # [Field]
    prompt: str
    tools: list          # [Field]
    target: list         # len(tools) + 1 probabilities; last = "nothing relevant"
    meta: dict = field(default_factory=dict)

    @property
    def label(self) -> int:
        """Index of the most likely answer; len(tools) means "nothing relevant"."""
        return max(range(len(self.target)), key=lambda i: self.target[i])

    def check(self):
        assert 0 < len(self.tools) <= MAX_SLOTS and len(self.state) <= MAX_SLOTS
        for f in self.state + self.tools:
            assert 0 < len(f.name.encode()) <= NAME_BYTES, f.name
            assert f.lo <= f.value <= f.hi or not f.arg, (f.name, f.value, f.lo, f.hi)
        assert len(self.prompt.encode()) <= PROMPT_BYTES, self.prompt
        assert len(self.target) == len(self.tools) + 1
        assert abs(sum(self.target) - 1) < 1e-6 and min(self.target) >= 0
        names = [f.name for f in self.state]
        assert len(names) == len(set(names)), names
        return self

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @staticmethod
    def from_json(line: str) -> "Scenario":
        d = json.loads(line)
        return Scenario([Field(**f) for f in d["state"]], d["prompt"],
                        [Field(**f) for f in d["tools"]], d["target"], d.get("meta", {}))

    def show(self) -> str:
        """Human-readable rendering, for eyeballing generated data."""
        def fmt(f):
            if not f.arg:
                return f"{f.name}()"
            v = f"{f.value:g}"
            return f"{f.name}={v} [{f.lo:g}..{f.hi:g}]"
        lines = [f"  state : {'  '.join(fmt(f) for f in self.state)}",
                 f"  prompt: {self.prompt!r}"]
        for i, t in enumerate(self.tools):
            p = self.target[i]
            lines.append(f"  {'->' if p > 0 else '  '} [{i}] {fmt(t):28s} {p:.2f}" if p > 0
                         else f"     [{i}] {fmt(t)}")
        pn = self.target[-1]
        lines.append(f"  {'->' if pn > 0 else '  '} [nothing relevant]{'':14s}{pn:.2f}" if pn > 0
                     else "     [nothing relevant]")
        m = self.meta
        lines.insert(0, f"# {m.get('intent')}  cls={m.get('cls')}  room={m.get('room')}"
                        f"{'  AMBIGUOUS' if m.get('ambiguous') else ''}")
        return "\n".join(lines)
