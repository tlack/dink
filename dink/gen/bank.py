"""LLM phrase bank for the generator (see dink/llm/phrases.py), and the annotator that keeps
the group split honest for text the generator didn't write.

A generator template records the words and templates it used, and the splitter holds records
out by them. An LLM phrase could quietly use a held-out word ("toastier") or a held-out
template's wording ("secure the ..."), so annotate() finds the generator's own vocabulary and
template wording inside the phrase and records those too. Each phrase also gets an id
(uses["llm"]), so the splitter can hold out a share of LLM phrasings as their own test set.
"""
import json
import re
import zlib
from pathlib import Path

from . import home as H


def _vocab():
    out = set()
    for r in H.ROOMS:
        out |= set(r.words)
    for k in H.KINDS.values():
        out |= set(k.nouns) | set(k.up) | set(k.down)
    for opts in H.INTENSITY.values():
        out |= set(opts)
    out |= set(H.POLITE)
    for h in H.HOUSE.values():
        out |= set(h["words"])
    for phrases in H.SCENES.values():
        out |= set(phrases)
    out |= set(H.OFF_DOMAIN)
    return {v: re.compile(rf"\b{re.escape(v.strip())}\b") for v in out if v.strip()}


def _templates():
    """Template -> regex over its literal wording (slots become gaps). Templates with less
    than two words of their own wording are too generic to recognize, and are skipped."""
    tpls = list(H.RELATIVE) + list(H.NUMERIC_ALL) + list(H.QUESTIONS)
    for v in list(H.TURN.values()) + list(H.LOCKS.values()):
        tpls += v
    for v in H.INDIRECT.values():
        tpls += [t for t, _ in v]
    out = {}
    for t in set(tpls):
        parts = [p.strip() for p in re.split(r"\{\w+\}", t) if p.strip()]
        if sum(len(p.split()) for p in parts) < 2:
            continue
        out[t] = re.compile(r"\b" + r"\b.*?\b".join(re.escape(p) for p in parts) + r"\b")
    return out


VOCAB = _vocab()
TEMPLATES = _templates()


def annotate(text):
    """-> (words, templates) of the generator's that this text uses."""
    return ([v for v, rx in VOCAB.items() if rx.search(text)],
            [t for t, rx in TEMPLATES.items() if rx.search(text)])


def phrase_id(text):
    return f"{zlib.crc32(text.encode()):08x}"


def load(path="data/llm/phrases.jsonl"):
    """-> {key tuple: [text, ...]}"""
    bank = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            bank.setdefault(tuple(r["key"]), []).append(r["text"])
    return bank
