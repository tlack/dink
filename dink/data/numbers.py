"""Pull numbers out of a prompt, with a unit hint, for the relation step.

A prompt number has no range of its own; the model compares it with each tool in that
tool's declared range. The hint says how: "raw" (72, "72 degrees": a value in the tool's
units) or "level" ("50 percent", "half": a fraction of any range). Speech recognition
writes numbers as words ("seventy two"), so number words are converted too.
"""
import re

MAX_NUMBERS = 3

ONES = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
        "ninety": 90}
LEVEL_WORDS = {"half": 0.5, "halfway": 0.5, "quarter": 0.25, "full": 1.0, "max": 1.0, "maximum": 1.0,
               "minimum": 0.0}
PERCENT = re.compile(r"^(%|percent|per cent|pct)")


def words_to_digits(text):
    """"set it to seventy two" -> "set it to 72"; "one hundred" -> "100" (0..999)."""
    out, i, toks = [], 0, text.split()
    while i < len(toks):
        w = toks[i].strip(",.!?")
        if w in ONES or w in TENS or (w == "a" and i + 1 < len(toks) and toks[i + 1] == "hundred"):
            n, j = 0, i
            if toks[j].strip(",.") in ONES and j + 1 < len(toks) and toks[j + 1].strip(",.") == "hundred" \
                    or toks[j] == "a":
                n += 100 * (ONES.get(toks[j].strip(",."), 1))
                j += 2
                if j < len(toks) and toks[j] == "and":
                    j += 1
            if j < len(toks) and toks[j].strip(",.") in TENS:
                n += TENS[toks[j].strip(",.")]
                j += 1
                if j < len(toks) and toks[j].strip(",.") in ONES and ONES[toks[j].strip(",.")] < 10:
                    n += ONES[toks[j].strip(",.")]
                    j += 1
            elif j < len(toks) and toks[j].strip(",.") in ONES:
                n += ONES[toks[j].strip(",.")]
                j += 1
            if j > i:
                out.append(str(n))
                i = j
                continue
        out.append(toks[i])
        i += 1
    return " ".join(out)


def extract(prompt, max_numbers=MAX_NUMBERS):
    """-> [(value, kind)] with kind "raw" or "level", in order of appearance."""
    text = words_to_digits(prompt.lower())
    found = []
    for m in re.finditer(r"-?\d+(?:\.\d+)?", text):
        v = float(m.group())
        rest = text[m.end():].lstrip()
        found.append((v / 100, "level") if PERCENT.match(rest) else (v, "raw"))
    for w, lv in LEVEL_WORDS.items():
        if re.search(rf"\b{w}\b", text) and not any(k == "level" for _, k in found):
            if w in ("full", "max", "maximum", "minimum") and not re.search(rf"\b(to|at|on)\s+{w}\b", text):
                continue                            # "full" alone is often not a setting ("i'm full")
            found.append((lv, "level"))
    return found[:max_numbers]
