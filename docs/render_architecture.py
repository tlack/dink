"""Render docs/architecture.dot to docs/architecture.svg, with theme-aware colors.

Graphviz writes fixed colors; the .dot uses placeholder colors (#0000xx) that are rewritten
here into CSS variables with fallbacks, so the SVG follows a page's light / dark theme and
still reads on its own (e.g. on GitHub).

  uv run python docs/render_architecture.py
"""
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
TOKENS = {  # placeholder -> (css variable, fallback)
    "#000001": ("--ink", "#17211e"), "#000002": ("--muted", "#56655f"), "#000003": ("--query", "#1f7a5a"),
    "#000004": ("--cached", "#a86b12"), "#000005": ("--train", "#6b7a76"), "#000006": ("--panel", "#ffffff"),
    "#000007": ("--fill-query", "#e8f4ee"), "#000008": ("--fill-cached", "#f7eedd"),
    "#000009": ("--fill-train", "#eceeed"), "#00000a": ("--rule", "#d5ddd9"), "#00000b": ("--fill-out", "#d3ecdf"),
}


def themed(svg):
    def fix(tag):
        style = []
        for attr in ("fill", "stroke"):
            m = re.search(rf'\s{attr}="(#0000[0-9a-f]{{2}})"', tag)
            if m and m.group(1) in TOKENS:
                var, fb = TOKENS[m.group(1)]
                style.append(f"{attr}:var({var},{fb})")
                tag = tag.replace(m.group(0), "")
        if style:
            tag = re.sub(r"(/?>)$", f' style="{";".join(style)}"\\1', tag)
        return tag
    svg = re.sub(r"<(?:text|path|polygon|ellipse|polyline)\b[^>]*>", lambda m: fix(m.group(0)), svg)
    svg = svg.replace('font-family="Helvetica,sans-Serif"', 'font-family="var(--sans, Helvetica, sans-serif)"')
    svg = re.sub(r'font-family="Helvetica[^"]*"', 'font-family="var(--sans, Helvetica, sans-serif)"', svg)
    svg = re.sub(r"<\?xml[^>]*>\s*<!DOCTYPE[^>]*>\s*", "", svg, flags=re.S)
    leftover = set(re.findall(r"#0000[0-9a-f]{2}", svg))
    assert not leftover, leftover
    return svg


def main():
    svg = subprocess.run(["dot", "-Tsvg", str(HERE / "architecture.dot")], capture_output=True, text=True,
                         check=True).stdout
    (HERE / "architecture.svg").write_text(themed(svg))
    print("wrote", HERE / "architecture.svg")


if __name__ == "__main__":
    main()
