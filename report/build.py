#!/usr/bin/env python3
"""Builds report/report.pdf from report/report.md with pandoc and XeLaTeX.

    python3 analysis/plot_scaling.py && python3 analysis/make_tables.py   # figures + tables
    python3 report/build.py                                               # report/report.pdf

A line of the form
    <!-- include: tables/report_strong_vanda.md | Caption text -->
is replaced by that file (a Markdown table written by analysis/make_tables.py) followed by the
caption. A missing or empty file becomes an italic "pending" note, so the report builds before
every job has run. Every table, included or written in report.md, is then preceded by a
\needspace (header.tex) sized from its estimated printed height, so that a short table never splits
across pages and a heading right above it moves with it; a one-line lead-in likewise stays with
the list it introduces. The expanded Markdown goes to
report/build/report.md.
"""
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "report.md")
OUT = os.path.join(HERE, "report.pdf")
BUILD = os.path.join(HERE, "build")
INCLUDE = re.compile(r"^<!-- include: (\S+?)(?:\s*\|\s*(.*?))?\s*-->\s*$")


def expand(text):
    out = []
    for line in text.splitlines():
        m = INCLUDE.match(line)
        if not m:
            out.append(line)
            continue
        path, caption = os.path.join(HERE, m.group(1)), m.group(2)
        body = open(path).read().strip() if os.path.exists(path) else ""
        if body.startswith("|"):
            out.append(body)
            if caption:
                out += ["", f"Table: {caption}"]
        elif body:
            out.append(body)
        else:
            out.append(f"*[Pending: {caption or m.group(1)}. No data yet.]*")
    return "\n".join(out) + "\n"


# Printed-height model of a pandoc pipe table, in baselines of the 11pt body text (13.6pt).
FOOT_LINE = 0.81      # a \footnotesize table line (11pt) in body baselines
CAP_LINE = 0.88       # a \small caption line (12pt)
TABLE_CHARS = 110     # \footnotesize characters across the 16.6 cm text width (on the safe side)
CAPTION_CHARS = 100   # \small characters across the text width
PANDOC_COLUMNS = 72   # pandoc wraps a table to the text width if any source line is longer


def shown(cell):
    """Rough printed text of a Markdown/LaTeX table cell."""
    return re.sub(r"[$`*{}^_\\]", "", re.sub(r"\\[a-zA-Z]+", "x", cell)).strip()


def table_height(rows, caption):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    sep = next(i for i, r in enumerate(cells) if all(set(c) <= set("-: ") for c in r))
    widths = [len(c) or 1 for c in cells[sep]]  # pandoc's relative column widths
    chars = [max(4, TABLE_CHARS * w / sum(widths) - 3) for w in widths]
    wrap = max(map(len, rows)) > PANDOC_COLUMNS
    lines = sum(max((len(textwrap.wrap(shown(c), int(n))) or 1) if wrap else 1 for c, n in zip(r, chars))
                for i, r in enumerate(cells) if i != sep)
    cap = math.ceil(len(shown(caption)) / CAPTION_CHARS) if caption else 0
    return FOOT_LINE * lines + CAP_LINE * cap + 2.5  # + rules and the space around the table


def keep_tables_whole(text):
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        if not lines[i].startswith("|"):
            out.append(lines[i])
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].startswith("|"):
            j += 1
        k = j
        while k < len(lines) and not lines[k].strip():
            k += 1
        caption = lines[k][6:] if k < len(lines) and lines[k].startswith("Table:") else ""
        need = table_height(lines[i:j], caption)
        at = len(out)
        while at and not out[at - 1].strip():
            at -= 1
        if at and out[at - 1].startswith("#"):  # a heading right above the table goes with it
            at, need = at - 1, need + 3
        else:
            at = len(out)
        out[at:at] = ["", f"\\needspace{{{need:.1f}\\baselineskip}}", ""]
        out += lines[i:j]
        i = j
    return "\n".join(out)


def keep_leadins(text):
    """A one-line lead-in ("**Limitations.**") never ends a page without its list."""
    lines, out = text.split("\n"), []
    for i, line in enumerate(lines):
        nxt = next((l for l in lines[i + 1:] if l.strip()), "")
        if (0 < len(line) <= 140 and not line.startswith(("-", "|", "#", "!", "\\", "<", " ")) and
                (i == 0 or not lines[i - 1].strip()) and i + 1 < len(lines) and not lines[i + 1].strip()
                and re.match(r"(- |\d+\. )", nxt)):
            out += ["\\needspace{4\\baselineskip}", ""]
        out.append(line)
    return "\n".join(out)


def main():
    if not shutil.which("pandoc"):
        sys.exit("pandoc not found")
    os.makedirs(BUILD, exist_ok=True)
    md = os.path.join(BUILD, "report.md")
    with open(SRC) as fh, open(md, "w") as out:
        out.write(keep_leadins(keep_tables_whole(expand(fh.read()))))
    cmd = ["pandoc", md, "-o", OUT, "--pdf-engine=xelatex", f"--resource-path={HERE}",
           "-H", os.path.join(HERE, "header.tex"),
           "-V", "geometry:a4paper,margin=2.2cm", "-V", "fontsize=11pt",
           "-V", "mainfont=STIXGeneral", "-V", "mathfont=STIX Two Math", "-V", "monofont=Menlo",
           "-V", "monofontoptions=Scale=0.82", "-V", "colorlinks=true", "-V", "linkcolor=blue",
           "-V", "urlcolor=blue", "-V", "papersize=a4"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    warnings = [l for l in r.stderr.splitlines() if "Missing character" in l or "rror" in l]
    for w in warnings[:20]:
        print(w)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        sys.exit(r.returncode)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
