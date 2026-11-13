#!/usr/bin/env python3
"""Builds report/report.pdf from report/report.md with pandoc and XeLaTeX.

    python3 analysis/plot_scaling.py && python3 analysis/make_tables.py   # figures + tables
    python3 report/build.py                                               # report/report.pdf

A line of the form
    <!-- include: tables/report_strong_vanda.md | Caption text -->
is replaced by that file (a Markdown table written by analysis/make_tables.py) followed by the
caption. A missing or empty file becomes an italic "pending" note, so the report builds before
every job has run. The expanded Markdown goes to report/build/report.md.
"""
import os
import re
import shutil
import subprocess
import sys

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


def main():
    if not shutil.which("pandoc"):
        sys.exit("pandoc not found")
    os.makedirs(BUILD, exist_ok=True)
    md = os.path.join(BUILD, "report.md")
    with open(SRC) as fh, open(md, "w") as out:
        out.write(expand(fh.read()))
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
