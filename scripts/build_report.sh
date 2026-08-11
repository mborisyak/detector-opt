#!/usr/bin/env bash
# Build docs/enzyme-retune-report.pdf.
#
# Goes through LaTeX rather than straight to PDF for one reason: the decision tables have long
# multi-line cells, and pandoc's booktabs output rules only the header and the foot, so the rows run
# together and are unreadable. A rule after every row cannot be expressed in a markdown table, so
# this inserts one into the generated .tex before compiling. Everything else is pandoc's.
set -euo pipefail
cd "$(dirname "$0")/.."

SOURCE=docs/enzyme-retune-report.md
TEX=$(mktemp -t report-XXXXXX.tex)
trap 'rm -f "$TEX" "${TEX%.tex}".{aux,log,out}' EXIT

pandoc "$SOURCE" -s -o "$TEX" --pdf-engine=xelatex --resource-path=.:docs

python3 - "$TEX" <<'PY'
import re
import sys

path = sys.argv[1]
lines = open(path).read().split("\n")
out, inside = [], False
for line in lines:
    if "\\begin{longtable}" in line:
        inside = True
    elif "\\end{longtable}" in line:
        inside = False
    out.append(line)
    # A pandoc table row ends at a line closing with `\\`; the structural rules and the header
    # machinery end differently, so ruling on that alone puts one line under each row and nowhere
    # else. `\midrule\endhead` already separates the header, so skip it to avoid a double rule.
    if inside and re.search(r"\\\\\s*$", line) and "endhead" not in line and "endfirsthead" not in line:
        out.append("\\hline")
open(path, "w").write("\n".join(out))
PY

xelatex -interaction=batchmode -halt-on-error -output-directory "$(dirname "$TEX")" "$TEX" > /dev/null
xelatex -interaction=batchmode -halt-on-error -output-directory "$(dirname "$TEX")" "$TEX" > /dev/null
mv "${TEX%.tex}.pdf" docs/enzyme-retune-report.pdf
echo "wrote docs/enzyme-retune-report.pdf ($(pdfinfo docs/enzyme-retune-report.pdf | awk '/Pages/{print $2}') pages)"
