#!/usr/bin/env bash
# Usage: ./build.sh [slides|notes]  (default: both) -> output/slides.pdf, output/notes.pdf
set -euo pipefail
cd "$(dirname "$0")"

TEX=/Library/TeX/texbin
mkdir -p output

slides() { "$TEX/latexmk" -pdf -interaction=nonstopmode -file-line-error -output-directory=output slides.tex; }
notes() { pandoc notes.md -o output/notes.pdf --pdf-engine="$TEX/pdflatex"; }

targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(slides notes)
for t in "${targets[@]}"; do "$t"; done
