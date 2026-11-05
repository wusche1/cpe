#!/usr/bin/env bash
# Build the report and/or the slides. Usage: ./build.sh [main|slides]  (default: both)
set -euo pipefail
cd "$(dirname "$0")"

TEX=/Library/TeX/texbin
mkdir -p output

build() {
  local doc=$1
  "$TEX/pdflatex" -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=./output "$doc.tex"
  "$TEX/bibtex" "./output/$doc" || true   # no \cite yet is not an error
  "$TEX/pdflatex" -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=./output "$doc.tex"
  "$TEX/pdflatex" -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=./output "$doc.tex"
  echo "built output/$doc.pdf"
}

docs=("$@")
[ ${#docs[@]} -eq 0 ] && docs=(main slides)
for doc in "${docs[@]}"; do build "$doc"; done
