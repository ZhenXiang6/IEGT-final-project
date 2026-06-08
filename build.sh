#!/usr/bin/env bash
# build.sh — compile a Beamer deck with XeLaTeX (run twice for TOC/refs) and open the PDF.
# Usage:  ./build.sh [file.tex]      (default: OTT_enhanced_slides.tex)
#         NO_OPEN=1 ./build.sh       (compile only, don't open the PDF)
cd "$(dirname "$0")" || exit 1
TEX="${1:-OTT_enhanced_slides.tex}"
BASE="${TEX%.tex}"
LOG=/tmp/build_xelatex.log

echo "compiling ${TEX} (xelatex ×2)…"
if xelatex -interaction=nonstopmode -halt-on-error "$TEX" >"$LOG" 2>&1 \
   && xelatex -interaction=nonstopmode -halt-on-error "$TEX" >"$LOG" 2>&1; then
  pages=$(pdfinfo "${BASE}.pdf" 2>/dev/null | awk '/^Pages/{print $2}')
  echo "✓ ${BASE}.pdf built (${pages:-?} pages)"
  [ -z "$NO_OPEN" ] && open "${BASE}.pdf"
else
  echo "✗ compile error — first messages:"
  grep -nE '^\! |:[0-9]+: ' "$LOG" | head -20
  echo "  (full log: $LOG)"
  exit 1
fi
