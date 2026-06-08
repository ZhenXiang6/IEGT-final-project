#!/usr/bin/env bash
# watch.sh — auto-recompile whenever the .tex changes. No extra tools needed.
# Usage:  ./watch.sh [file.tex]      (default: OTT_enhanced_slides.tex)
#         Ctrl-C to stop.
# Tip: open the PDF in Preview.app (or VS Code's PDF viewer) — it refreshes on each rebuild.
cd "$(dirname "$0")" || exit 1
TEX="${1:-OTT_enhanced_slides.tex}"
BASE="${TEX%.tex}"
LOG=/tmp/watch_xelatex.log

echo "👀 watching ${TEX} — edit & save to auto-compile (Ctrl-C to stop)"
last=""
while true; do
  cur=$(stat -f %m "$TEX" 2>/dev/null)
  if [ "$cur" != "$last" ] && [ -n "$cur" ]; then
    last="$cur"
    printf "[%s] compiling… " "$(date +%H:%M:%S)"
    if xelatex -interaction=nonstopmode -halt-on-error "$TEX" >"$LOG" 2>&1 \
       && xelatex -interaction=nonstopmode -halt-on-error "$TEX" >"$LOG" 2>&1; then
      echo "✓ ${BASE}.pdf updated"
    else
      echo "✗ error:"
      grep -nE '^\! |:[0-9]+: ' "$LOG" | head -8
    fi
  fi
  sleep 1
done
