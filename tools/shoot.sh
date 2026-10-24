#!/bin/bash
# Uso: tools/shoot.sh ruta/slides.html N [salida_dir]  -> captura la diapositiva N (1-based) en PNG
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
HTML="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
OUT="${3:-/tmp}/slide_$2.png"
"$CHROME" --headless=new --disable-gpu --hide-scrollbars --user-data-dir=$(mktemp -d) --window-size=1280,720 --virtual-time-budget=4000 --screenshot="$OUT" "file://$HTML#$2" >/dev/null 2>&1
echo "$OUT"
