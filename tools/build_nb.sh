#!/bin/bash
# Uso: tools/build_nb.sh tools/src/tema.py book/cap/tema/notebook.ipynb
# Convierte un script jupytext (formato percent) en cuaderno y lo ejecuta para guardar las salidas.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$ROOT/.venv/bin/jupytext" --to ipynb "$1" -o "$2" >/dev/null
"$ROOT/.venv/bin/jupyter" nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 "$2" 2>&1 | tail -3
"$ROOT/.venv/bin/python" -c "
import json, sys
path = sys.argv[1]
notebook = json.load(open(path))
notebook['metadata'].pop('jupytext', None)
json.dump(notebook, open(path, 'w'), indent=1, ensure_ascii=False)
" "$2"
