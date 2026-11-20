#!/bin/sh
# Uses existing local dependencies, source snapshot and pinned E5 cache only.
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo"
if [ ! -x "$repo/.venv/bin/python" ]; then
  echo 'Missing .venv: follow README setup instructions.' >&2
  exit 1
fi
if [ ! -f "$repo/data/application-multicrop-v2/final-snapshot/manifest.json" ]; then
  echo 'Private multi-crop collection unavailable. See docs/multicrop-application.md; no synthetic fallback was substituted.' >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/src"
exec "$repo/.venv/bin/python" -m krishisaarthi.ui.server --root "$repo" --real --collection "$repo/data/application-multicrop-v2/final-snapshot" --port "${PORT:-8767}"
