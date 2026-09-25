#!/usr/bin/env bash
# Inspect the staged resource root and write the preparation manifest.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)/src"
ROOT="${TWINDYN_DATA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/data}"

python3 -m twindyn.cli.prepare_data \
  --set "data.root=${ROOT}" \
  --output-dir "${ROOT}/.." \
  --write-ordered-axis \
  "$@"
