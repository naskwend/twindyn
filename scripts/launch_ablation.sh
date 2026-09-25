#!/usr/bin/env bash
# Substitute every component one at a time and write the ablation table.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)/src"
export TWINDYN_DATA_ROOT="${TWINDYN_DATA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/data}"

python3 -m twindyn.cli.ablation \
  --set "data.root=${TWINDYN_DATA_ROOT}" \
  "$@"
