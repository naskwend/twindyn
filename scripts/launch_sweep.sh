#!/usr/bin/env bash
# Sweep the axis length and the state dimension and select the open quantities.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)/src"
export TWINDYN_DATA_ROOT="${TWINDYN_DATA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/data}"

python3 -m twindyn.cli.sweep \
  --set "data.root=${TWINDYN_DATA_ROOT}" \
  "$@"
