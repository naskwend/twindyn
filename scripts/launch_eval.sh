#!/usr/bin/env bash
# Score a fitted checkpoint on the hold-out split and on every external cohort.
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <checkpoint> [--experiment NAME] [extra --set overrides]" >&2
  exit 2
fi

CHECKPOINT="$1"
shift

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)/src"
export TWINDYN_DATA_ROOT="${TWINDYN_DATA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/data}"

python3 -m twindyn.cli.evaluate \
  --checkpoint "${CHECKPOINT}" \
  --set "data.root=${TWINDYN_DATA_ROOT}" \
  "$@"
