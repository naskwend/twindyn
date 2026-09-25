#!/usr/bin/env bash
# Fit the representation and the read-out for one experiment configuration.
set -euo pipefail

EXPERIMENT="${1:-main}"
shift || true

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)/src"
export TWINDYN_DATA_ROOT="${TWINDYN_DATA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/data}"

if command -v torchrun >/dev/null 2>&1 && [ "${WORLD_SIZE:-1}" -gt 1 ]; then
  torchrun --nproc_per_node="${WORLD_SIZE}" \
    -m twindyn.cli.train --experiment "${EXPERIMENT}" --set "data.root=${TWINDYN_DATA_ROOT}" "$@"
else
  python3 -m twindyn.cli.train --experiment "${EXPERIMENT}" --set "data.root=${TWINDYN_DATA_ROOT}" "$@"
fi
