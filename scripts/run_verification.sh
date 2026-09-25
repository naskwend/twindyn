#!/usr/bin/env bash
# Run both verification passes and write the root artefacts, manifest last.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}/src"

python3 -m twindyn.cli.verify --root "${ROOT}" "$@"
