#!/usr/bin/env bash
set -euo pipefail
# Isolate Python from sourced ROS workspaces and user-site packages.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCOPE_PYTHON:-$ROOT/.env/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo 'Environment missing. Run: bash setup.sh' >&2
  exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$PYTHON" "$ROOT/demo.py" "$@"
