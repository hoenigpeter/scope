#!/usr/bin/env bash
set -euo pipefail
# Isolate Python from sourced ROS workspaces and user-site packages.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${SCOPE_PYTHON:-$ROOT/.env/bin/python}" "$ROOT/scripts/fetch_assets.py" "$@"
