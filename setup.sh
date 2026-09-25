#!/usr/bin/env bash
# Create a project-local conda environment and compile pinned TEASER++ sources.
set -euo pipefail
# Isolate Python from sourced ROS workspaces and user-site packages.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(uname -s)-$(uname -m)" != Linux-x86_64 ]]; then
  echo 'This installer supports Linux x86_64.' >&2; exit 1
fi
CONDA="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA" ]]; then
  mkdir -p "$ROOT/.bootstrap"
  URL='https://github.com/conda-forge/miniforge/releases/download/25.3.1-0/Miniforge3-25.3.1-0-Linux-x86_64.sh'
  curl -fL --retry 3 "$URL" -o "$ROOT/.bootstrap/miniforge.sh"
  curl -fL --retry 3 "$URL.sha256" -o "$ROOT/.bootstrap/miniforge.sha256"
  EXPECTED="$(cut -d ' ' -f 1 "$ROOT/.bootstrap/miniforge.sha256")"
  ACTUAL="$(sha256sum "$ROOT/.bootstrap/miniforge.sh" | cut -d ' ' -f 1)"
  [[ "$EXPECTED" == "$ACTUAL" ]] || { echo 'Miniforge checksum mismatch' >&2; exit 1; }
  if [[ ! -x "$ROOT/.bootstrap/conda/bin/conda" ]]; then
    bash "$ROOT/.bootstrap/miniforge.sh" -b -p "$ROOT/.bootstrap/conda"
  fi
  CONDA="$ROOT/.bootstrap/conda/bin/conda"
fi
if [[ ! -x "$ROOT/.env/bin/python" ]]; then
  "$CONDA" create --prefix "$ROOT/.env" --file "$ROOT/conda-linux-64.lock.txt" --yes
fi
# Activation runs compiler activation hooks required for the C++ extension.
eval "$("$CONDA" shell.bash hook)"
conda activate "$ROOT/.env"
python -m pip install -r "$ROOT/requirements.lock.txt"
bash "$ROOT/scripts/build_teaser.sh"
python -m pip check
python -m unittest discover -s "$ROOT/tests" -v
python -m pip freeze > "$ROOT/.env/pip-freeze.txt"
conda list --explicit > "$ROOT/.env/conda-explicit.txt"
echo 'Setup complete. Run: bash run_demo.sh'
