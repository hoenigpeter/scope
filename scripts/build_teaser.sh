#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/.build/TEASER-plusplus"
REV='baf69d948d77e8fe496a82e2fa3b1f41a9b1156f'
mkdir -p "$ROOT/.build"
if [[ ! -d "$SOURCE/.git" ]]; then
  git clone https://github.com/MIT-SPARK/TEASER-plusplus.git "$SOURCE"
fi
git -C "$SOURCE" checkout --detach "$REV"
# Upstream leaves PMC unpinned; pin it here. Skip the unconditional test-only
# googletest download, which is unnecessary with BUILD_TESTS=OFF.
python - "$SOURCE" <<'PY'
from pathlib import Path
import subprocess
import sys
root = Path(sys.argv[1])
for name in ['CMakeLists.txt', 'cmake/pmc.CMakeLists.txt.in', 'teaser/src/graph.cc']:
    text = subprocess.check_output(['git', '-C', str(root), 'show', 'HEAD:' + name], text=True)
    if name == 'CMakeLists.txt':
        start, end = text.index('# googletest'), text.index('# pmc (')
        text = text[:start] + text[end:]
    elif name == 'teaser/src/graph.cc':
        text = text.replace('vector<', 'std::vector<')
    else:
        text = text.replace('GIT_REPOSITORY    https://github.com/jingnanshi/pmc.git',
                            'GIT_REPOSITORY    https://github.com/jingnanshi/pmc.git\n        GIT_TAG a2dfd612a501bca83c47206255dbbff619481f97')
    (root / name).write_text(text)
PY
cmake -S "$SOURCE" -B "$SOURCE/build" -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=OFF -DBUILD_DOC=OFF -DBUILD_TEASER_FPFH=OFF \
  -DBUILD_PYTHON_BINDINGS=ON -DTEASERPP_PYTHON_VERSION=3.10 \
  -DPYTHON_EXECUTABLE="$(command -v python)" -DENABLE_DIAGNOSTIC_PRINT=OFF \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DCMAKE_INSTALL_RPATH="$CONDA_PREFIX/lib"
cmake --build "$SOURCE/build" --parallel "${BUILD_JOBS:-2}"
cmake --install "$SOURCE/build"
python -m pip install --no-build-isolation "$SOURCE/build/python"
python -c 'import teaserpp_python; print("TEASER++ ready")'
