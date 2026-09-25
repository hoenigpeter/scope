#!/usr/bin/env python3
"""Package portable source, tests, documentation and asset metadata."""
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    '.gitignore', 'assets.json', 'CITATION.cff', 'citation.bib',
    'conda-linux-64.lock.txt', 'DATA_NOTICE.md', 'demo.py', 'environment.yml',
    'inference.py', 'LICENSE', 'README.md', 'requirements.lock.txt',
    'requirements.txt', 'run_demo.sh', 'setup.sh', 'VALIDATION.md', 'view_results.py',
)


def source_files(root):
    files = [root / name for name in ROOT_FILES]
    for folder, suffixes in [('scope', {'.py'}), ('scripts', {'.py', '.sh'}),
                             ('tests', {'.py'}), ('docs', {'.md', '.png', '.gif', '.json'})]:
        for path in (root / folder).rglob('*'):
            relative = path.relative_to(root)
            if (path.is_file() and not path.is_symlink() and path.suffix in suffixes
                    and not any(part.startswith('.') or part == '__pycache__' for part in relative.parts)):
                files.append(path)
    if any(not path.is_file() or path.is_symlink() for path in files):
        raise ValueError('A required source file is missing or is a symbolic link')
    return sorted(files)


def main():
    output = ROOT / 'artifacts/scope-source.tar.gz'
    output.parent.mkdir(exist_ok=True)
    with tarfile.open(output, 'w:gz') as tar:
        for path in source_files(ROOT):
            tar.add(path, arcname=str(Path('scope') / path.relative_to(ROOT)))
    print(f'{output}: {output.stat().st_size / 1024:.0f} KiB')


if __name__ == '__main__':
    main()
