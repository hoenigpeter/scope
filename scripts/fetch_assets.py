#!/usr/bin/env python3
"""Download, verify and install the public model and REAL275 sample archives."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def asset_path(root, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Invalid asset path: {name}')
    target = (root / path).resolve()
    allowed = target == root / 'weights/scope.pth' or target.is_relative_to(root / 'data/real275_demo')
    if not allowed or not target.is_relative_to(root):
        raise ValueError(f'Invalid asset path: {name}')
    return target


def installed(root, entry):
    """Only reuse files that match the release's individual file hashes."""
    for name, checksum in entry['files'].items():
        path = asset_path(root, name)
        if not path.is_file() or digest(path) != checksum:
            return False
    return bool(entry['files'])


def download(entry, destination):
    for attempt in range(3):
        try:
            print(f"Downloading {entry['filename']} ({entry['bytes'] / 2**20:.1f} MiB)...", flush=True)
            request = urllib.request.Request(entry['url'], headers={'User-Agent': 'SCOPE-release/1.0'})
            with urllib.request.urlopen(request, timeout=60) as response, destination.open('wb') as target:
                count, next_report = 0, 64 * 1024 * 1024
                while block := response.read(1024 * 1024):
                    count += len(block)
                    if count > entry['bytes']:
                        raise ValueError('Download exceeds expected archive size')
                    target.write(block)
                    if count >= next_report:
                        print(f"  {count / entry['bytes']:.0%}", flush=True)
                        next_report += 64 * 1024 * 1024
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def install_archive(root, entry, archive, staging):
    if archive.stat().st_size != entry['bytes'] or digest(archive) != entry['sha256']:
        raise ValueError(f"Archive checksum/size mismatch: {entry['filename']}")
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(entry['files']):
            raise ValueError('Archive contents do not match the release manifest')
        for member in members:
            asset_path(root, member.name)
            if not member.isfile():
                raise ValueError(f'Expected a regular file: {member.name}')
        # Verify every staged file before modifying any installed files.
        for member in members:
            target = staging / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open('wb') as dest:
                shutil.copyfileobj(source, dest)
            if digest(target) != entry['files'][member.name]:
                raise ValueError(f'File checksum mismatch: {member.name}')
        for member in members:
            target = asset_path(root, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / member.name, target)


def ensure_assets(root=ROOT, filenames=None, archive_dir=None, force=False):
    root = root.resolve()
    entries = json.loads((root / 'assets.json').read_text())['assets']
    for entry in entries:
        if filenames is not None and entry['filename'] not in filenames:
            continue
        if not force and installed(root, entry):
            print(f"Verified installed {entry['filename']}", flush=True)
            continue
        if not archive_dir and not entry['url']:
            raise ValueError(f"No download URL for {entry['filename']}")
        # Project-local staging keeps atomic replacements on the same filesystem.
        with tempfile.TemporaryDirectory(prefix='.asset-install-', dir=root) as tmp:
            staging = Path(tmp) / 'files'
            archive = archive_dir / entry['filename'] if archive_dir else Path(tmp) / 'download.tar.gz'
            if not archive_dir:
                download(entry, archive)
            install_archive(root, entry, archive, staging)
        print(f"Installed {entry['filename']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir', type=Path, help='Use local downloaded archives instead of URLs')
    parser.add_argument('--force', action='store_true', help='Reinstall even if installed file checksums match')
    args = parser.parse_args()
    try:
        ensure_assets(archive_dir=args.archive_dir, force=args.force)
    except (OSError, ValueError, tarfile.TarError) as exc:
        parser.exit(1, f'Asset installation failed: {exc}\n')


if __name__ == '__main__':
    main()
