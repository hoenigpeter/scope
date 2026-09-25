#!/usr/bin/env python3
"""Create deterministic upload archives and their download manifest."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def archive(name, paths):
    output = ROOT / 'artifacts' / name
    output.parent.mkdir(exist_ok=True)
    with output.open('wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=1) as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as tar:
            for path in paths:
                info = tar.gettarinfo(str(path), arcname=str(path.relative_to(ROOT)))
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ''
                info.mode = 0o644
                with path.open('rb') as f:
                    tar.addfile(info, f)
    digest = hashlib.sha256()
    with output.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            digest.update(block)
    return dict(filename=name, url='', sha256=digest.hexdigest(), bytes=output.stat().st_size,
                files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def main():
    entries = [archive('scope-weights.tar.gz', [ROOT / 'weights/scope.pth']),
               archive('scope-real275-demo.tar.gz', sorted(p for p in (ROOT / 'data/real275_demo').rglob('*') if p.is_file()))]
    manifest_path = ROOT / 'assets.json'
    if manifest_path.exists():
        old = {a['filename']: a['url'] for a in json.loads(manifest_path.read_text())['assets']}
        for entry in entries:
            entry['url'] = old.get(entry['filename'], '')
    manifest_path.write_text(json.dumps(dict(assets=entries), indent=2) + '\n')
    for entry in entries:
        print(f"{entry['filename']}: {entry['bytes']/2**20:.2f} MiB; SHA256 {entry['sha256']}")


if __name__ == '__main__':
    main()
