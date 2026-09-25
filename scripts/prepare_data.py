#!/usr/bin/env python3
"""Export a small deterministic REAL275 subset from the research detector JSON."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


def decode_rle(rle, height, width):
    counts = rle['counts']
    if not isinstance(counts, list) or any(not isinstance(n, int) or n < 0 for n in counts):
        raise ValueError('Expected uncompressed COCO RLE integer counts')
    if sum(counts) != height * width or rle['size'] != [height, width]:
        raise ValueError('RLE shape/count mismatch')
    return np.repeat(np.arange(len(counts)) % 2, counts).astype(np.uint8).reshape((height, width), order='F')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real-root', type=Path, required=True)
    p.add_argument('--detections', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('data/real275_demo'))
    p.add_argument('--frames-per-scene', type=int, default=3)
    args = p.parse_args()
    if args.frames_per_scene < 1:
        p.error('--frames-per-scene must be positive')
    if args.output.exists():
        p.error('Output already exists; choose a new directory to avoid stale samples')
    source = json.loads(args.detections.read_text())
    categories = {c['id']: c['name'] for c in source['categories']}
    args.output.mkdir(parents=True)
    frames = []
    scenes = sorted({d['scene_id'] for d in source['data']})
    for scene in scenes:
        candidates = sorted((d for d in source['data'] if d['scene_id'] == scene and d['predictions']), key=lambda d: d['frame_id'])
        indices = np.unique(np.linspace(0, len(candidates)-1, min(args.frames_per_scene, len(candidates)), dtype=int))
        for index in indices:
            entry = candidates[index]
            name = f"scene_{scene}/{int(entry['frame_id']):04d}"
            dest = args.output / name
            dest.mkdir(parents=True)
            for key, filename in [('color_file_name', 'rgb.png'), ('depth_file_name', 'depth.png')]:
                source_path = (args.real_root / entry[key]).resolve()
                if not source_path.is_relative_to(args.real_root.resolve()):
                    raise ValueError('Source path escapes dataset root')
                shutil.copyfile(source_path, dest / filename)
            objects = []
            for i, pred in enumerate(entry['predictions']):
                mask = decode_rle(pred['segmentation'], entry['height'], entry['width'])
                bbox = list(map(int, pred['bbox']))
                x0, y0, x1, y1 = bbox
                if not (0 <= x0 < x1 <= entry['width'] and 0 <= y0 < y1 <= entry['height']):
                    raise ValueError(f'Invalid detector box in {name}: {bbox}')
                if not mask[y0:y1, x0:x1].any():
                    raise ValueError(f'Empty detector mask in {name}')
                mask_name = f'mask_{i:02d}.png'
                Image.fromarray(mask * 255).save(dest / mask_name)
                objects.append(dict(id=i, category=categories[pred['category_id']],
                                    score=float(pred['score']), bbox_xyxy=bbox, mask=f'{name}/{mask_name}'))
            frames.append(dict(id=name.replace('/', '_'), scene=int(scene), frame=int(entry['frame_id']),
                               rgb=f'{name}/rgb.png', depth=f'{name}/depth.png', objects=objects))
    files = {str(f.relative_to(args.output)): hashlib.sha256(f.read_bytes()).hexdigest()
             for f in sorted(args.output.rglob('*.png'))}
    manifest = dict(format_version=1, dataset='REAL275 test', selection='Evenly spaced frames per scene; no quality filtering',
                    masks='Saved detector predictions from the research evaluation JSON; not ground-truth masks',
                    detections_sha256=hashlib.sha256(args.detections.read_bytes()).hexdigest(),
                    intrinsics=[591.0125, 590.16775, 322.525, 244.11084], depth_units_per_meter=1000,
                    frames=frames, sha256=files)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Prepared {len(frames)} frames, {sum(len(f["objects"]) for f in frames)} detections, '
          f'{sum(f.stat().st_size for f in args.output.rglob("*") if f.is_file()) / 2**20:.1f} MiB')


if __name__ == '__main__':
    main()
