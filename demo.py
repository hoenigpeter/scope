#!/usr/bin/env python3
"""Run SCOPE on prepared REAL275 frames; save predictions and view RGB-D overlays."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import open3d as o3d

from scope import SCOPE
from scope.geometry import crop, resize_crop, restore, backproject, normal_image, register

ROOT = Path(__file__).resolve().parent
CHECKPOINT_SHA256 = '10956158ac89d6a0f5e80ce706c959274fb8221b95bdd169dcc14c2e460c2e6a'


def sha256(path):
    with path.open('rb') as f:
        digest = hashlib.sha256()
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def local_file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Asset path escapes data directory: {relative}')
    return path


def cloud(points, colors):
    result = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    result.colors = o3d.utility.Vector3dVector(colors)
    return result


def infer_object(model, rgb, depth, intrinsics, mask, bbox, seed, device):
    rgb_crop, meta = crop(rgb, bbox)
    mask_crop = resize_crop(mask.astype(np.uint8), meta) > 0
    # Reproject the resized mask as evaluation does. Restrict it to the detector
    # box; use every valid pixel for normals rather than randomly leaving holes.
    full_mask = restore(mask_crop.astype(np.uint8), meta, depth.shape).astype(bool)
    x0, y0, x1, y1 = bbox
    box_mask = np.zeros(depth.shape, bool)
    box_mask[y0:y1, x0:x1] = True
    full_mask &= box_mask
    normals = resize_crop(normal_image(depth, intrinsics, full_mask), meta)

    def tensor(image):
        image = image.copy()
        image[~mask_crop] = 0
        return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)[None].to(device) / 127.5 - 1

    prediction = model.predict(tensor(rgb_crop), tensor(normals), generator=torch.Generator(device=device).manual_seed(seed))
    nocs = ((prediction[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    nocs[~mask_crop] = 0
    full_nocs = restore(nocs, meta, depth.shape)
    points, pixels = backproject(depth, intrinsics, full_mask)
    rotation, translation, scale, quality = register(full_nocs[pixels], points, seed)
    canonical = full_nocs[pixels].astype(np.float64) / 127.5 - 1
    keep = np.any(np.abs(canonical + 1) > 5 / 255, axis=1)
    canonical = canonical[keep]
    aligned = scale * (canonical @ rotation.T) + translation
    return dict(rotation=rotation.tolist(), translation_m=translation.tolist(),
                scale_m_per_signed_nocs_unit=scale, **quality), nocs, aligned, (canonical + 1) / 2


def show(geometries, title, screenshot=None, gif=None, frames=48, fps=12, orbit_degrees=25):
    viewer = o3d.visualization.VisualizerWithKeyCallback()
    width, height = (640, 480) if gif is not None else (1280, 800)
    if not viewer.create_window(window_name=title, width=width, height=height):
        raise RuntimeError('Open3D could not create a window. Use --headless or run in a desktop session.')
    try:
        for geometry in geometries:
            viewer.add_geometry(geometry)
        options = viewer.get_render_option()
        options.background_color = np.ones(3)
        options.point_size = 2 if gif is not None else 3
        view = viewer.get_view_control()
        view.set_front([0, 0, -1])
        view.set_up([0, -1, 0])
        view.set_zoom(0.65 if gif is not None else 0.45)
        groups = {
            'S': [geometries[0]],
            'N': [g for g in geometries[1:] if isinstance(g, o3d.geometry.PointCloud)],
            'A': [g for g in geometries[1:] if isinstance(g, o3d.geometry.TriangleMesh)],
        }
        def toggle_group(items):
            visible = True
            def callback(vis):
                nonlocal visible
                for item in items:
                    if visible:
                        vis.remove_geometry(item, reset_bounding_box=False)
                    else:
                        vis.add_geometry(item, reset_bounding_box=False)
                visible = not visible
                return False
            return callback
        for key, items in groups.items():
            viewer.register_key_callback(ord(key), toggle_group(items))
        if gif is not None:
            gif.parent.mkdir(parents=True, exist_ok=True)
            images = []
            palette = None
            for index in range(frames):
                phase = 2 * np.pi * index / frames
                yaw = np.deg2rad(orbit_degrees) * np.sin(phase)
                pitch = np.deg2rad(5) * np.cos(phase)
                view.set_front([np.sin(yaw) * np.cos(pitch), np.sin(pitch),
                                -np.cos(yaw) * np.cos(pitch)])
                view.set_up([0, -1, 0])
                viewer.poll_events()
                viewer.update_renderer()
                pixels = np.asarray(viewer.capture_screen_float_buffer(do_render=True))
                rgb = np.rint(pixels * 255).clip(0, 255).astype(np.uint8)
                frame = Image.fromarray(rgb)
                if palette is None:
                    palette = frame.quantize(colors=256, dither=Image.Dither.NONE)
                    colors = np.array(palette.getpalette(), dtype=np.int32).reshape(-1, 3)
                    white_index = int(np.argmin(np.sum((colors - 255) ** 2, axis=1)))
                    colors[white_index] = 255
                    palette.putpalette(colors.ravel().tolist())
                indexed = frame.quantize(palette=palette, dither=Image.Dither.NONE)
                # Palette quantization may map white to a nearby off-white bin.
                white_mask = Image.fromarray((np.all(rgb == 255, axis=2) * 255).astype(np.uint8))
                indexed.paste(white_index, (0, 0), white_mask)
                images.append(indexed)
            images[0].save(gif, save_all=True, append_images=images[1:], loop=0,
                           duration=round(1000 / fps), disposal=2, background=white_index, optimize=False)
            print(f'Saved {frames}-frame orbit to {gif}', flush=True)
        elif screenshot is None:
            viewer.run()
        else:
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            for _ in range(10):
                viewer.poll_events()
                viewer.update_renderer()
            viewer.capture_screen_image(str(screenshot), do_render=True)
    finally:
        viewer.destroy_window()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'data/real275_demo')
    p.add_argument('--weights', type=Path, default=ROOT / 'weights/scope.pth')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/demo')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--verify-reference', action='store_true', help='Compare all 18 demo frames with the released reference poses')
    p.add_argument('--no-download', action='store_true', help='Require local assets; do not access the network')
    p.add_argument('--device', default='auto')
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--limit', type=int, help='Only run this many frames')
    p.add_argument('--scene', type=int, choices=range(1, 7))
    p.add_argument('--min-score', type=float, default=0.5)
    p.add_argument('--overlay-color', choices=['nocs', 'green'], default='nocs')
    args = p.parse_args()
    if args.verify_reference and (args.scene is not None or args.limit is not None or args.steps != 10 or args.seed != 0 or args.min_score != 0.5):
        p.error('--verify-reference requires the complete subset, 10 steps, seed 0 and min-score 0.5')
    if args.steps < 1 or (args.limit is not None and args.limit < 1):
        p.error('--steps and --limit must be positive')
    if not 0 <= args.min_score <= 1:
        p.error('--min-score must be in [0, 1]')
    if not args.headless and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        p.error('No desktop display detected; use --headless to save point clouds and poses')
    manifest_path = args.data / 'manifest.json'
    if not args.no_download:
        from scripts.fetch_assets import ensure_assets
        filenames = []
        if args.weights.resolve() == ROOT / 'weights/scope.pth':
            filenames.append('scope-weights.tar.gz')
        if args.data.resolve() == ROOT / 'data/real275_demo':
            filenames.append('scope-real275-demo.tar.gz')
        if filenames:
            ensure_assets(filenames=filenames)
    if not manifest_path.is_file() or not args.weights.is_file():
        p.error('Missing assets. Run bash scripts/fetch_assets.sh or supply local --weights and --data paths.')
    manifest = json.loads(manifest_path.read_text())
    if manifest['format_version'] != 1:
        p.error('Unsupported data manifest version')
    print('Verifying checkpoint and data checksums...', flush=True)
    weight_hash = sha256(args.weights)
    if weight_hash != CHECKPOINT_SHA256:
        p.error('Checkpoint SHA-256 does not match the published export')
    for filename, expected in manifest['sha256'].items():
        if sha256(local_file(args.data, filename)) != expected:
            p.error(f'Data checksum mismatch: {filename}')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f'Loading SCOPE on {device}; {args.steps} diffusion steps', flush=True)
    model = SCOPE.from_checkpoint(args.weights, device=device, num_inference_steps=args.steps)
    frames = [f for f in manifest['frames'] if args.scene is None or f['scene'] == args.scene]
    if args.limit:
        frames = frames[:args.limit]
    if not frames:
        p.error('No frames selected')
    args.output.mkdir(parents=True, exist_ok=True)
    run = dict(seed=args.seed, steps=args.steps, device=str(device), min_score=args.min_score,
               checkpoint_sha256=weight_hash, manifest_sha256=sha256(manifest_path),
               versions={n: importlib.metadata.version(n) for n in ['torch', 'numpy', 'open3d', 'transformers', 'diffusers', 'Pillow']},
               frames=[], failures=0)
    for frame in frames:
        print(f"Processing {frame['id']} ...", flush=True)
        folder = args.output / frame['id']
        folder.mkdir(parents=True, exist_ok=True)
        rgb = np.asarray(Image.open(local_file(args.data, frame['rgb'])).convert('RGB'))
        depth_raw = np.asarray(Image.open(local_file(args.data, frame['depth'])))
        if depth_raw.ndim != 2 or depth_raw.shape != rgb.shape[:2]:
            raise ValueError('Expected aligned RGB and single-channel uint16 depth')
        depth = depth_raw.astype(np.float64) / manifest['depth_units_per_meter']
        depth[depth_raw == 32001] = 0
        points, pixels = backproject(depth, manifest['intrinsics'])
        scene = cloud(points, rgb[pixels].astype(np.float64) / 255)
        o3d.io.write_point_cloud(str(folder / 'scene.ply'), scene)
        geometries = [scene]
        results = []
        for obj in frame['objects']:
            if obj['score'] < args.min_score:
                continue
            # Stable per object seed: filtering frames or scores cannot change it.
            object_seed = (args.seed + int(frame['scene']) * 1000000 + int(frame['frame']) * 100 + obj['id']) % (2**32)
            mask = np.asarray(Image.open(local_file(args.data, obj['mask'])).convert('L')) > 0
            start = time.perf_counter()
            try:
                pose, nocs, aligned, colors = infer_object(model, rgb, depth, manifest['intrinsics'], mask, obj['bbox_xyxy'], object_seed, device)
            except ValueError as exc:
                print(f"  {obj['category']} {obj['id']}: FAILED: {exc}", flush=True)
                results.append(dict(**obj, error=str(exc), seed=object_seed))
                run['failures'] += 1
                continue
            stem = f"object_{obj['id']:02d}_{obj['category']}"
            Image.fromarray(nocs).save(folder / f'{stem}_nocs.png')
            if args.overlay_color == 'green':
                colors = np.tile([0.1, 1.0, 0.2], (len(aligned), 1))
            overlay = cloud(aligned, colors)
            o3d.io.write_point_cloud(str(folder / f'{stem}_aligned.ply'), overlay)
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
            transform = np.eye(4)
            transform[:3, :3] = pose['rotation']
            transform[:3, 3] = pose['translation_m']
            axes.transform(transform)
            o3d.io.write_triangle_mesh(str(folder / f'{stem}_axes.ply'), axes)
            geometries.extend([overlay, axes])
            results.append(dict(**obj, **pose, seed=object_seed, elapsed_s=time.perf_counter()-start))
            print(f"  {obj['category']} {obj['id']}: residual {pose['residual_median_m']*1000:.1f} mm; inliers {pose['inlier_fraction']:.1%}", flush=True)
        (folder / 'poses.json').write_text(json.dumps(results, indent=2) + '\n')
        run['frames'].append(dict(id=frame['id'], objects=results))
        (args.output / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
        if not args.headless:
            show(geometries, f"SCOPE | {frame['id']} | S scene, N NOCS, A axes; Q next | axes: X red, Y green, Z blue")
    successes = sum('rotation' in obj for f in run['frames'] for obj in f['objects'])
    print(f"Saved {successes} poses; {run['failures']} failures to {args.output}", flush=True)
    if args.verify_reference:
        from scripts.verify_results import verify
        try:
            verify(args.output / 'run.json')
        except ValueError as exc:
            p.exit(1, f'{exc}\n')
    if run['failures'] or successes == 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
