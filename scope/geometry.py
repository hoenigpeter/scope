"""RGB-D geometry. All camera coordinates use x right, y down, z forward, in meters.

Crop helpers adapted from the original SCOPE evaluation and ROS2 export.
Modified for the standalone release: finite-depth validation, explicit pixel
correspondences, deterministic sampling, and no redundant camera-axis flips.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def crop(image, bbox, size=160):
    """Evaluation's clipped, square-padded 1.5x crop; return inverse mapping."""
    x0, y0, x1, y1 = map(int, bbox)
    h, w = image.shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(f"Invalid exclusive xyxy box: {bbox} for {w}x{h}")
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    half = int(max(x1 - x0, y1 - y0) * 1.5) // 2
    left, top = max(cx - half, 0), max(cy - half, 0)
    right, bottom = min(cx + half, w), min(cy + half, h)
    side = max(right - left, bottom - top)
    dx, dy = (side - right + left) // 2, (side - bottom + top) // 2
    meta = dict(left=left, top=top, right=right, bottom=bottom,
                dx=dx, dy=dy, side=side, size=size)
    return resize_crop(image, meta, Image.Resampling.BILINEAR), meta


def resize_crop(image, meta, interpolation=Image.Resampling.NEAREST):
    m = meta
    canvas = np.zeros((m['side'], m['side']) + image.shape[2:], dtype=image.dtype)
    h, w = m['bottom'] - m['top'], m['right'] - m['left']
    canvas[m['dy']:m['dy']+h, m['dx']:m['dx']+w] = image[m['top']:m['bottom'], m['left']:m['right']]
    return np.asarray(Image.fromarray(canvas).resize((m['size'], m['size']), interpolation)).copy()


def restore(image, meta, shape):
    """Invert a crop with nearest-neighbor sampling, including clipped edges."""
    m = meta
    resized = np.asarray(Image.fromarray(image).resize((m['side'], m['side']), Image.Resampling.NEAREST))
    out = np.zeros(tuple(shape[:2]) + image.shape[2:], dtype=image.dtype)
    h, w = m['bottom'] - m['top'], m['right'] - m['left']
    out[m['top']:m['bottom'], m['left']:m['right']] = resized[m['dy']:m['dy']+h, m['dx']:m['dx']+w]
    return out


def backproject(depth, intrinsics, mask=None):
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= mask.astype(bool)
    rows, cols = np.where(valid)
    z = depth[rows, cols]
    fx, fy, cx, cy = intrinsics
    points = np.column_stack(((cols-cx)*z/fx, (rows-cy)*z/fy, z))
    return points, (rows, cols)


def normal_image(depth, intrinsics, mask):
    import open3d as o3d
    points, pixels = backproject(depth, intrinsics, mask)
    if len(points) < 10:
        raise ValueError('Fewer than 10 valid object depth pixels')
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.estimate_normals()
    cloud.normalize_normals()
    cloud.orient_normals_towards_camera_location(np.zeros(3))
    image = np.zeros((*depth.shape, 3), np.uint8)
    image[pixels] = ((np.asarray(cloud.normals) + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return image


def register(nocs, points, seed=0, max_points=500, noise_bound=0.02):
    """Fit camera = scale * (R @ signed_NOCS) + t using TEASER++."""
    import teaserpp_python as teaser
    from threadpoolctl import threadpool_limits
    # Match the paper's quantized, unique correspondence selection, but only
    # deduplicate valid object pixels (background cannot consume a valid pair).
    _, indices = np.unique(nocs, axis=0, return_index=True)
    indices = np.sort(indices)
    src = nocs[indices].astype(np.float64) / 127.5 - 1
    dst = points[indices]
    valid = np.any(np.abs(src + 1) > 5 / 255, axis=1)
    src, dst = src[valid], dst[valid]
    if len(src) < 10:
        raise ValueError('Fewer than 10 unique NOCS/depth correspondences')
    if len(src) > max_points:
        idx = np.random.default_rng(seed).choice(len(src), max_points, replace=False)
        src, dst = src[idx], dst[idx]
    # Parallel maximum-clique tie-breaking can select different inlier sets.
    # Limit OpenMP only while constructing and running the registration solver.
    with threadpool_limits(limits=1, user_api='openmp'):
        params = teaser.RobustRegistrationSolver.Params()
        params.cbar2 = 1
        params.noise_bound = noise_bound
        params.estimate_scaling = True
        params.rotation_estimation_algorithm = teaser.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        params.rotation_gnc_factor = 1.4
        params.rotation_max_iterations = 1000
        params.rotation_cost_threshold = 1e-12
        solver = teaser.RobustRegistrationSolver(params)
        solver.solve(src.T, dst.T)
        solution = solver.getSolution()
    rotation, translation, scale = solution.rotation, solution.translation, float(solution.scale)
    if not (np.isfinite(rotation).all() and np.isfinite(translation).all() and np.isfinite(scale) and scale > 0):
        raise ValueError('Registration returned an invalid similarity transform')
    residual = np.linalg.norm(scale * (src @ rotation.T) + translation - dst, axis=1)
    return rotation, translation, scale, dict(
        correspondences=len(src), residual_median_m=float(np.median(residual)),
        inlier_fraction=float(np.mean(residual <= noise_bound)), noise_bound_m=noise_bound)
