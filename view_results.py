#!/usr/bin/env python3
"""Reopen saved scene/overlay/axes without running inference again."""
import argparse
from pathlib import Path
import open3d as o3d
from demo import show


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frame_directory', type=Path)
    output = parser.add_mutually_exclusive_group()
    output.add_argument('--screenshot', type=Path, help='Render one image and close; requires OpenGL/display')
    output.add_argument('--gif', type=Path, help='Render a looping camera orbit; requires OpenGL/display')
    parser.add_argument('--frames', type=int, default=48, help='Animation frames (default: 48)')
    parser.add_argument('--fps', type=int, default=12, help='Animation frame rate (default: 12)')
    parser.add_argument('--orbit-degrees', type=float, default=25, help='Maximum yaw each side of the camera view')
    args = parser.parse_args()
    if args.frames < 4 or not 1 <= args.fps <= 50 or not 0 < args.orbit_degrees <= 75:
        parser.error('Use at least 4 frames, 1–50 fps and an orbit angle in (0, 75] degrees')
    if not (args.frame_directory / 'scene.ply').is_file():
        parser.error('Frame directory must contain scene.ply')
    geometries = [o3d.io.read_point_cloud(str(args.frame_directory / 'scene.ply'))]
    geometries += [o3d.io.read_point_cloud(str(p)) for p in sorted(args.frame_directory.glob('*_aligned.ply'))]
    geometries += [o3d.io.read_triangle_mesh(str(p)) for p in sorted(args.frame_directory.glob('*_axes.ply'))]
    show(geometries, f'SCOPE | {args.frame_directory.name}', screenshot=args.screenshot,
         gif=args.gif, frames=args.frames, fps=args.fps, orbit_degrees=args.orbit_degrees)


if __name__ == '__main__':
    main()
