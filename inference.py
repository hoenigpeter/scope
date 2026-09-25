#!/usr/bin/env python3
"""Run SCOPE on one already-cropped RGB/normal image pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scope import SCOPE


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict a 160x160 NOCS map from RGB and surface normals."
    )
    parser.add_argument("--rgb", type=Path, required=True, help="RGB crop")
    parser.add_argument(
        "--normals",
        type=Path,
        required=True,
        help="Surface-normal crop encoded from [-1,1] to [0,255]",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        help="Optional binary object mask; pixels outside it are zeroed before normalization",
    )
    parser.add_argument("--output", type=Path, default=Path("nocs.png"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(__file__).resolve().parent / "weights" / "scope.pth",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _load_image(path: Path, interpolation: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize((SCOPE.image_size, SCOPE.image_size), interpolation)
    return np.asarray(image, dtype=np.float32).copy()


def _to_tensor(image: np.ndarray, mask: np.ndarray | None, device: torch.device) -> torch.Tensor:
    if mask is not None:
        image *= mask[..., None]
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor / 127.5 - 1.0


def main() -> None:
    args = _parser().parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    rgb = _load_image(args.rgb, Image.Resampling.BILINEAR)
    normals = _load_image(args.normals, Image.Resampling.NEAREST)
    mask = None
    if args.mask:
        mask_image = Image.open(args.mask).convert("L").resize(
            (SCOPE.image_size, SCOPE.image_size), Image.Resampling.NEAREST
        )
        mask = np.asarray(mask_image) > 0

    model = SCOPE.from_checkpoint(
        args.weights, device=device, num_inference_steps=args.steps
    )
    rng = torch.Generator(device=device).manual_seed(args.seed)
    prediction = model.predict(
        _to_tensor(rgb, mask, device),
        _to_tensor(normals, mask, device),
        generator=rng,
    )

    nocs = ((prediction[0].permute(1, 2, 0) + 1.0) / 2.0).clamp(0, 1)
    output = (nocs.cpu().numpy() * 255.0).round().astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="RGB").save(args.output)
    print(f"Saved NOCS map to {args.output}")


if __name__ == "__main__":
    main()
