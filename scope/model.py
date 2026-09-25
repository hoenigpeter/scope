"""SCOPE inference model.

This module intentionally contains only the published bottleneck
cross-attention architecture and its denoising loop.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, DPMSolverSinglestepScheduler, UNet2DConditionModel
from transformers import Dinov2Config, Dinov2Model


class SCOPE(nn.Module):
    """RGB- and normal-conditioned diffusion model for NOCS prediction."""

    image_size = 160

    def __init__(self, num_training_steps: int = 1000, num_inference_steps: int = 10):
        super().__init__()

        self.model = UNet2DConditionModel(
            sample_size=self.image_size,
            in_channels=9,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            mid_block_type="UNetMidBlock2DCrossAttn",
            cross_attention_dim=768,
        )

        # Construct DINOv2 locally. Its trained parameters are part of the SCOPE
        # checkpoint, so inference does not contact Hugging Face.
        dino_config = Dinov2Config(
            image_size=518,
            patch_size=14,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            mlp_ratio=4,
        )
        self.dino_model = Dinov2Model(dino_config)
        self.dino_model.eval()
        self.dino_model.requires_grad_(False)

        # Keep the training scheduler as a registered part of the published
        # architecture, even though only the inference scheduler is used here.
        self.train_noise_scheduler = DDPMScheduler(num_train_timesteps=num_training_steps)
        self.inference_noise_scheduler = DPMSolverSinglestepScheduler(
            num_train_timesteps=num_training_steps,
            algorithm_type="dpmsolver++",
            thresholding=True,
        )
        self.num_inference_steps = num_inference_steps

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cuda",
        num_inference_steps: int = 10,
    ) -> "SCOPE":
        model = cls(num_inference_steps=num_inference_steps)
        state_dict = torch.load(
            Path(checkpoint_path), map_location="cpu", weights_only=True, mmap=True
        )
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        return model

    def _dino_embeddings(self, rgb: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(
            (rgb + 1.0) / 2.0,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        mean = resized.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = resized.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        return self.dino_model((resized - mean) / std).last_hidden_state

    @torch.inference_mode()
    def predict(
        self,
        rgb: torch.Tensor,
        normals: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Predict NOCS maps from normalized BCHW tensors in [-1, 1]."""
        expected = (3, self.image_size, self.image_size)
        if rgb.ndim != 4 or tuple(rgb.shape[1:]) != expected:
            raise ValueError(f"rgb must have shape (B, {expected[0]}, {expected[1]}, {expected[2]})")
        if normals.shape != rgb.shape:
            raise ValueError("normals must have the same shape as rgb")

        self.inference_noise_scheduler.set_timesteps(
            self.num_inference_steps, device=rgb.device
        )
        embeddings = self._dino_embeddings(rgb)
        nocs = torch.randn(
            rgb.shape, dtype=rgb.dtype, device=rgb.device, generator=generator
        )

        for timestep in self.inference_noise_scheduler.timesteps:
            sample = torch.cat((rgb, normals, nocs), dim=1)
            residual = self.model(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=embeddings,
            ).sample
            nocs = self.inference_noise_scheduler.step(
                residual, timestep, nocs
            ).prev_sample

        return nocs
