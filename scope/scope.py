import torch
import torch.nn.functional as F

from tqdm.auto import tqdm

from diffusers import (
    DDPMScheduler,
    DPMSolverSinglestepScheduler,
    UNet2DConditionModel,
)
from torchvision import transforms
from transformers import AutoModel
import torch.nn as nn

import torchvision.transforms as T

class SCOPE(nn.Module):

    def __init__(self, input_nc = 9, output_nc = 3, image_size=128, num_training_steps=1000, num_inference_steps=5):

        super(SCOPE, self).__init__()

        self.model = UNet2DConditionModel(
            sample_size=image_size,
            in_channels=input_nc,
            out_channels=output_nc,
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
            cross_attention_dim=768,
        )

        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-base')

        self.train_noise_scheduler = DDPMScheduler(num_train_timesteps=num_training_steps)
        self.inference_noise_scheduler = DPMSolverSinglestepScheduler(num_train_timesteps=num_training_steps, algorithm_type="dpmsolver++", thresholding=True)

        self.num_training_steps = num_training_steps
        self.num_inference_steps = num_inference_steps

        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._transform = transforms.Compose([
            T.Resize((224, 224)),
            self.norm,
        ])

    def forward(self, rgb_image, normals_image, nocs_gt):
        noise = torch.randn(nocs_gt.shape, dtype=nocs_gt.dtype, device=nocs_gt.device)

        bsz = nocs_gt.shape[0]

        timesteps = torch.randint(
            0, self.train_noise_scheduler.config.num_train_timesteps, (bsz,), device=nocs_gt.device
        ).long()

        noisy_latents = self.train_noise_scheduler.add_noise(nocs_gt, noise, timesteps)

        latents = torch.cat([rgb_image, normals_image, noisy_latents], dim=1)

        embeddings = self.get_embeddings(rgb_image)

        model_output = self.model(latents, timesteps, embeddings).sample

        loss = F.mse_loss(model_output.float(), noise.float(), reduction="mean")

        return loss

    def get_embeddings(self, rgb_image):
        rgb_image = self._transform(((rgb_image + 1) / 2))

        with torch.no_grad():
            dino_embeddings = self.dino_model(rgb_image)

        dino_embeddings = dino_embeddings.last_hidden_state
        return dino_embeddings

    def inference(self, rgb_image, normals_image):
        self.inference_noise_scheduler.set_timesteps(self.num_inference_steps)

        embeddings = self.get_embeddings(rgb_image)

        nocs_noise = torch.randn(rgb_image.shape, dtype=rgb_image.dtype, device=rgb_image.device)

        for timestep in tqdm(self.inference_noise_scheduler.timesteps):

            input = torch.cat(
                [rgb_image, normals_image, nocs_noise], dim=1
            )

            with torch.no_grad():
                noisy_residual = self.model(input, timestep, embeddings).sample
            previous_noisy_sample = self.inference_noise_scheduler.step(noisy_residual, timestep, nocs_noise).prev_sample

            nocs_noise = previous_noisy_sample

        nocs_estimated = nocs_noise

        return nocs_estimated