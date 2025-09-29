import os

import time

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import json
import webdataset as wds

from scope.utils import create_webdataset_dist, setup_environment, create_webdataset, create_webdataset_megapose, custom_collate_fn, make_log_dirs, plot_progress_imgs, parse_args, load_config
from scope.scope import SCOPE

# def setup_model_and_optimizer(config, device):
#     # Instantiate model

#     if config.use_pre_trained:
#         config.weight_dir

#     generator = SCOPE(
#         input_nc=9,
#         output_nc=3,
#         image_size=config.image_size,
#         num_training_steps=config.num_training_steps,
#         num_inference_steps=config.num_inference_steps,
#     ).to(device)

#     # Create optimizer
#     optimizer = torch.optim.Adam(
#         generator.parameters(),
#         lr=config.lr,
#         betas=(config.beta1, config.beta2),
#         eps=config.epsilon,
#     )

#     # Wrap in DDP if in distributed mode
#     if config.distributed:
#         generator = DDP(generator, device_ids=[device.index], output_device=device.index)

#     return generator, optimizer

def find_latest_checkpoint(weight_dir):
    checkpoint_files = [
        f for f in os.listdir(weight_dir)
        if f.startswith("generator_checkpoint") and f.endswith(".pth")
    ]
    if not checkpoint_files:
        return None
    # Sort by epoch and step if available (e.g., epoch_3_60000 > epoch_3_30000 > epoch_3)
    def sort_key(f):
        parts = f.replace(".pth", "").split("_")
        nums = [int(p) for p in parts if p.isdigit()]
        return nums
    checkpoint_files.sort(key=sort_key, reverse=True)
    return os.path.join(weight_dir, checkpoint_files[0])

def setup_model_and_optimizer(config, device):
    generator = SCOPE(
        input_nc=9,
        output_nc=3,
        image_size=config.image_size,
        num_training_steps=config.num_training_steps,
        num_inference_steps=config.num_inference_steps,
    ).to(device)

    optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
    )

    start_epoch = 0
    start_step = 0

    if config.use_pre_trained and config.weight_dir:
        latest_ckpt = find_latest_checkpoint(config.weight_dir)
        if latest_ckpt and os.path.exists(latest_ckpt):
            print(f"Loading checkpoint: {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            generator.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            start_step = checkpoint.get("step", 0)
            print(f"Checkpoint loaded successfully from epoch {start_epoch}, step {start_step}.")
        else:
            print(f"No checkpoint found in {config.weight_dir}. Starting from scratch.")

    if config.distributed:
        generator = DDP(generator, device_ids=[device.index], output_device=device.index)

    return generator, optimizer, start_epoch, start_step

def setup_distributed(config):
    """
    Initializes the distributed training environment.
    Assumes launch via torchrun or torch.multiprocessing.spawn.
    """

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        config.rank = int(os.environ["RANK"])
        config.world_size = int(os.environ["WORLD_SIZE"])
        config.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
        config.rank = 0
        config.world_size = 1
        config.local_rank = 0

    config.distributed = config.world_size > 1

    if config.distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(config.local_rank)
        config.device = torch.device("cuda", config.local_rank)
        dist.barrier()
    else:
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_device(config):
    if config.distributed:
        # Each process uses its local GPU
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_shards_for_rank(all_shards, rank, world_size):
    """Distributes shards evenly across ranks"""
    return all_shards[rank::world_size]

def process_batch(batch, device):
    rgb = torch.clamp(batch['rgb'].float(), 0, 255).permute(0, 3, 1, 2).to(device)
    normals = torch.clamp(batch['normals'].float(), 0, 255).permute(0, 3, 1, 2).to(device)
    mask = batch['mask'].float().permute(0, 3, 1, 2).to(device)
    nocs = batch['nocs'].float().permute(0, 3, 1, 2).to(device)

    binary_mask = (mask > 0).float()

    rgb = rgb * binary_mask
    normals = normals * binary_mask

    rgb = (rgb / 127.5) - 1
    normals = (normals / 127.5) - 1
    mask = mask / 255.0
    nocs = (nocs / 127.5) - 1

    return rgb, normals, mask, nocs

def train_step(generator, optimizer, batch, device, iteration, config):
    generator.train()
    rgb, normals, mask, nocs = process_batch(batch, device)

    if iteration < config.warmup_steps:
        lr = config.lr * (iteration / config.warmup_steps)
    else:
        lr = config.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    loss = generator(rgb, normals, nocs)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), lr, rgb, normals, mask, nocs

def run_validation(generator, val_dataloader, device):
    generator.eval()
    total_loss = 0.0
    iterations = 0

    with torch.no_grad():
        for batch in val_dataloader:
            rgb, normals, mask, nocs = process_batch(batch, device)
            loss = generator(rgb, normals, nocs)
            total_loss += loss.item()
            iterations += 1

    return total_loss / iterations

def save_model_step(model, optimizer, epoch, step, save_dir):
    model_state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
    }

    os.makedirs(save_dir, exist_ok=True)
    torch.save(checkpoint, os.path.join(save_dir, f'generator_checkpoint_epoch_{epoch}_{step}.pth'))

def save_model(model, optimizer, epoch, save_dir):
    model_state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
    }

    os.makedirs(save_dir, exist_ok=True)
    torch.save(checkpoint, os.path.join(save_dir, f'generator_checkpoint_epoch_{epoch}.pth'))

def log_and_plot(loss_log, epoch, iteration, loss, lr, generator, rgb, normals, mask, nocs, config, is_val=False):
    suffix = f"val_{epoch:03d}" if is_val else f"{epoch:03d}_{iteration:03d}"
    imgfn = os.path.join(config.val_img_dir, f"{suffix}.jpg")

    with torch.no_grad():
        nocs_est = generator.module.inference(rgb, normals)

    plot_progress_imgs(imgfn, rgb, normals, nocs, nocs_est, mask, config)

    entry = {
        "epoch": epoch,
        "iteration" if not is_val else "val_iteration": iteration if not is_val else 0,
        "regression_nocs_loss" if not is_val else "val_regression_nocs_loss": loss,
        "learning_rate" if not is_val else "val_learning_rate": lr,
    }
    loss_log.append(entry)

import re

def count_tars(path_pattern):
    match = re.search(r"\{(\d+)\.\.(\d+)\}", path_pattern)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        return end - start + 1
    else:
        raise ValueError("No brace expansion pattern found.")

def main(config):
    setup_distributed(config)
    device = get_device(config)

    generator, optimizer, start_epoch, _ = setup_model_and_optimizer(config, device)

    iteration = 0
    loss_log = []

    val_dataset = create_webdataset_dist(config, augment=False)
    val_dataloader = wds.WebLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=1, drop_last=True, collate_fn=custom_collate_fn,
    )

    if dist.get_rank() == 0:
        make_log_dirs([config.weight_dir, config.val_img_dir])

        num_tars = float(count_tars(config.train_data_root))
        num_of_samples = num_tars * 21000.
        world_size = float(os.environ["WORLD_SIZE"])
        num_steps_per_epoch = (num_of_samples / float(config.batch_size) ) / world_size
        print(f"num_of_samples: {num_of_samples}")
        print(f"batch size per gpu: {config.batch_size}")
        print(f"num of gpus: {world_size}")
        print(f"num_steps_per_epoch: {num_steps_per_epoch}")
        print()

    for epoch in range(start_epoch, config.max_epochs):
        train_dataset = create_webdataset_megapose(config, augment=config.augmentation)

        train_dataloader = wds.WebLoader(
            train_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=1, drop_last=True, collate_fn=custom_collate_fn,
        )

        #train_dataloader = train_dataloader.unbatched().shuffle(1000).batched(config.batch_size)

        # if dist.get_rank() == 0:
        #     val_dataloader = val_dataloader.unbatched().shuffle(1000).batched(config.batch_size)
       
        generator.train()

        for step, batch in enumerate(train_dataloader):
            loss, lr, rgb, normals, mask, nocs = train_step(generator, optimizer, batch, device, iteration, config)
            iteration += 1

            if dist.get_rank() == 0 and step % 10 == 0:
                print(f"epoch: {epoch} step: {step}/{int(num_steps_per_epoch)} loss: {loss} lr: {lr}")

            if (step + 1) % config.iter_cnt == 0 and dist.get_rank() == 0:
                log_and_plot(loss_log, epoch, iteration, loss, lr, generator, rgb, normals, mask, nocs, config)
            
            if dist.get_rank() == 0 and step % config.model_save_interval == 0:
                save_model_step(generator, optimizer, epoch, step, config.weight_dir)

        if dist.get_rank() == 0:
            val_loss = run_validation(generator, val_dataloader, device)
            log_and_plot(loss_log, epoch, iteration, val_loss, lr, generator, rgb, normals, mask, nocs, config, is_val=True)

            if epoch % config.save_epoch_interval == 0:
                save_model(generator, optimizer, epoch, config.weight_dir)

            with open(os.path.join(config.weight_dir, "loss_log.json"), "w") as f:
                json.dump(loss_log, f, indent=4)

    if dist.get_rank() == 0:
        save_model(generator, optimizer, config.max_epochs, config.weight_dir)

if __name__ == "__main__":
    args = parse_args()
    
    config = load_config(args.config)
    
    main(config)