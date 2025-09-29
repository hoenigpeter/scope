import os

import time

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import json
import webdataset as wds

from scope.utils import setup_environment, create_webdataset, create_webdataset_megapose, custom_collate_fn, make_log_dirs, plot_progress_imgs, parse_args, load_config
from scope.scope import SCOPE

def main(config):

    setup_environment(str(config.gpu_id))
    make_log_dirs([config.weight_dir, config.val_img_dir])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model instantiation and compilation
    generator = SCOPE(input_nc = 9, output_nc = 3, image_size=config.input_size, num_training_steps=config.num_training_steps, num_inference_steps=config.num_inference_steps)
    generator.to(device)
    print(generator)

    # Optimizer instantiation
    optimizer = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.epsilon)

    # Instantiate train and val dataset + dataloaders
    train_dataset = create_webdataset_megapose(config.train_data_root, config.input_size, config.shuffle_buffer, augment=config.augmentation)
    val_dataset = create_webdataset(config.val_data_root, config.input_size, config.shuffle_buffer, augment=False)

    train_dataloader = wds.WebLoader(
        train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.train_num_workers, drop_last=True, collate_fn=custom_collate_fn,
    )
    val_dataloader = wds.WebLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.val_num_workers, drop_last=True, collate_fn=custom_collate_fn,
    )

    # Training Loop
    epoch = 0
    iteration = 0
    loss_log = []

    for epoch in range(config.max_epochs):

        start_time_epoch = time.time()
        running_loss = 0.0 

        generator.train()

        # # Shuffle before epoch
        train_dataloader.unbatched().shuffle(1000).batched(config.batch_size)
        val_dataloader.unbatched().shuffle(1000).batched(config.batch_size)
       
        for step, batch in enumerate(train_dataloader):
            start_time_iteration = time.time()

            # Update learning rate with linear warmup
            if iteration < config.warmup_steps:
                lr = config.lr * (iteration / config.warmup_steps)
            else:
                lr = config.lr  # Use target learning rate after warmup
            
            # Update the optimizer learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # unwrap the batch
            rgb_images = batch['rgb']
            normal_images = batch['normals']
            mask_images = batch['mask']
            nocs_images = batch['nocs']

            # Normalize mask to be binary (0 or 1)
            binary_mask = (mask_images > 0).float()  # Converts mask to 0 or 1
            binary_mask = binary_mask.permute(0, 3, 1, 2).to(device)  # Make sure mask has same shape
            
            # RGB processing
            rgb_images = torch.clamp(rgb_images.float(), min=0.0, max=255.0)
            rgb_images = rgb_images.permute(0, 3, 1, 2)
            rgb_images = rgb_images.to(device)
            rgb_images = rgb_images * binary_mask
            rgb_images_gt = (rgb_images.float() / 127.5) - 1

            # Normals processing
            normal_images = torch.clamp(normal_images.float(), min=0.0, max=255.0)
            normal_images = normal_images.permute(0, 3, 1, 2)
            normal_images = normal_images.to(device)
            normal_images = normal_images * binary_mask
            normal_images_gt = (normal_images.float() / 127.5) - 1

            # MASK processing
            mask_images_gt = mask_images.float() / 255.0
            mask_images_gt = mask_images_gt.permute(0, 3, 1, 2)
            mask_images_gt = mask_images_gt.to(device)

            # NOCS processing
            nocs_images_float = nocs_images.float()
            nocs_images_normalized = (nocs_images_float / 127.5) - 1
            nocs_images_normalized = nocs_images_normalized.permute(0, 3, 1, 2)
            nocs_images_normalized_gt = nocs_images_normalized.to(device)

            # forward pass through generator
            loss = generator(rgb_images_gt, normal_images_gt, nocs_images_normalized_gt)
            
            # Loss backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Optimizer gradient update
            optimizer.step()
            elapsed_time_iteration = time.time() - start_time_iteration

            running_loss += loss.item()
            iteration += 1

            if (step + 1) % config.iter_cnt == 0:
                avg_loss = running_loss / config.iter_cnt

                elapsed_time_iteration = time.time() - start_time_iteration
                lr_current = optimizer.param_groups[0]['lr']
                print("Epoch {:02d}, Iter {:03d}, Loss: {:.4f}, lr_gen: {:.6f}, Time: {:.4f} seconds".format(
                    epoch, step, avg_loss, lr_current, elapsed_time_iteration))

                # Log to JSON
                loss_log.append({
                    "epoch": epoch,
                    "iteration": iteration,
                    "regression_nocs_loss": avg_loss,
                    "learning_rate": lr_current,
                    "time_per_100_iterations": elapsed_time_iteration
                })

                running_loss = 0

                embeddings = generator.get_embeddings(rgb_images_gt)
                nocs_estimated = generator.inference(rgb_images_gt, normal_images_gt, embeddings)

                imgfn = config.val_img_dir + "/{:03d}_{:03d}.jpg".format(epoch, iteration)
                plot_progress_imgs(imgfn, rgb_images_gt, normal_images_gt, nocs_images_normalized_gt, nocs_estimated, mask_images_gt, config)

            if (step + 1) % config.model_save_interval == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': generator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(config.weight_dir, f'generator_checkpoint_epoch_{epoch}_ {step}.pth'))

        elapsed_time_epoch = time.time() - start_time_epoch
        print("Time for the whole epoch: {:.4f} seconds".format(elapsed_time_epoch))

        generator.eval()
        running_loss = 0.0

        val_iter = 0
        start_time_epoch = time.time()

        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                start_time_iteration = time.time()

                # unwrap the batch
                rgb_images = batch['rgb']
                normal_images = batch['normals']
                mask_images = batch['mask']
                nocs_images = batch['nocs']
               
                # Normalize mask to be binary (0 or 1)
                binary_mask = (mask_images > 0).float()
                binary_mask = binary_mask.permute(0, 3, 1, 2).to(device)
                
                # RGB processing
                rgb_images = torch.clamp(rgb_images.float(), min=0.0, max=255.0)
                rgb_images = rgb_images.permute(0, 3, 1, 2)
                rgb_images = rgb_images.to(device)
                rgb_images = rgb_images * binary_mask
                rgb_images_gt = (rgb_images.float() / 127.5) - 1

                # Normals processing
                normal_images = torch.clamp(normal_images.float(), min=0.0, max=255.0)
                normal_images = normal_images.permute(0, 3, 1, 2)
                normal_images = normal_images.to(device)
                normal_images = normal_images * binary_mask
                normal_images_gt = (normal_images.float() / 127.5) - 1

                # MASK processing
                mask_images_gt = mask_images.float() / 255.0
                mask_images_gt = mask_images_gt.permute(0, 3, 1, 2)
                mask_images_gt = mask_images_gt.to(device)

                # NOCS processing
                nocs_images_float = nocs_images.float()
                nocs_images_normalized = (nocs_images_float / 127.5) - 1
                nocs_images_normalized = nocs_images_normalized.permute(0, 3, 1, 2)
                nocs_images_normalized_gt = nocs_images_normalized.to(device)

                # forward pass through generator
                loss = generator(rgb_images_gt, normal_images_gt, nocs_images_normalized_gt)

                elapsed_time_iteration = time.time() - start_time_iteration

                running_loss += loss.item()

                val_iter+=1

        avg_loss = running_loss / val_iter

        loss_log.append({
            "epoch": epoch,
            "val_regression_nocs_loss": avg_loss,
            "val_learning_rate": lr_current,
            "val_time_per_100_iterations": elapsed_time_iteration
        })
        
        elapsed_time_epoch = time.time() - start_time_epoch
        print("Time for the validation: {:.4f} seconds".format(elapsed_time_epoch))
        print("Val Loss: {:.4f}".format(avg_loss))

        embeddings = generator.get_embeddings(rgb_images_gt)
        nocs_estimated = generator.inference(rgb_images_gt, normal_images_gt, embeddings)

        imgfn = config.val_img_dir + "/val_{:03d}.jpg".format(epoch)
        plot_progress_imgs(imgfn, rgb_images_gt, normal_images_gt, nocs_images_normalized_gt, nocs_estimated, mask_images_gt, config)
        
        if epoch % config.save_epoch_interval == 0:
            # Save model and optimizer state
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(config.weight_dir, f'generator_checkpoint_epoch_{epoch}.pth'))

        # Save loss log to JSON after each epoch
        with open(os.path.join(config.weight_dir, "loss_log.json"), "w") as f:
            json.dump(loss_log, f, indent=4)

        epoch += 1

    # Save final checkpoint after training
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': generator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(config.weight_dir, f'generator_checkpoint_epoch_{epoch}.pth'))

if __name__ == "__main__":
    args = parse_args()
    
    config = load_config(args.config)
    
    main(config)