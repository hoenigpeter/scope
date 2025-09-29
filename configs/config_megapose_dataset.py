# General Training Settings
max_epochs = 100
batch_size = 48
test_batch_size = 1
shuffle_buffer = 1000
save_epoch_interval = 1
model_save_interval = 30000

gpu_id = 0

distributed = True

# Data Paths
train_data_root = "/scratch/peter.hoenig/datasets/megapose_nocs_cropped/{00000000..00002079}.tar"    # is different!
#train_data_root = "/scratch/peter.hoenig/datasets/megapose_nocs_cropped_test/{00000000..00000003}.tar"    # is different!
val_data_root = "/scratch/peter.hoenig/datasets/real275_val_normals/scene_1.tar"
test_data_root = val_data_root

# Directories for Saving Weights and Validation Images
experiment_name = "MEGAPOSE_160px_dist_full"
weight_dir = "/share/peter.hoenig/scope/weights_MEGAPOSE_160px_dist_full"
val_img_dir = "./val_img_" + experiment_name
test_img_dir = "./test_img_" + experiment_name

ply_output_dir = "./plys_" + experiment_name
pkl_output_dir = "./pkls_" + experiment_name
png_output_dir = "./pngs_" + experiment_name
bboxes_output_dir = "./bboxes_" + experiment_name

# Input Data Settings
image_size = 128

num_training_steps = 1000   # watch out this parameter has also to be 1000 during inference
num_inference_steps = 5

use_pre_trained = True
#weight_file = 'generator_epoch_12_0.pth'

# Optimizer Settings
lr = 1e-4
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
warmup_steps = 1000

# Dataloader Settings
train_num_workers = 4
val_num_workers = 1

# Augmentation Settings
augmentation = False

# Visualization Settings
iter_cnt = 100
num_imgs_log = 8
