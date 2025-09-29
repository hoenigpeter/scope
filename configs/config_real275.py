# General Training Settings
max_epochs = 100
batch_size = 1
test_batch_size = 1
shuffle_buffer = 1000
save_epoch_interval = 5
gpu_id = 0

fx = 591.0125
fy = 590.16775
cx = 322.525
cy = 244.11084
depth_scale_factor = 1.0

width = 640
height = 480

# Data Paths
train_data_root = "/ssd3/datasets_bop/camera_real275_datasets/camera_pbr_all_obj_oriented_normals/shard-{000000..000509}.tar"    # is different!
#train_data_root = "/ssd3/datasets_bop/camera_real275_datasets/real275_val_normals/scene_1.tar"
val_data_root = "/ssd3/datasets_bop/camera_real275_datasets/real275_val_normals/scene_1.tar"
test_data_root = val_data_root

coco_json_path = "/ssd3/real_camera_dataset/real275_test_3d_bbox.json"
test_images_root = "/ssd3/real_camera_dataset/real_test"

class_name = None
num_categories = 6
num_points_to_sample = 1000
minimum_points = 1
depth_mask_erosion_strength = 2

# Directories for Saving Weights and Validation Images
experiment_name = "REAL275_SCOPE_5_inf_1000_points_upright_random"
weight_dir = "./weights_MEGAPOSE_160px_dist_full"
val_img_dir = "./val_img_" + experiment_name
test_img_dir = "./test_img_" + experiment_name

ply_output_dir = "./plys_" + experiment_name
pkl_output_dir = "./pkls_" + experiment_name
png_output_dir = "./pngs_" + experiment_name
bboxes_output_dir = "./bboxes_" + experiment_name

weight_file = 'generator_checkpoint_epoch_24_0.pth'

noise_bound = 0.02  # 0.01
rotation_max_iterations = 1000 # 1000
rotation_cost_threshold = 1e-12  # 1e-12

##### REFINEMENT
refinement = True
num_refinement_steps = 10

# Input Data Settings
image_size = 160

num_training_steps = 1000   # watch out this parameter has also to be 1000 during inference
num_inference_steps = 5

# Optimizer Settings
lr = 1e-4
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
warmup_steps = 1000

# Dataloader Settings
train_num_workers = 2
val_num_workers = 1

# Augmentation Settings
augmentation = False
center_crop = False

# Visualization Settings
iter_cnt = 100
num_imgs_log = 8
