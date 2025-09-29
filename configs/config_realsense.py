fx = 606.6173706054688
fy = 605.2778930664062
cx = 322.375
cy = 232.67811584472656
 
width = 640
height = 480

num_points_to_sample = 1000
minimum_points = 10
depth_mask_erosion_strength = 0

weight_dir = "./weights_MEGAPOSE_160px_dist_full/"
weight_file = 'generator_checkpoint_epoch_24_0.pth'

noise_bound = 0.02  # 0.01
rotation_max_iterations = 1000 # 1000
rotation_cost_threshold = 1e-12  # 1e-12

##### REFINEMENT
refinement = True
num_refinement_steps = 6

# Input Data Settings
input_size = 128

num_training_steps = 1000   # watch out this parameter has also to be 1000 during inference
num_inference_steps = 5