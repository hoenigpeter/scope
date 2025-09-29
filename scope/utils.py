# parts of the code from: https://github.com/hughw19/NOCS_CVPR2019/blob/master/detect_eval.py

import sys, os

from imgaug import augmenters as iaa
import imgaug.augmenters as iaa  # noqa
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch 
import webdataset as wds
import open3d as o3d
import io
import PIL.Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

import argparse
import importlib.util
from torchvision import transforms
import json
import imageio
from scipy.ndimage import binary_erosion

class CustomDataset(Dataset):
    def __init__(self, coco_json_path, root_dir, image_size=128, augment=False, center_crop=True, is_depth=False, depth_scale_factor=1.0, enlarge_factor=1.5):

        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        self.root_dir = root_dir
        self.image_size = image_size
        self.augment = augment
        self.center_crop = center_crop
        self.is_depth = is_depth
        self.depth_scale_factor = depth_scale_factor

        self.enlarge_factor = enlarge_factor

        #self.images = {img['id']: img for img in self.coco_data['images']}
        self.data = self.coco_data['data']
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        print(self.categories)
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_data = self.data[idx]
        frame_id = image_data['frame_id']
        scene_id = image_data['scene_id']

        predictions = image_data['predictions']
        gts = image_data['gts']

        rgb_filename = image_data['color_file_name']
        depth_filename = image_data['depth_file_name']

        img_path = os.path.join(self.root_dir, rgb_filename)
        depth_path = os.path.join(self.root_dir, depth_filename)

        rgb_image = Image.open(img_path).convert("RGB")
        rgb_image = np.array(rgb_image)

        with open(depth_path, 'rb') as f:
            depth_bytes = f.read()
        
        depth_image = imageio.imread(depth_bytes).astype(np.float32) / 1000
        depth_image *= self.depth_scale_factor
        print("depth_image.shape: ", depth_image.shape)
        print("max depth: ", np.max(depth_image))
        print("min depth: ", np.min(depth_image))

        rgb_crops = []
        mask_crops = []
        bboxes = []
        metadatas = []
        category_names = []
        category_ids = []
        instance_ids = []
        diffNOCS_ids = []
        masks = []
        scores = []

        for prediction in predictions:

            category_ids.append(int(prediction['category_id']))

            if 'instance_id' in prediction:
                instance_ids.append(int(prediction['instance_id']))

            if 'diffNOCS_id' in prediction:
                diffNOCS_ids.append(int(prediction['diffNOCS_id']))
                category_names.append(self.categories[prediction['diffNOCS_id']])
            else:
                category_names.append(self.categories[prediction['category_id']])

            scores.append(prediction['score'])

            bbox = np.array(prediction['bbox'], dtype=int)

            mask = self.decode_segmentation(prediction['segmentation'], image_data['width'], image_data['height'])
            mask = 1 - mask

            enlarged_bbox = get_enlarged_bbox(bbox, rgb_image.shape, bbox_scaler=self.enlarge_factor)

            rgb_crop, metadata = crop_and_resize(rgb_image, enlarged_bbox, bbox, target_size=self.image_size, interpolation=Image.BILINEAR)
            mask_crop, metadata = crop_and_resize(mask, enlarged_bbox, bbox, target_size=self.image_size, interpolation=Image.NEAREST)
            
            rgb_crops.append(torch.tensor(rgb_crop, dtype=torch.uint8))
            masks.append(torch.tensor(mask, dtype=torch.uint8))
            mask_crops.append(torch.tensor(mask_crop, dtype=torch.uint8))
            metadatas.append(metadata)
            bboxes.append(bbox)

        return {
            "frame_id": frame_id,
            "scene_id": scene_id,
            "rgb": transforms.ToTensor()(rgb_image),
            "depth": transforms.ToTensor()(depth_image),
            "masks": masks,
            "rgb_crops": rgb_crops,
            "mask_crops": mask_crops,
            "bboxes": bboxes,
            "metadatas": metadatas,
            "category_names": category_names,
            "category_ids": category_ids,
            "instance_ids": instance_ids,
            "scores": scores,
            "gts": gts,
        }

    def decode_segmentation(self, rle, width, height):
        mask = np.zeros(width * height, dtype=np.uint8)

        rle_counts = rle['counts']
        current_position = 0

        for i in range(len(rle_counts)):
            run_length = rle_counts[i]
            if i % 2 == 0:
                mask[current_position:current_position + run_length] = 1
            current_position += run_length

        return mask.reshape((height, width), order="F")

class COCODataset(Dataset):
    def __init__(self, coco_json_path, root_dir, image_size=128, augment=False, center_crop=True, is_depth=False, depth_scale_factor=1.0, enlarge_factor=1.5):

        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        self.root_dir = root_dir
        self.image_size = image_size
        self.augment = augment
        self.center_crop = center_crop
        self.is_depth = is_depth
        self.depth_scale_factor = depth_scale_factor

        self.enlarge_factor = enlarge_factor

        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = self.coco_data['annotations']

        if 'gts' in self.coco_data:
            self.gts = self.coco_data['gts']

        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_info = self.images[annotation['image_id']]
        category_name = self.categories[annotation['category_id']]
        category_id = annotation['category_id']

        if 'gts' in self.coco_data:
            gts = self.gts[annotation['image_id']]

        rgb_filename = img_info['file_name']
        depth_filename = rgb_filename.replace("rgb", "depth")
        depth_filename = depth_filename.replace("jpg", "png")

        #depth_filename = img_info['depth_file_name']

        img_path = os.path.join(self.root_dir, rgb_filename)
        depth_path = os.path.join(self.root_dir, "depth/" + depth_filename)

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        with open(depth_path, 'rb') as f:
            depth_bytes = f.read()
        
        depth_image = imageio.imread(depth_bytes).astype(np.float32) / 1000
        depth_image *= self.depth_scale_factor

        bbox = np.array(annotation['bbox'], dtype=int)

        mask = self.decode_segmentation(annotation['segmentation'], img_info['width'], img_info['height'])
        mask = 1 - mask

        enlarged_bbox = get_enlarged_bbox(bbox, image.shape, bbox_scaler=self.enlarge_factor)

        cropped_image, metadata = crop_and_resize(image, enlarged_bbox, bbox, target_size=self.image_size, interpolation=Image.BILINEAR)
        cropped_mask, metadata = crop_and_resize(mask, enlarged_bbox, bbox, target_size=self.image_size, interpolation=Image.NEAREST)
        
        if 'gts' not in self.coco_data:
            return {
                "rgb_crop": cropped_image,
                "mask_crop": cropped_mask,
                "rgb": transforms.ToTensor()(image),  # For later post-processing
                "depth": transforms.ToTensor()(depth_image),
                "bbox": bbox,
                "mask": torch.tensor(mask, dtype=torch.uint8),
                "metadata": metadata,
                "category_name": category_name,
                "category_id": category_id,
                "rgb_filename": rgb_filename,
            }
    
        return {
            "rgb_crop": cropped_image,
            "mask_crop": cropped_mask,
            "rgb": transforms.ToTensor()(image),  # For later post-processing
            "depth": transforms.ToTensor()(depth_image),
            "bbox": bbox,
            "mask": torch.tensor(mask, dtype=torch.uint8),
            "metadata": metadata,
            "category_name": category_name,
            "category_id": category_id,
            "gts": gts,
        }

    def decode_segmentation(self, rle, width, height):
        mask = np.zeros(width * height, dtype=np.uint8)

        rle_counts = rle['counts']
        current_position = 0

        for i in range(len(rle_counts)):
            run_length = rle_counts[i]
            if i % 2 == 0:
                mask[current_position:current_position + run_length] = 1
            current_position += run_length

        return mask.reshape((height, width), order="F")
    
def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('config', type=str, help="Path to the config file")
    return parser.parse_args()

def make_log_dirs(dir_list):
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)

def custom_collate_fn(batch):
    rgb_batch = torch.stack([torch.tensor(item[0]) for item in batch])
    normals_batch = torch.stack([torch.tensor(item[1]) for item in batch])
    mask_batch = torch.stack([torch.tensor(item[2]) for item in batch])
    nocs_batch = torch.stack([torch.tensor(item[3]) for item in batch])
    info_batch = [item[4] for item in batch]

    return {
        'rgb': rgb_batch,
        'normals': normals_batch,
        'mask': mask_batch,
        'nocs': nocs_batch,
        'info': info_batch,
    }

def collate_fn_val(batch):
    frame_id = [(item['frame_id']) for item in batch]
    scene_id = [(item['scene_id']) for item in batch]
    rgb_images = torch.stack([torch.tensor(item['rgb']) for item in batch])
    depth_images = torch.stack([torch.tensor(item['depth']) for item in batch])
    mask_images = [(item['masks']) for item in batch]
    rgb_crops = [(item['rgb_crops']) for item in batch]
    mask_crops = [(item['mask_crops']) for item in batch]
    bboxes = [(item['bboxes']) for item in batch]
    metadatas = [(item['metadatas']) for item in batch]
    category_names = [(item['category_names']) for item in batch]
    category_ids = [(item['category_ids']) for item in batch]
    scores = [(item['scores']) for item in batch]    
    gts = [(item['gts']) for item in batch]

    return {
        "frame_id": frame_id,
        "scene_id": scene_id,
        "rgb": rgb_images,
        "depth": depth_images,
        "mask": mask_images,
        "rgb_crops": rgb_crops,
        "mask_crops": mask_crops,
        "bboxes": bboxes,
        "metadatas": metadatas,
        "category_names": category_names,
        "category_ids": category_ids,
        "scores": scores,
        "gts": gts,
    }

def collate_fn(batch):
    rgb_images = torch.stack([torch.tensor(item['rgb']) for item in batch])
    depth_images = torch.stack([torch.tensor(item['depth']) for item in batch])
    rgb_cropped = torch.stack([torch.tensor(item['rgb_crop']) for item in batch])
    mask_cropped = torch.stack([torch.tensor(item['mask_crop']) for item in batch])
    mask_images = torch.stack([torch.tensor(item['mask']) for item in batch])
    bboxes = torch.stack([torch.tensor(item['bbox']) for item in batch])
    metadata = [(item['metadata']) for item in batch]
    category_name = [(item['category_name']) for item in batch]
    category_id = [(item['category_id']) for item in batch]

    if "gts" in batch[0]:
        gts = [(item['gts']) for item in batch]
        return {
            "rgb": rgb_images,
            "depth": depth_images,
            "rgb_crop": rgb_cropped,
            "mask_crop": mask_cropped,
            "mask": mask_images,
            "bbox": bboxes,
            "metadata": metadata,
            "category_name": category_name,
            "category_id": category_id,
            "gts": gts,
        }

    return {
        "rgb": rgb_images,
        "depth": depth_images,
        "rgb_crop": rgb_cropped,
        "mask_crop": mask_cropped,
        "mask": mask_images,
        "bbox": bboxes,
        "metadata": metadata,
        "category_name": category_name,
        "category_id": category_id,
    }

def post_process_crop_to_original(crop, original_image, bbox):
    # Resize crop back to bounding box size
    x_min, y_min, width, height = map(int, bbox)
    crop_resized = Image.fromarray(crop).resize((width, height), Image.BILINEAR)

    # Place resized crop back on original image
    original_image = np.array(original_image)
    original_image[y_min:y_min + height, x_min:x_min + width] = np.array(crop_resized)
    return original_image

def custom_collate_fn_test(batch):
    rgb_batch = torch.stack([torch.tensor(item[0]) for item in batch])
    mask_batch = torch.stack([torch.tensor(item[1]) for item in batch])
    nocs_batch = torch.stack([torch.tensor(item[2]) for item in batch])
    depth_batch = torch.stack([torch.tensor(item[3]) for item in batch])
    info_batch = [item[4] for item in batch]

    return {
        'rgb': rgb_batch,
        'mask': mask_batch,
        'nocs': nocs_batch,
        'metric_depth': depth_batch,
        'info': info_batch,
    }

def load_depth_image(depth_bytes):
    depth_image = Image.open(io.BytesIO(depth_bytes)).convert("I;16")
    depth_array = np.array(depth_image, dtype=np.float32)
    return depth_array

def load_image(data):
    with io.BytesIO(data) as stream:
        img = PIL.Image.open(stream)
        img.load()
        img = img.convert("rgb".upper())
        img = img.convert("RGB")
        return img

def create_webdataset_v2(dataset_paths, size=128, shuffle_buffer=1000, dino_mode="", augment=False, center_crop=False, class_name=None):

    dataset = wds.WebDataset(dataset_paths, shardshuffle=True) \
        .decode() \
        .shuffle(shuffle_buffer, initial=size) \
        .to_tuple("rgb.png", "nocs.png", "normals_no_aug.png", "normals_with_aug.png", "mask_visib.png", dino_mode + ".npy", "info.json") \
        .map_tuple( 
            lambda rgb: preprocess(image=load_image(rgb), size=size, interpolation=Image.BICUBIC, augment=augment), 
            lambda nocs: preprocess(image=load_image(nocs), size=size, interpolation=Image.NEAREST), 
            lambda normals_no_aug: preprocess(image=load_image(normals_no_aug), size=size, interpolation=Image.NEAREST, augment=True), 
            lambda normals_with_aug: preprocess(image=load_image(normals_with_aug), size=size, interpolation=Image.NEAREST, augment=True), 
            lambda mask: preprocess(image=load_image(mask), size=size, interpolation=Image.NEAREST),
            lambda dino_pca: preprocess(image=dino_pca, size=size, interpolation=Image.NEAREST),
            lambda info: info)
    return dataset
    
def create_webdataset(dataset_paths, size=128, shuffle_buffer=1000, augment=False):

    dataset = wds.WebDataset(dataset_paths, shardshuffle=True) \
        .decode() \
        .shuffle(shuffle_buffer, initial=size) \
        .to_tuple("rgb.png", "normals.png", "mask_visib.png", "nocs.png", "info.json") \
        .map_tuple( 
            lambda rgb: preprocess(image=load_image(rgb), size=size, interpolation=Image.BICUBIC, augment=augment), 
            lambda normals: preprocess(image=load_image(normals), size=size, interpolation=Image.NEAREST, augment=True), 
            lambda mask: preprocess(image=load_image(mask), size=size, interpolation=Image.NEAREST), 
            lambda nocs: preprocess(image=load_image(nocs), size=size, interpolation=Image.NEAREST), 
            lambda info: info)

    return dataset

def create_webdataset_megapose(dataset_paths, size=128, shuffle_buffer=1000, augment=False):

    dataset = wds.WebDataset(dataset_paths, shardshuffle=True) \
        .decode() \
        .shuffle(shuffle_buffer, initial=size) \
        .to_tuple("rgb.png", "normals.png", "mask.png", "nocs.png", "info.json") \
        .map_tuple( 
            lambda rgb: preprocess(image=load_image(rgb), size=size, interpolation=Image.BICUBIC, augment=augment), 
            lambda normals: preprocess(image=load_image(normals), size=size, interpolation=Image.NEAREST, augment=True), 
            lambda mask: preprocess(image=load_image(mask), size=size, interpolation=Image.NEAREST), 
            lambda nocs: preprocess(image=load_image(nocs), size=size, interpolation=Image.NEAREST), 
            lambda info: info)

    return dataset

def preprocess(image, size, interpolation, augment=False):
    
    img_array = np.array(image).astype(np.uint8)

    if img_array.shape[2] == 6:
        return img_array

    h, w = img_array.shape[0], img_array.shape[1]

    crop = min(h, w)
    img_array = img_array[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]

    if augment:
        seq_syn = iaa.Sequential([
                            iaa.Sometimes(0.3, iaa.CoarseDropout(p=0.2, size_percent=0.05)),
                            iaa.Sometimes(0.5, iaa.Dropout(p=(0.0, 0.1)))
                            ], random_order=True)

        img_array = seq_syn.augment_image(img_array)

    image = Image.fromarray(img_array)
    image = image.resize((size, size), resample=interpolation)
    img_array = np.array(image).astype(np.uint8)   

    return img_array

def setup_environment(gpu_id):
    if len(sys.argv) != 2:
        print("Usage: python3 train.py <gpu_id>")
        sys.exit()

    if gpu_id == '-1':
        gpu_id = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

def plot_progress_imgs(imgfn, rgb_images, normal_images, nocs_images_normalized_gt, nocs_estimated, mask_images, config):
    _,ax = plt.subplots(config.num_imgs_log, 5, figsize=(10,20))
    col_titles = ['RGB Image', 'Normals GT', 'NOCS GT', 'NOCS Estimated', 'Mask GT']
    
    # Add column titles
    for i, title in enumerate(col_titles):
        ax[0, i].set_title(title, fontsize=12)

    for i in range(config.num_imgs_log):
        ax[i, 0].imshow(((rgb_images[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0))
        ax[i, 1].imshow(((normal_images[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0))
        ax[i, 2].imshow(((nocs_images_normalized_gt[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0))
        ax[i, 3].imshow(((nocs_estimated[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0))
        ax[i, 4].imshow(((mask_images[i])).detach().cpu().numpy().transpose(1, 2, 0))

    plt.tight_layout()
    plt.savefig(imgfn, dpi=300)
    plt.close()

def plot_single_image(output_dir, iteration, nocs_estimated, plot_image=False):

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare filename with padded iteration (e.g., 00000001.png)
    imgfn = os.path.join(output_dir, f'{iteration:08d}.png')

    # Plot NOCS estimated map (scale to 0-1 for visualization)
    #plt.imshow(((nocs_estimated[0] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0))
    plt.imshow(nocs_estimated)
    plt.axis('off')  # Turn off axis for cleaner output
    if plot_image: plt.show()

    # Save the figure
    plt.savefig(imgfn, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def crop_and_resize(img, enlarged_bbox, original_bbox, target_size=128, interpolation=Image.NEAREST):

    crop_xmin, crop_ymin, crop_xmax, crop_ymax = enlarged_bbox
    orig_xmin, orig_ymin, orig_xmax, orig_ymax = original_bbox

    enlarged_size = max(crop_ymax - crop_ymin, crop_xmax - crop_xmin)
    
    # Initialize the cropped image with zeros
    if img.ndim == 3:
        cropped_img = np.zeros((enlarged_size, enlarged_size, img.shape[2]), dtype=img.dtype)
    else:
        cropped_img = np.zeros((enlarged_size, enlarged_size), dtype=img.dtype)
    
    # Calculate offsets to center the cropped area
    y_offset = (enlarged_size - (crop_ymax - crop_ymin)) // 2
    x_offset = (enlarged_size - (crop_xmax - crop_xmin)) // 2
    
    # Crop and pad the image
    if img.ndim == 3:
        cropped_img[y_offset:y_offset + (crop_ymax - crop_ymin), x_offset:x_offset + (crop_xmax - crop_xmin)] = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]
    else:
        cropped_img[y_offset:y_offset + (crop_ymax - crop_ymin), x_offset:x_offset + (crop_xmax - crop_xmin)] = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

    # Resize the image to the target size
    cropped_img_pil = Image.fromarray(cropped_img)
    cropped_img_pil = cropped_img_pil.resize((target_size, target_size), interpolation)
    cropped_img = np.array(cropped_img_pil)
    
    # Store metadata for restoring the original bounding box
    scale_factor = target_size / enlarged_size
    original_offset_x = orig_xmin - crop_xmin + x_offset
    original_offset_y = orig_ymin - crop_ymin + y_offset
    metadata = {
        "enlarged_bbox": enlarged_bbox,
        "scale_factor": scale_factor,
        "original_bbox_size": (orig_xmax - orig_xmin, orig_ymax - orig_ymin),
        "original_offset": (original_offset_x, original_offset_y),
    }

    # If the image has multiple channels, transpose for compatibility if needed
    if cropped_img.ndim == 3:
        cropped_img = np.transpose(cropped_img, (2, 0, 1))

    return cropped_img, metadata

def get_enlarged_bbox(bbox, img_shape, bbox_scaler=1.5):
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    center_x = (bbox[2] + bbox[0]) // 2
    center_y = (bbox[3] + bbox[1]) // 2
        
    enlarged_size = int(max(bbox_width, bbox_height) * bbox_scaler)
    print("enlarged_size: ", enlarged_size)

    crop_xmin = max(center_x - enlarged_size // 2, 0)
    crop_xmax = min(center_x + enlarged_size // 2, img_shape[1])
    crop_ymin = max(center_y - enlarged_size // 2, 0)
    crop_ymax = min(center_y + enlarged_size // 2, img_shape[0])

    return np.array([crop_xmin, crop_ymin, crop_xmax, crop_ymax])

def restore_original_bbox_crop(cropped_resized_img, metadata, interpolation=Image.NEAREST):

    scale_factor = metadata['scale_factor']
    original_bbox_size = metadata['original_bbox_size']
    original_offset = metadata['original_offset']

    enlarged_size = int(cropped_resized_img.shape[1] / scale_factor)

    cropped_img_pil = Image.fromarray(cropped_resized_img)
    restored_enlarged_img = cropped_img_pil.resize((enlarged_size, enlarged_size), interpolation)
    restored_enlarged_img = np.array(restored_enlarged_img)

    original_offset_x, original_offset_y = original_offset
    if restored_enlarged_img.ndim == 3:
        original_bbox_crop = restored_enlarged_img[
            original_offset_y:original_offset_y + original_bbox_size[1], 
            original_offset_x:original_offset_x + original_bbox_size[0], 
            :
        ]
    else:
        original_bbox_crop = restored_enlarged_img[
            original_offset_y:original_offset_y + original_bbox_size[1], 
            original_offset_x:original_offset_x + original_bbox_size[0]
        ]

    return original_bbox_crop

def overlay_nocs_on_rgb(full_scale_rgb, nocs_image, mask_image, bbox):
    binary_mask = (mask_image > 0).astype(np.uint8)
    masked_nocs = nocs_image.copy()
    masked_nocs[binary_mask == 0] = 0

    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox

    overlay_image = full_scale_rgb.copy()
    
    overlay_region = overlay_image[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax]
    overlay_image[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax] = np.where(
        mask_image[:, :, np.newaxis] > 0, masked_nocs, overlay_region
    )

    return overlay_image

def paste_mask_on_black_canvas(base_image, mask_image, bbox):

    canvas_shape = base_image.shape
    if base_image.ndim == 2:  # Single-channel image
        canvas_shape = (*canvas_shape, 1)  # Add a channel dimension
    canvas = np.zeros((canvas_shape[0], canvas_shape[1],1), dtype=base_image.dtype)

    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox

    if mask_image.ndim == 2:  # Single-channel mask
        mask_image = np.expand_dims(mask_image, axis=-1)  # Match dimensions

    # Only update pixels where mask is non-zero
    canvas[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax] = mask_image

    if base_image.ndim == 2:
        canvas = np.squeeze(canvas, axis=-1)

    return canvas

def paste_nocs_on_black_canvas(base_image, mask_image, bbox):

    canvas_shape = base_image.shape
    canvas = np.zeros((canvas_shape[0], canvas_shape[1], 3), dtype=base_image.dtype)

    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox

    canvas[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax] = mask_image

    return canvas

def combine_images_overlapping(images):
    if not images:
        raise ValueError("The list of images is empty.")
    
    # Ensure all images have the same shape
    img_height, img_width, img_channels = images[0].shape
    for img in images:
        if img.shape != (img_height, img_width, img_channels):
            raise ValueError("All images must have the same dimensions.")
    
    # Create a blank black canvas
    combined_image = np.zeros_like(images[0], dtype=np.uint8)
    
    # Overlay images by copying non-black pixels
    for img in images:
        mask = np.any(img != [0, 0, 0], axis=-1)  # Find where the image is not black
        combined_image[mask] = img[mask]
    
    return combined_image

def teaserpp_solve(src, dst, config):
    import teaserpp_python

    # Populate the parameters
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = config.noise_bound
    solver_params.estimate_scaling = True
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = config.rotation_max_iterations
    solver_params.rotation_cost_threshold = config.rotation_cost_threshold

    # Create the TEASER++ solver and solve the registration problem
    teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)

    teaserpp_solver.solve(src, dst)

    # Get the solution
    solution = teaserpp_solver.getSolution()
    R_inliers = teaserpp_solver.getRotationInliers()
    t_inliers = teaserpp_solver.getTranslationInliers()
    s_inliers = teaserpp_solver.getScaleInliers()
    #print("Solution is:", solution)

    # Extract rotation, translation, and scale from the solution
    R = solution.rotation
    t = solution.translation
    s = solution.scale

    return R, t, s, len(R_inliers), len(t_inliers), len(s_inliers)

def backproject(depth, intrinsics, instance_mask=None):
    intrinsics = np.array([[intrinsics['fx'], 0, intrinsics['cx']], [0, intrinsics['fy'], intrinsics['cy']],[0,0,1]])
    intrinsics_inv = np.linalg.inv(intrinsics)

    #non_zero_mask = np.logical_and(depth > 0, depth < 5000)
    non_zero_mask = (depth > 0)

    if instance_mask is not None:
        instance_mask = np.squeeze(instance_mask)
        final_instance_mask = np.logical_and(instance_mask, non_zero_mask)
    else:
        final_instance_mask = np.ones_like(depth, dtype=bool)

    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])

    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0) # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid # [3, num_pixel]
    xyz = np.transpose(xyz) #[num_pixel, 3]

    z = depth[idxs[0], idxs[1]]
    pts = xyz * z[:, np.newaxis]/xyz[:, -1:]
    # pts[:, 0] = -pts[:, 0]
    # pts[:, 1] = -pts[:, 1]

    return pts, idxs

def sample_point_cloud(src, dst, num_samples):
    if src.shape[1] < num_samples:
        raise ValueError("The number of samples exceeds the number of available points.")

    indices = np.random.choice(src.shape[1], num_samples, replace=False)

    return src[:, indices], dst[:, indices]

def create_line_set(src_points, dst_points, color=[0, 1, 0]):
    src_points = np.asarray(src_points, dtype=np.float64).T
    dst_points = np.asarray(dst_points, dtype=np.float64).T

    if src_points.shape[1] != 3 or dst_points.shape[1] != 3:
        raise ValueError("Points must have a shape of (N, 3)")

    lines = [[i, i + len(src_points)] for i in range(len(src_points))]

    line_set = o3d.geometry.LineSet()

    all_points = np.concatenate((src_points, dst_points), axis=0)
    line_set.points = o3d.utility.Vector3dVector(all_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    colors = [color] * len(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

def create_open3d_point_cloud(points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.T)
    pcd.paint_uniform_color(color)
    return pcd

def show_pointcloud(points, axis_size=1.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axis])

def filter_points(src, dst, filter_value=0, tolerance=1/255):
    mask = ~(np.max(np.abs(src - filter_value), axis=0) <= tolerance)
    indices_removed = np.where(~mask)[0]
    src_filtered = src[:, mask]
    dst_filtered = dst[:, mask]
    
    return src_filtered, dst_filtered, indices_removed

def rotate_transform_matrix_180_z(transform_matrix):
    z_180_RT = np.zeros((4, 4), dtype=np.float32)
    z_180_RT[:3, :3] = np.diag([-1, -1, 1])
    z_180_RT[3, 3] = 1

    RT = np.matmul(z_180_RT,transform_matrix)
    
    return RT

def remove_duplicate_pixels(img_array):
    processed_array = np.zeros_like(img_array)
    
    flat_array = img_array.reshape(-1, 3)
    _, unique_indices = np.unique(flat_array, axis=0, return_index=True)

    processed_array.reshape(-1, 3)[unique_indices] = flat_array[unique_indices]
    
    return processed_array

def project_pointcloud_to_image(pointcloud, pointnormals, fx, fy, cx, cy, image_shape):
    # Extract 3D points
    x, y, z = pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2]

    # Avoid division by zero
    z = np.where(z == 0, 1e-6, z)

    # Project points to the image plane
    u = (x * fx / z) + cx
    v = (y * fy / z) + cy

    # Round to nearest integer and convert to pixel indices
    u = np.round(u).astype(int)
    v = np.round(v).astype(int)

    # Create an empty image
    height, width, _ = image_shape
    image = np.zeros(image_shape, dtype=np.float32)

    # Keep points within image bounds
    valid_indices = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid_indices]
    v = v[valid_indices]
    normals = pointnormals[valid_indices]

    # Project valid points onto the image
    image[v, u] = normals  # Set pixel values to the point normals (nx, ny, nz)

    return image

def create_pointnormals(dst, invert=False):
    pcd_normals = create_open3d_point_cloud(dst, [1, 0, 0])
    o3d.geometry.PointCloud.estimate_normals(pcd_normals)
    pcd_normals.normalize_normals()
    pcd_normals.orient_normals_towards_camera_location()

    if invert:
        pcd_normals.normals = o3d.utility.Vector3dVector(-np.asarray(pcd_normals.normals))

    normals = np.asarray(pcd_normals.normals)
    return normals

def erode_depth_mask(mask_np, depth_mask_erosion_strength=2, threshold=127):
    """
    Erodes a binary mask. Accepts bool or uint8 inputs. For uint8, it applies a threshold.

    Parameters:
        mask_np (np.ndarray): Input binary mask (bool or uint8).
        depth_mask_erosion_strength (int): Number of pixels for erosion.
        threshold (int): Threshold to binarize uint8 mask.

    Returns:
        np.ndarray: Eroded mask with same dtype as input.
    """
    original_dtype = mask_np.dtype

    if original_dtype == np.bool_:
        mask_bool = mask_np
    elif original_dtype == np.uint8:
        mask_bool = mask_np > threshold
    else:
        raise TypeError(f"Unsupported dtype {original_dtype}. Only bool or uint8 are supported.")

    structure = np.ones((2 * depth_mask_erosion_strength + 1,) * 2, dtype=bool)
    shrunk_mask = binary_erosion(mask_bool, structure=structure, iterations=1)

    if original_dtype == np.uint8:
        return shrunk_mask.astype(np.uint8) * 255  # preserve mask scale
    return shrunk_mask