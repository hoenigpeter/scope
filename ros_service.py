import os
import numpy as np
import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import time
import pickle
from scope.utils import restore_original_bbox_crop, paste_nocs_on_black_canvas, teaserpp_solve, \
                    backproject, sample_point_cloud, create_open3d_point_cloud, load_config, parse_args, \
                    filter_points, rotate_transform_matrix_180_z, erode_depth_mask, \
                    remove_duplicate_pixels, get_enlarged_bbox, crop_and_resize, project_pointcloud_to_image, create_pointnormals
                    
from scope.scope import SCOPE

import argparse

import cv2
import rospy
from std_msgs.msg import Header
from object_detector_msgs.msg import Pose3DBBoxWithConfidence
from object_detector_msgs.srv import estimate_poses_3dbbox, estimate_poses_3dbboxResponse
from geometry_msgs.msg import Pose, Vector3
from cv_bridge import CvBridge, CvBridgeError
import tf

from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

DEBUG = False

def o3d_to_ros(pcd, frame_id):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(points)

    data = np.hstack((points, colors))

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="r", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name="g", offset=16, datatype=PointField.FLOAT32, count=1),
        PointField(name="b", offset=20, datatype=PointField.FLOAT32, count=1),
    ]

    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    return pc2.create_cloud(header, fields, data)

class SCOPE_ROS_Wrapper:
    def __init__(self, config):
            if rospy.has_param('intrinsics'):
                cam = np.asarray(rospy.get_param('intrinsics'))
            else:
                cam = np.array([[config.fx, 0, config.cx],
                                [0, config.fy, config.cy],
                                [0, 0, 1]])
                
            self.frame_id = rospy.get_param('color_frame_id')
            self.depth_encoding = rospy.get_param('depth_encoding')
            self.depth_scale = rospy.get_param('depth_scale')

            self.camera_intrinsics = {
                'fx': cam[0][0],
                'fy': cam[1][1],
                'cx': cam[0][2],
                'cy': cam[1][2],
                'width': config.width,
                'height': config.height,
            }

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.generator = SCOPE(input_nc = 9, output_nc = 3, image_size=config.input_size, num_training_steps=config.num_training_steps, num_inference_steps=config.num_inference_steps)
            
            model_path = os.path.join(config.weight_dir, config.weight_file)
            state_dict = torch.load(model_path, map_location=self.device)

            self.generator.load_state_dict(state_dict["model_state_dict"])
            self.generator.to(self.device)
            self.generator.eval()

            self.service = rospy.Service("/estimate_poses_scope", estimate_poses_3dbbox, self.estimate_pose)
            print("Pose Estimation with SCOPE is ready.")

    def estimate_pose(self, req):
        detection = req.det
        rgb = req.rgb
        depth = req.depth

        width, height = rgb.width, rgb.height

        try:
            rgb_image = CvBridge().imgmsg_to_cv2(rgb, "bgr8")
        except CvBridgeError as e:
            print(e)

        try:
            depth.encoding = self.depth_encoding
            depth_img = CvBridge().imgmsg_to_cv2(depth, self.depth_encoding)
            depth_img = depth_img/int(self.depth_scale)

        except CvBridgeError as e:
            print(e)

        estimates = []

        print("detection.name: ", detection.name)
        mask = detection.mask
        mask = np.zeros((height, width), dtype=np.uint8)
        mask_ids = np.array(detection.mask)
        mask[np.unravel_index(mask_ids, (height, width))] = 255

        bbox = detection.bbox
        bbox_obj = [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]

        DEBUG and cv2.imwrite("./mask.png", mask)
        if config.depth_mask_erosion_strength > 0:
            print("Eroding depth mask with strength: ", config.depth_mask_erosion_strength)
            mask = erode_depth_mask(mask, config.depth_mask_erosion_strength)
        DEBUG and cv2.imwrite("./mask_eroded.png", mask)

        enlarged_bbox = get_enlarged_bbox(bbox_obj, rgb_image.shape, bbox_scaler=1.5)
        mask_cropped, metadata = crop_and_resize(mask, enlarged_bbox, bbox_obj, target_size=config.input_size, interpolation=Image.NEAREST)
        rgb_cropped, _ = crop_and_resize(rgb_image, enlarged_bbox, bbox_obj, target_size=config.input_size, interpolation=Image.BICUBIC)

        binary_mask = (torch.Tensor(mask_cropped) > 0).float()
        binary_mask = binary_mask.to(self.device)

        DEBUG and cv2.imwrite("./mask.png", mask)
        DEBUG and cv2.imwrite("./cropped_mask.png", mask_cropped)
        DEBUG and cv2.imwrite("./cropped_rgb.png", np.transpose(rgb_cropped, (1,2,0)))

        rgb_cropped = torch.clamp(torch.Tensor(rgb_cropped).float(), min=0.0, max=255.0).unsqueeze(0)
        rgb_cropped = rgb_cropped.to(self.device)
        rgb_cropped = rgb_cropped * binary_mask
        rgb_images_gt = (rgb_cropped.float() / 127.5) - 1

        dst, idxs = backproject(depth_img, self.camera_intrinsics, mask)
        dst = dst.T

        normals = create_pointnormals(dst)
        normals = project_pointcloud_to_image(dst.T, normals, self.camera_intrinsics['fx'], self.camera_intrinsics['fy'], self.camera_intrinsics['cx'], self.camera_intrinsics['cy'], (config.height, config.width, 3))
        mask_normals = np.all(normals == [0, 0, 0], axis=-1)
        normals = ((normals + 1) * 127.5).clip(0, 255).astype(np.uint8)
        normals[mask_normals] = [0, 0, 0]
        normal_images_crop, _ = crop_and_resize(normals, enlarged_bbox, bbox_obj, target_size=config.input_size, interpolation=Image.NEAREST)

        normal_images = torch.tensor(torch.Tensor(normal_images_crop), dtype=torch.float32)
        normal_images = torch.clamp(normal_images.float(), min=0.0, max=255.0).unsqueeze(0)
        normal_images = normal_images.to(self.device)

        normal_images = normal_images * binary_mask
        normal_images_gt = (normal_images.float() / 127.5) - 1

        mask_resized = restore_original_bbox_crop(mask_cropped, metadata)
        binary_mask = (mask_resized > 0).astype(np.uint8)

        DEBUG and cv2.imwrite("./normals_cropped.png", np.transpose(normal_images_crop, (1, 2, 0)))
        DEBUG and cv2.imwrite("./mask_resized.png", mask_resized)

        max_inliers = 0.0
        best_Rt = None
        best_R = None
        best_t = None
        best_s = None
        best_src_filtered = None

        dst = dst.T
        dst[:, 0] = -dst[:, 0]
        dst[:, 1] = -dst[:, 1]
        dst = dst.T

        for ref_step in range(config.num_refinement_steps):
            print("Refinement step: ", ref_step)

            nocs_estimated = self.generator.inference(rgb_images_gt, normal_images_gt)

            nocs_estimated = (nocs_estimated + 1) / 2

            nocs_estimated_np = (nocs_estimated).squeeze().permute(1, 2, 0).cpu().numpy()  # Convert to HWC

            DEBUG and cv2.imwrite("./nocs_estimated.png", (nocs_estimated_np * 255).astype(np.uint8))

            nocs_estimated_resized = restore_original_bbox_crop((nocs_estimated_np * 255).astype(np.uint8), metadata, interpolation=Image.NEAREST)

            nocs_estimated_resized_holes = remove_duplicate_pixels(nocs_estimated_resized)
            nocs_estimated_resized_masked = nocs_estimated_resized_holes.copy()
            nocs_estimated_resized_masked[binary_mask == 0] = 0

            nocs_full_np = paste_nocs_on_black_canvas((rgb_image * 255).astype(np.uint8), nocs_estimated_resized_masked.astype(np.uint8), bbox_obj)

            nocs_full_np = (nocs_full_np.astype(float) / 127.5) - 1
            src = nocs_full_np[idxs[0], idxs[1], :].T

            src_filtered, dst_filtered, _ = filter_points(src, dst, filter_value=-1.0, tolerance=5/255)

            if src_filtered.shape[1] > 4:

                num_points_to_sample = config.num_points_to_sample

                if num_points_to_sample > src_filtered.shape[1]:
                    src_filtered, dst_filtered = src_filtered, dst_filtered
                else:
                    src_filtered, dst_filtered = sample_point_cloud(src_filtered, dst_filtered, num_points_to_sample)

                R, t, s, R_inliers, t_inliers, s_inliers = teaserpp_solve(src_filtered, dst_filtered, config)
                R_inliers = float(R_inliers)/float(config.num_points_to_sample)
                t_inliers = float(t_inliers)/float(config.num_points_to_sample)
                num_total_pairs = (float(config.num_points_to_sample) * (float(config.num_points_to_sample) - 1)) / 2
                s_inliers = float(s_inliers)/num_total_pairs
                print("R_inliers: ", R_inliers)
                print("t_inliers: ", t_inliers)
                print("s_inliers: ", s_inliers)

                inliers = (R_inliers + t_inliers + s_inliers) / 3
                #inliers = s_inliers

                print("inliers: ", inliers)
                print()

                pred_RT = np.eye((4), dtype=np.float32) 
                pred_RT[:3, :3] = R
                pred_RT[:3, 3] = t

                if inliers > max_inliers:
                    print("new best inlier!")
                    max_inliers = inliers
                    best_Rt = pred_RT
                    best_s = s
                    best_src_filtered = src_filtered

            if config.refinement == False:
                break 

        if best_Rt is not None:
            pred_RTs_transformed = rotate_transform_matrix_180_z(best_Rt)
            src_transformed = best_s * np.matmul(pred_RTs_transformed[ 0:3,0:3 ], best_src_filtered) + pred_RTs_transformed[:3, 3].reshape(3, 1)
            pcd_src_transformed = create_open3d_point_cloud(src_transformed, [0, 1, 0])  # Green
            pub = rospy.Publisher("/pointcloud", PointCloud2, queue_size=1)
            msg = o3d_to_ros(pcd_src_transformed, self.frame_id)
            pub.publish(msg)

            min_coords = np.min(src_transformed, axis=1)
            max_coords = np.max(src_transformed, axis=1)
            size = max_coords - min_coords
            bbox = [bbox_obj[1], bbox_obj[0], bbox_obj[3], bbox_obj[2]]

            R_0 = np.eye(4,4)
            R_0[ 0:3,0:3 ] = pred_RTs_transformed[ 0:3,0:3 ]

            rot_quat = tf.transformations.quaternion_from_matrix(R_0)

            print("frame id: ", self.frame_id)
            br = tf.TransformBroadcaster()
            br.sendTransform((pred_RTs_transformed[:3, 3][0], pred_RTs_transformed[:3, 3][1], pred_RTs_transformed[:3, 3][2]),
                        rot_quat,
                        rospy.Time.now(),
                        f"pose_{detection.name}",
                        self.frame_id)

            estimate = Pose3DBBoxWithConfidence()
            estimate.name = detection.name
            estimate.confidence = detection.score
            estimate.pose = Pose()
            estimate.pose.position.x = pred_RTs_transformed[:3, 3][0]
            estimate.pose.position.y = pred_RTs_transformed[:3, 3][1]
            estimate.pose.position.z = pred_RTs_transformed[:3, 3][2]
            estimate.pose.orientation.x = rot_quat[0]
            estimate.pose.orientation.y = rot_quat[1]
            estimate.pose.orientation.z = rot_quat[2]
            estimate.pose.orientation.w = rot_quat[3]
            estimate.size = Vector3()
            estimate.size.x = size[0]
            estimate.size.y = size[1]
            estimate.size.z = size[2]
            estimates.append(estimate)

        response = estimate_poses_3dbboxResponse()
        response.poses = estimates
        rospy.loginfo(f"Estimated {estimates}")
        return response

if __name__ == "__main__":
    args = parse_args()

    config = load_config(args.config)

    try:
        rospy.init_node(f'SCOPE')
        SCOPE_ROS_Wrapper(config)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass