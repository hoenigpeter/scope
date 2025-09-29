#! /usr/bin/env python3
import rospy
from object_detector_msgs.srv import detectron2_service_server, estimate_poses_3dbbox
from sensor_msgs.msg import Image

from std_msgs.msg import ColorRGBA

import numpy as np
import time

import cv2
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image

class PoseCalculator:
    def __init__(self):
        self.image_publisher = rospy.Publisher('image_with_roi', Image, queue_size=10)
        self.bridge = CvBridge()

        self.color_frame_id = rospy.get_param('color_frame_id')
        self.grasp_frame_id = rospy.get_param('grasp_frame_id')

        self.marker_id = 0

    def detect_objects(self, rgb):
        rospy.wait_for_service('detect_objects')
        try:
            detect_objects_service = rospy.ServiceProxy('detect_objects', detectron2_service_server)
            response = detect_objects_service(rgb)
            return response.detections.detections
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def estimate_object_poses(self, rgb, depth, detection):
        rospy.wait_for_service('estimate_poses_scope')
        try:
            estimate_poses_service = rospy.ServiceProxy('estimate_poses_scope', estimate_poses_3dbbox)
            response = estimate_poses_service(detection, rgb, depth)
            rospy.loginfo(f"RESPONSE: {response}")
            return response.poses
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
    
    def publish_annotated_image(self, rgb, detections):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        height, width, _ = cv_image.shape

        overlay = cv_image.copy()

        for detection in detections:
            xmin = int(detection.bbox.xmin)
            ymin = int(detection.bbox.ymin)
            xmax = int(detection.bbox.xmax)
            ymax = int(detection.bbox.ymax)

            font_size = 1.0
            line_size = 2

            # Draw bounding box
            cv2.rectangle(cv_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), line_size)

            class_name = detection.name
            score = detection.score
            label = f"{class_name}: {score:.2f}"
            cv2.putText(cv_image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), line_size)

            # Reconstruct mask from flat index list
            mask_indices = np.array(detection.mask, dtype=np.int32)
            mask = np.zeros((height * width), dtype=np.uint8)
            mask[mask_indices] = 1
            mask = mask.reshape((height, width))

            # Generate a consistent color
            color = tuple((hash(class_name) % 256, (hash(class_name + 'a') % 256), (hash(class_name + 'b') % 256)))
            color = np.array(color, dtype=np.uint8)

            # Blend color into the overlay where mask is active
            alpha = 0.5
            mask_3c = np.stack([mask] * 3, axis=-1)  # Shape (H, W, 3)

            # Only update where mask is 1
            overlay = np.where(mask_3c, (alpha * color + (1 - alpha) * overlay).astype(np.uint8), overlay)

        # Blend overlay onto original image
        cv2.addWeighted(overlay, 0.5, cv_image, 0.5, 0, cv_image)

        # Publish annotated image
        annotated_image_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        self.image_publisher.publish(annotated_image_msg)


    def publish_3dbbox_marker(self, size, quat, t_est, color=(0.0, 1.0, 0.0), frame_id=None):
        """
        Publishes a 3D bounding box as a wireframe marker in RViz.

        Args:
            size: geometry_msgs/Vector3 or (x, y, z) tuple in meters.
            quat: (x, y, z, w) quaternion orientation.
            t_est: (x, y, z) position in meters.
            color: Optional RGBA tuple (default: green).
            frame_id: Optional override for self.grasp_frame_id.
        """

        vis_pub = rospy.Publisher("/3dbbox_estimated", Marker, queue_size=1, latch=True)
        marker = Marker()
        marker.header.frame_id = self.color_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "3dbbox"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = t_est[0]
        marker.pose.position.y = t_est[1]
        marker.pose.position.z = t_est[2]

        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]

        # Set object size
        if isinstance(size, tuple) or isinstance(size, np.ndarray):
            marker.scale.x = size[0]
            marker.scale.y = size[1]
            marker.scale.z = size[2]
        else:
            marker.scale.x = size.x
            marker.scale.y = size.y
            marker.scale.z = size.z

        # Set color
        marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=0.3)

        # marker.lifetime = rospy.Duration(0)  # 0 = forever

        rospy.loginfo(f"Publishing 3D bounding box marker at {t_est} with size {size} and orientation {quat} in frame {marker.header.frame_id}")
        vis_pub.publish(marker)

if __name__ == "__main__":
    rospy.init_node("calculate_poses")
    try:
        pose_calculator = PoseCalculator()
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():

            # get RGB and Depth messages from the topics
            rgb = rospy.wait_for_message(rospy.get_param('color_topic'), Image)
            depth = rospy.wait_for_message(rospy.get_param('depth_topic'), Image)

            # ###############################
            # DETECTION EXAMPLE
            # ###############################

            t0 = time.time()
            detections = pose_calculator.detect_objects(rgb)
            time_detections = time.time() - t0

            if detections is not None:
                pose_calculator.publish_annotated_image(rgb, detections)
                for detection in detections:
                    print(detection.name)

            print()
            print("... received object detection.")

            t0 = time.time()
            if detections is None or len(detections) == 0:
                print("nothing detected")
            else:

                # ###############################
                # POSE ESTIMATION EXAMPLE
                # ###############################

                estimated_poses_camFrame = []
                object_names = []

                try:
                    for detection in detections:
                        estimated_pose = pose_calculator.estimate_object_poses(rgb, depth, detection)[0]
                        R = np.array([estimated_pose.pose.orientation.x, estimated_pose.pose.orientation.y,  estimated_pose.pose.orientation.z, estimated_pose.pose.orientation.w])
                        t = np.array([estimated_pose.pose.position.x, estimated_pose.pose.position.y, estimated_pose.pose.position.z])
                        size = np.array([estimated_pose.size.x, estimated_pose.size.y, estimated_pose.size.z])
                        pose_calculator.publish_3dbbox_marker(size, R, t)
                        estimated_poses_camFrame.append(estimated_pose)
                        object_names.append(detection.name)
                     
                except Exception as e:
                    rospy.logerr(f"{e}")

                time_object_poses = time.time() - t0

                print()
                rate.sleep()

    except rospy.ROSInterruptException:
        pass

