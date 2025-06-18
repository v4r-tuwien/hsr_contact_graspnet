#!/usr/bin/env python

import sys

import rospy

from geometry_msgs.msg import Pose
from object_detector_msgs.srv import estimate_poses, estimate_posesResponse

import os
import sys
import numpy as np
import scipy

import tensorflow as tf
from visualization_msgs.msg import Marker

import cv2
from cv_bridge import CvBridge, CvBridgeError

###################################
#### contact_graspnet imports #####
###################################

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "contact_graspnet")
print(BASE_DIR)
#BASE_DIR = "/contact_graspnet/contact_graspnet" # defined in run_container.sh
sys.path.append(os.path.join(BASE_DIR))
import config_utils
from data import regularize_pc_point_count, depth2pc, load_available_input_data

from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image

###################################
######## Utils funcitons ##########
###################################


def preprocess_image(x, mode='caffe'):
    x = x.astype(np.float32)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
    elif mode == 'caffe':
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x


def input_resize(image, target_size, intrinsics):
    # image: [y, x, c] expected row major
    # target_size: [y, x] expected row major
    # instrinsics: [fx, fy, cx, cy]

    intrinsics = np.asarray(intrinsics)
    y_size, x_size, c_size = image.shape

    if (y_size / x_size) < (target_size[0] / target_size[1]):
        resize_scale = target_size[0] / y_size
        crop = int((x_size - (target_size[1] / resize_scale)) * 0.5)
        image = image[:, crop:(x_size-crop), :]
        image = cv2.resize(image, (int(target_size[1]), int(target_size[0])))
        intrinsics = intrinsics * resize_scale
    else:
        resize_scale = target_size[1] / x_size
        crop = int((y_size - (target_size[0] / resize_scale)) * 0.5)
        image = image[crop:(y_size-crop), :, :]
        image = cv2.resize(image, (int(target_size[1]), int(target_size[0])))
        intrinsics = intrinsics * resize_scale

    return image, intrinsics


def toPix_array(translation, fx, fy, cx, cy):

    xpix = ((translation[:, 0] * fx) / translation[:, 2]) + cx
    ypix = ((translation[:, 1] * fy) / translation[:, 2]) + cy
    #zpix = translation[2] * fxkin

    return np.stack((xpix, ypix), axis=1)


def run_inference(sess, gcn_model, rgb, depth, cam_K, depth_cut=[0.2, 1.5], skip_border=False, filter_grasps=True, forward_passes=1, bbox=None, mask=None):

    rospy.loginfo("mask min: {}, max: {}".format(np.min(mask), np.max(mask)) if mask is not None else "No mask provided")

    print('Converting depth to point cloud(s)...')

    if mask is not None:
        rospy.loginfo("Using mask to crop the image")
        segmap = np.where(mask, 1, 0)
    else:
        rospy.logwarn("No mask no service.")
        return None, None

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.imshow(segmap * 255, cmap='gray')
    plt.title('Segmentation Map (segmap)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/root/hsr_contact_graspnet/segmap_debug.png')  # Or any writable path
    plt.close()

    print("Segmentation map saved to /tmp/segmap_debug.png")

    local_regions = True
    pc_full, pc_segments, pc_colors = gcn_model.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb, skip_border_objects=skip_border, z_range=depth_cut)

    print("pc_full: ", np.unique(pc_full), pc_full.shape)
    print("pc_segments: ", pc_segments[1].shape)

    import open3d as o3d
    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_segments[1])

    # Optionally, set a uniform color (e.g., gray)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    print(pcd)

    # Save to a PLY file
    o3d.io.write_point_cloud("/root/hsr_contact_graspnet/output.ply", pcd)

    print("pc_colors: ", np.unique(pc_colors), pc_colors.shape)

    print('Generating Grasps...')
    pred_grasps_cam, scores, contact_pts, _ = gcn_model.predict_scene_grasps(sess, pc_full, pc_segments=pc_segments, local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes)

    print("pred_grasp_cam: ", pred_grasps_cam)
    print("scores: ", scores)
    print("contact_pts: ", contact_pts)

    # Check if any grasps were returned
    if not scores or 1.0 not in scores or len(scores[1.0]) == 0:
        rospy.logwarn("No grasp scores found at confidence level 1.0.")
        return None, None

    idx = np.argmax(scores[1.0])

    return pred_grasps_cam[1.0], idx


#################################
############### ROS #############
#################################
class estimateGraspPose:
    def __init__(self, name):
        # ROS params

        checkpoint_path = "/root/hsr_contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001"
        pose_estimator_topic = "/find_graspposes_contact_graspnet"
        
        self.intrinsics = np.array(rospy.get_param('/pose_estimator/intrinsics'))
        rospy.loginfo(f"[{name}] Using intrinsics: {self.intrinsics}")

        self.frame_id = rospy.get_param('/pose_estimator/color_frame_id', 'head_rgbd_sensor_rgb_frame')
        self.depth_encoding = rospy.get_param('/pose_estimator/depth_encoding', 'mono16')
        self.depth_scale = rospy.get_param('/pose_estimator/depth_scale', 1000.0)
        self.gripper_offset = rospy.get_param('/pose_estimator/gripper_offset', 0.015)

        # Camera intrinsics
        rospy.loginfo(f"[{name}] Waiting for camera info...")

        ##################################
        # Building and loading the model #
        ##################################

        self.depth_cut = [0.2, 1.5]
        self.filter_grasps = True
        self.batch_size = 1
        self.skip_border_objects = False

        print('checkpoint path: ', checkpoint_path)
        global_config = config_utils.load_config(checkpoint_path, self.batch_size, arg_configs="")
        print('global config: ', global_config)
        self.grasp_estimator = GraspEstimator(global_config)
        self.grasp_estimator.build_network()

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(save_relative_paths=True)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)

        # Load weights
        self.grasp_estimator.load_weights(self.sess, saver, checkpoint_path, mode='test')

        self.service = rospy.Service(pose_estimator_topic, estimate_poses, self.estimate_grasp_poses)
        
        self.marker_pub = rospy.Publisher('/grasping_pipeline/grasp_marker', Marker, queue_size=10)

        print("Grasp Pose Estimation with Contact-GraspNet is ready.")

    def estimate_grasp_poses(self, req):
        rospy.loginfo("Calling Contact GraspNet")
        rospy.loginfo(f"{self.intrinsics[0,2]}")
        detection = req.det
        rgb = req.rgb
        depth = req.depth

        width, height = rgb.width, rgb.height
        assert width == 640 and height == 480

        try:
            rgb_image = CvBridge().imgmsg_to_cv2(rgb, "bgr8")
            print()
        except CvBridgeError as e:
            print(e)

        try:
            depth.encoding = self.depth_encoding
            depth_img = CvBridge().imgmsg_to_cv2(depth, self.depth_encoding)
            depth_img = depth_img / int(self.depth_scale)

        except CvBridgeError as e:
            print(e)

        response = estimate_posesResponse()
        estimates = []

        mask = detection.mask
        mask = np.zeros((height, width), dtype=np.uint8)
        mask_ids = np.array(detection.mask)
        mask[np.unravel_index(mask_ids, (height, width))] = 255
        mask = mask > 127

        bbox = detection.bbox
        bbox = [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]

        # Run inference
        pred_grasps_cam, idx = run_inference(
            self.sess,
            self.grasp_estimator,
            rgb_image,
            depth_img,
            self.intrinsics,
            self.depth_cut,
            self.skip_border_objects,
            self.filter_grasps,
            self.batch_size,
            bbox,
            mask)

        # Check for valid predictions
        if pred_grasps_cam is None or idx is None:
            rospy.logwarn("No valid grasp predictions to process.")
            response.poses = estimates
            return response

        for i, pred_grasp in enumerate(pred_grasps_cam):

            current_orientation = pred_grasp[:3, :3]
            # Rotate the grasp by 90 degrees
            pred_grasp[:3, :3] = np.dot(current_orientation, scipy.spatial.transform.Rotation.from_euler('z', np.pi/2).as_matrix())

            # Move the grasp along its z-axis
            grasp_translation = [0, 0, self.gripper_offset]
            head_camera_translation = current_orientation.dot(grasp_translation)
            pred_grasp[0, 3] += head_camera_translation[0]
            pred_grasp[1, 3] += head_camera_translation[1]
            pred_grasp[2, 3] += head_camera_translation[2]

            pose = self.create_pose(pred_grasp)
            
            if i == idx:
                estimates.append(pose)

        rospy.loginfo('marker published')

        response.poses = estimates
        rospy.loginfo(f"Estimated {estimates}")
        return response
    
    def create_pose(self, grasp_pose):

        msg = Pose()
        msg.position.x = grasp_pose[0, 3]
        msg.position.y = grasp_pose[1, 3]
        msg.position.z = grasp_pose[2, 3]

        rotation_matrix = grasp_pose[:3, :3]
        r = scipy.spatial.transform.Rotation.from_matrix(rotation_matrix)
        quat = r.as_quat()

        msg.orientation.x = quat[0]
        msg.orientation.y = quat[1]
        msg.orientation.z = quat[2]
        msg.orientation.w = quat[3]
        
        return msg
    
    def add_marker(self, pose_msg, id, color):
      
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time()
        marker.ns = 'grasp_marker ' + str(id)
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        marker.pose = pose_msg

        marker.scale.x = 0.1
        marker.scale.y = 0.05
        marker.scale.z = 0.01

        if color == "b":
            marker.color.a = 1.0
            marker.color.r = 0
            marker.color.g = 0
            marker.color.b = 1.0  
        else:
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0
            marker.color.b = 0

        self.marker_pub.publish(marker)


if __name__ == '__main__':
    rospy.init_node('contact_graspnet')
    server = estimateGraspPose(rospy.get_name())
    rospy.spin()
