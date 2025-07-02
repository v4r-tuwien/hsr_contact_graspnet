#!/usr/bin/env python

import sys

import rospy
import message_filters
#import tf2_ros

import actionlib
from geometry_msgs.msg import Pose, PoseArray, Quaternion, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from robokudo_msgs.msg import GenericImgProcAnnotatorResult, GenericImgProcAnnotatorAction
# from contact_graspnet_node.srv import returnGrasps

import os
import sys
import math
import numpy as np
import copy
import json
import scipy

import tensorflow as tf
from visualization_msgs.msg import Marker

import ros_numpy
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

from contact_grasp_estimator import GraspEstimator

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

    print('Is inference looping?')

    #image, intrinsics = input_resize(image,
    #                     [480, 640],
    #                     [cam_fx, cam_fy, cam_cx, cam_cy])
    image_raw = copy.deepcopy(rgb)
    depth = depth*0.001
    #image = preprocess_image(rgb)
    #image_mask = copy.deepcopy(image)

    print("image stats: ", np.unique(depth), depth.shape)
    print("z range: ", depth_cut, depth_cut)
    print("cam_K: ", cam_K, cam_K.shape)

    #if pc_full is None:
    print('Converting depth to point cloud(s)...')

    if bbox is not None:
        rospy.loginfo("Using bbox to crop the image")
        x_min = bbox.x_offset
        x_max = x_min + bbox.width
        y_min = bbox.y_offset
        y_max = y_min + bbox.height

        segmap = np.zeros(depth.shape)
        segmap[y_min:y_max, x_min:x_max] = 1
    elif mask is not None:
        rospy.loginfo("Using mask to crop the image")
        mask_numpy = ros_numpy.numpify(mask)
        segmap = np.where(mask_numpy, 1, 0)
    else:
        segmap = np.ones(depth.shape)

    local_regions = True
    pc_full, pc_segments, pc_colors = gcn_model.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb, skip_border_objects=skip_border, z_range=depth_cut)

    print("pc_full: ", np.unique(pc_full), pc_full.shape)
    print("pc_segments: ", pc_segments)
    print("pc_colors: ", np.unique(pc_colors), pc_colors.shape)

    print('Generating Grasps...')
    pred_grasps_cam, scores, contact_pts, _ = gcn_model.predict_scene_grasps(sess, pc_full, pc_segments=pc_segments, local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes)

    print("pred_grasp_cam: ", pred_grasps_cam)
    print("scores: ", scores)
    print("contact_pts: ", contact_pts)

    idx = np.argmax(scores[1.0])

    return pred_grasps_cam[1.0], idx


#################################
############### ROS #############
#################################
class estimateGraspPose:
    def __init__(self, name):
        # ROS params

        camera_topic = "/hsrb/head_rgbd_sensor/depth_registered/camera_info"
        checkpoint_path = "/contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001"
        pose_estimator_topic = "/pose_estimator/find_grasppose_cg"
    
        # Camera intrinsics
        rospy.loginfo(f"[{name}] Waiting for camera info...")
        self.camera_info = rospy.wait_for_message(camera_topic, CameraInfo)
        self.intrinsics = np.array([v for v in self.camera_info.K]).reshape(3, 3)

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

        self.server = actionlib.SimpleActionServer(pose_estimator_topic,
                                                   GenericImgProcAnnotatorAction,
                                                   execute_cb=self.callback,
                                                   auto_start=False)
        self.marker_pub = rospy.Publisher('/grasping_pipeline/grasp_marker', Marker, queue_size=10)
        self.server.start()
        print("Grasp Pose Estimation with Contact-GraspNet is ready.")

    def callback(self, req):
        rospy.loginfo("Calling Contact GraspNet")
 
        rgb = ros_numpy.numpify(req.rgb)
        depth = ros_numpy.numpify(req.depth).astype(np.float32)

        bbox = req.bb_detections[0]
        mask = req.mask_detections[0]

        # Run inference
        pred_grasps_cam, idx = run_inference(
            self.sess,
            self.grasp_estimator,
            rgb,
            depth,
            self.intrinsics,
            self.depth_cut,
            self.skip_border_objects,
            self.filter_grasps,
            self.batch_size,
            bbox,
            mask)

        response = GenericImgProcAnnotatorResult()

        for i, pred_grasp in enumerate(pred_grasps_cam):

            current_orientation = pred_grasp[:3, :3]
            # Rotate the grasp by 90 degrees
            pred_grasp[:3, :3] = np.dot(current_orientation, scipy.spatial.transform.Rotation.from_euler('z', np.pi/2).as_matrix())

            # Move the grasp along its z-axis
            grasp_translation = [0, 0, 0.015]
            head_camera_translation = current_orientation.dot(grasp_translation)
            pred_grasp[0, 3] += head_camera_translation[0]
            pred_grasp[1, 3] += head_camera_translation[1]
            pred_grasp[2, 3] += head_camera_translation[2]

            pose = self.create_pose(pred_grasp)
            
            if i == idx:
                #self.add_marker(pose, i, "b")
                response.pose_results.append(pose)
                response.class_ids.append(-1)
            #else:
                #self.add_marker(pose, i, "r")

        rospy.loginfo('marker published')

        self.server.set_succeeded(response)

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
        marker.header.frame_id = "head_rgbd_sensor_rgb_frame"
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