#!/usr/bin/env python

import sys

import rospy
import message_filters
import tf2_ros

from actionlib import SimpleActionServer
from geometry_msgs.msg import Pose, PoseArray, Quaternion, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from object_detector_msgs.srv import get_poses, get_posesResponse
from object_detector_msgs.msg import PoseWithConfidence

import os
import sys
import math
import numpy as np
import copy
import transforms3d as tf3d
import json

import tensorflow as tf

import cv2
from cv_bridge import CvBridge, CvBridgeError

###################################
#### contact_graspnet imports #####
###################################

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def run_inference(image, model, threeD_boxes,
                   cam_fx, cam_fy, cam_cx, cam_cy):
    obj_names = []
    obj_poses = []
    obj_confs = []

    image, intrinsics = input_resize(image,
                         [480, 640],
                         [cam_fx, cam_fy, cam_cx, cam_cy])
    image_raw = copy.deepcopy(image)
    image = preprocess_image(image)
    #image_mask = copy.deepcopy(image)

    if pc_full is None:
        print('Converting depth to point cloud(s)...')
        pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgg, skip_border_objects=skip_border_objects, z_range=z_range)

    print('Generating Grasps...')
    pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(sess, pc_full, pc_segments=pc_segments, 
                                                                                          local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes)  

    # Visualize results          
    show_image(rgb, segmap)
    visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)

    return obj_names, obj_poses, obj_confs, image_raw


#################################
############### ROS #############
#################################
class estimateGraspPose:
    def __init__(self, name):
        # ROS params
    
        self.checkpoint_path = ''
        self.color_topic = '/camera/rgb/image_color'
        self.depth_topic = '/camera/depth/image_rect'
        self.camera_topic = '/camera/color/camera_info'
        self.service_name = "/contact_graspnet/return_grasp_point"
        self.node_type = 'continuous'
    
        try:
            self.checkpoint_path = rospy.get_param('/contact_graspnet/checkpoints')
        except KeyError:
            print("please set path to model! example:/home/desired/path/to/resnet_xy.h5")
        
        if rospy.has_param('/contact_graspnet/rgb_topic'):
            self.color_topic = rospy.get_param("/contact_graspnet/rgb_topic")
            print('RGB image callback set to: ', rgb_topic)
        if rospy.has_param('/contact_graspnet/dep_topic'):
            self.depth_topic = rospy.get_param("/contact_graspnet/dep_topic")
            print("Depth image callback set to: ", dep_topic)
        if rospy.has_param('/contact_graspnet/camera_info_topic'):
            self.camera_topic = rospy.get_param("/contact_graspnet/camera_info_topic")
            print("camera info callback set to: ", cam_topic)
        if rospy.has_param('/contact_graspnet/node_type'):
            self.node_type = rospy.get_param("/contact_graspnet/node_type")
            self.service_name = rospy.get_param("/contact_graspnet/service_call")
            print("service call set to: ", self.service_name)

        # Camera
        self.slop = 0.2  # max delay between rgb and depth image [seconds]
        sub_rgb, sub_depth = message_filters.Subscriber(self.color_topic, Image),\
                             message_filters.Subscriber(self.depth_topic, Image)
        sub_rgbd = message_filters.ApproximateTimeSynchronizer([sub_rgb, sub_depth], 10, self.slop)
        sub_rgbd.registerCallback(self._update_image)

        # Camera intrinsics
        rospy.loginfo(f"[{name}] Waiting for camera info...")
        self.camera_info = rospy.wait_for_message(self.camera_topic, CameraInfo)
        self.intrinsics = np.array([v for v in self.camera_info.K]).reshape(3, 3)
        self.cam_fx, self.cam_fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        self.cam_cx, self.cam_cy = self.intrinsics[0, 2], self.intrinsics[1, 2]

        self.bridge = CvBridge()
        self.pose_pub = rospy.Publisher("/contact_graspnet/grasp_pose", PoseArray, queue_size=10)
        self.viz_pub = rospy.Publisher("/contact_graspnet/debug_viz", Image, queue_size=10)
       
        ##################################
        # Building and loading the model #
        ##################################

        self.grasp_estimator = GraspEstimator(global_config)
        self.grasp_estimator.build_network()

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(save_relative_paths=True)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        # Load weights
        self.grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')

        ##################################
        # node type [continuous, server] #
        ##################################

        #self._server.start()
        if self.node_type == 'service':
            self.pose_srv = rospy.Service(self.service_name, get_poses, self.callback)
            rospy.loginfo(f"[{self.service_name}] Server ready")
        else:
            pass

        #if rospy.get_param('/locateobject/publish_tf', True):
        #    self._br = tf2_ros.TransformBroadcaster()
        #    self._publish_tf()

    def _update_image(self, rgb, depth):
        self.rgb, self.depth = rgb, depth

    def callback(self, req):
        print("Received request")
        rgb_cv = self.bridge.imgmsg_to_cv2(self.rgb, "8UC3")
        rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)

        # Run inference
        det_objs, det_poses, det_confs, viz_img = run_estimation(
                rgb_cv, self.model, self.threeD_boxes,
            self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy)
        msg = self.fill_msg(det_objs, det_poses, det_confs)
        self.viz_pose(viz_img)
        return msg

    def fill_msg(self, det_names, det_poses, det_confidences):
        msg = PoseArray()

        msg.header.frame_id = self.camera_info.header.frame_id
        msg.header.stamp = rospy.Time(0)

        for idx in range(len(det_names)):
            item = Pose()
            item.position.x = det_poses[idx][0] 
            item.position.y = det_poses[idx][1] 
            item.position.z = det_poses[idx][2] 
            item.orientation.w = det_poses[idx][3] 
            item.orientation.x = det_poses[idx][4] 
            item.orientation.y = det_poses[idx][5] 
            item.orientation.z = det_poses[idx][6]
            msg.poses.append(item)
        self.pose_pub.publish(msg)

        msg = get_posesResponse()
        for idx in range(len(det_names)):
            item = PoseWithConfidence()
            item.name = str(det_names[idx]) 
            item.confidence = det_confidences[idx]
            item.pose = Pose()
            det_pose = det_poses[idx]
            item.pose.position.x = det_pose[0]
            item.pose.position.y = det_pose[1]
            item.pose.position.z = det_pose[2]
            item.pose.orientation.w = det_pose[3]
            item.pose.orientation.x = det_pose[4]
            item.pose.orientation.y = det_pose[5]
            item.pose.orientation.z = det_pose[6]
            msg.poses.append(item)
        
        return msg
    
    def viz_pose(self, image):
        msg = Image()
        msg.header.frame_id = self.camera_info.header.frame_id
        msg.header.stamp = self.camera_info.header.stamp
        data = self.bridge.cv2_to_imgmsg(image, "passthrough")
        self.viz_pub.publish(data)

if __name__ == '__main__':
    rospy.init_node(rospy.get_name())
    print(rospy.get_name())
    server = PoseEstimation(rospy.get_name())
    rospy.spin()
