#!/usr/bin/env python

import sys

import rospy
import message_filters
#import tf2_ros

from actionlib import SimpleActionServer
from geometry_msgs.msg import Pose, PoseArray, Quaternion, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from contact_graspnet_node.srv import returnGrasps

import os
import sys
import math
import numpy as np
import copy
import json

import tensorflow as tf

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

#BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = "/contact_graspnet/contact_graspnet" # defined in run_container.sh
sys.path.append(os.path.join(BASE_DIR))
import config_utils
from data import regularize_pc_point_count, depth2pc, load_available_input_data

from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image
from contact_graspnet_node.srv import returnGrasps 

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


def run_inference(sess, gcn_model, rgb, depth, cam_K, depth_cut=[0.2, 1.5], skip_border=False, filter_grasps=True, forward_passes=1):
    
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

    # Visualize results          
    show_image(rgb, segmap)
    visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)

    return contact_pts


#################################
############### ROS #############
#################################
class estimateGraspPose:
    def __init__(self, name):
        # ROS params
    
        self.checkpoint_path = ''
        self.color_topic = '/camera/color/image_raw'
        self.depth_topic = '/camera/depth/image_rect_raw'
        self.camera_topic = '/camera/color/camera_info'
        self.service_name = "/contact_graspnet/return_grasp_point"
        self.node_type = 'continuous'
    
        try:
            self.checkpoint_path = rospy.get_param('/contact_graspnet_module/checkpoint')
        except KeyError:
            print("please set path to model! example:/home/desired/path/to/resnet_xy.h5")
        
        if rospy.has_param('/contact_graspnet_module/rgb_topic'):
            self.color_topic = rospy.get_param("/contact_graspnet_module/rgb_topic")
            print('RGB image callback set to: ', self.color_topic)
        if rospy.has_param('/contact_graspnet_module/dep_topic'):
            self.depth_topic = rospy.get_param("/contact_graspnet_module/dep_topic")
            print("Depth image callback set to: ", self.depth_topic)
        if rospy.has_param('/contact_graspnet_module/camera_info_topic'):
            self.camera_topic = rospy.get_param("/contact_graspnet_module/camera_info_topic")
            print("camera info callback set to: ", self.camera_topic)
        if rospy.has_param('/contact_graspnet_module/node_type'):
            self.node_type = rospy.get_param("/contact_graspnet_module/node_type")
            if self.node_type == "service":
                self.service_name = rospy.get_param("/contact_graspnet_module/service_call")
                print("service call set to: ", self.service_name)
            elif self.node_type == "continuous":
                print("Node type set to continuous")
            else:
                print("please set a valid node_type [service, continuous]")
                sys.exit(1)

        # Camera
        self.bridge = CvBridge()
        self.rgb, self.depth = None, None
        self.slop = 0.2  # max delay between rgb and depth image [seconds]
        rgb_sub, depth_sub = message_filters.Subscriber(self.color_topic, Image),\
                             message_filters.Subscriber(self.depth_topic, Image)
        self.sub_rgbd = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, self.slop)
        self.sub_rgbd.registerCallback(self._update_image)

        # Camera intrinsics
        rospy.loginfo(f"[{name}] Waiting for camera info...")
        print("Waiting for camera info")
        self.camera_info = rospy.wait_for_message(self.camera_topic, CameraInfo)
        self.intrinsics = np.array([v for v in self.camera_info.K]).reshape(3, 3)
        print("Got camera info")

        #self.pose_pub = rospy.Publisher("/contact_graspnet/grasp_pose", PoseArray, queue_size=10)
        #self.viz_pub = rospy.Publisher("/contact_graspnet/debug_viz", Image, queue_size=10)
       
        ##################################
        # Building and loading the model #
        ##################################

        self.depth_cut = [0.2, 1.5]
        self.filter_grasps = True
        self.batch_size = 1
        self.skip_border_objects = False

        print('checkpoint path: ', self.checkpoint_path)
        global_config = config_utils.load_config(self.checkpoint_path, self.batch_size, arg_configs="")
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
        self.grasp_estimator.load_weights(self.sess, saver, self.checkpoint_path, mode='test')

        ##################################
        # node type [continuous, server] #
        ##################################

        #self._server.start()
        if self.node_type == 'service':
            self._srv = rospy.Service(self.service_name, returnGrasps, self.srv_callback)
            rospy.loginfo(f"[{self.service_name}] Server ready")
        else:
            self._sub = rospy.Subscriber(self.color_topic, Image, self.callback)
            self._pub = rospy.Publisher("/contact_graspnet/grasp_point", Pose, queue_size=10)
            rospy.loginfo(f"[contact_graspnet callback started")

        #if rospy.get_param('/locateobject/publish_tf', True):
        #    self._br = tf2_ros.TransformBroadcaster()
        #    self._publish_tf()

    def _update_image(self, rgb, depth):
        # image bridge that, casual
        self.rgb = ros_numpy.numpify(rgb)
        #self.rgb = self.bridge.imgmsg_to_cv2(rgb, "8UC3")
        #self.rgb = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2RGB)
        #self.depth = self.bridge.imgmsg_to_cv2(depth, "16UC1")
        self.depth = ros_numpy.numpify(depth).astype(np.float32)

    def callback(self):
        print("Received request")
        #rgb_cv = self.bridge.imgmsg_to_cv2(self.rgb, "8UC3")
        #rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)

        # Run inference
        grasp_pose = run_inference(self.sess, self.grasp_estimator, self.rgb, self.depth, self.intrinsics, self.depth_cut, self.skip_border_objects, self.filter_grasps, self.batch_size)
        self.publish_pose(grasp_pose)

    def srv_callback(self, request):
        print("Received request")

        # Run inference
        grasp_pose = run_inference(self.sess, self.grasp_estimator, self.rgb, self.depth, self.intrinsics, self.depth_cut, self.skip_border_objects, self.filter_grasps, self.batch_size)
        msg = self.publish_pose(grasp_pose)
        return msg

    def publish_pose(self, grasp_pose):
        msg = Pose()
        msg.position.x = grasp_pose[0]
        msg.position.y = grasp_pose[1]
        msg.position.z = grasp_pose[2]
        msg.orientation.w = grasp_pose[3]
        msg.orientation.x = grasp_pose[4]
        msg.orientation.y = grasp_pose[5]
        msg.orientation.z = grasp_pose[6]
        if self.node_type == "continuous":
            self._pub.publish(msg)
        elif self.node_type == "service":
            return msg
        else:
            print("node_type is fucky")

    
    def viz_pose(self, image):
        msg = Image()
        msg.header.frame_id = self.camera_info.header.frame_id
        msg.header.stamp = self.camera_info.header.stamp
        data = self.bridge.cv2_to_imgmsg(image, "passthrough")
        self.viz_pub.publish(data)

if __name__ == '__main__':
    rospy.init_node('contact_graspnet')
    server = estimateGraspPose(rospy.get_name())
    #rospy.init_node('contact_graspnet')
    rospy.spin()
