#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PoseArray
from vader_msgs.msg import PepperArray
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.point_cloud2 as pc2
from pose_estimation import PoseEstimation, SequentialSegmentation
from msg_utils import pack_pepper_message, pack_debug_fruit_message, pack_debug_pcd
from msg_utils import wait_for_camera_info, pack_debug_pose_array_message, pack_pepper_array_message, pack_ordered_pepper_array_message

class FruitDetectionNode:
    def __init__(self):

        rospy.init_node('pose_estimation', anonymous=True)

        self.gripper_depth = None
        self.gripper_image = None

        self.cutter_depth = None
        self.cutter_image = None

        # Pose dict structure
        # pose_dict = {
        #     "fruit_position": fruit_center,
        #     "fruit_quaternion": quaternion,
        #     "fruit_pcd": fruit_pcd,
        #     "peduncle_position": peduncle_center,
        #     "peduncle_quaternion": peduncle_quaternion,
        #     "peduncle_pcd": peduncle_pcd
        # }
        self.gripper_pose_dict_array = []
        self.cutter_pose_dict_array = []


        self.gripper_cam_frame_id = "gripper_cam_depth_optical_frame"
        self.cutter_cam_frame_id = "cutter_cam_depth_optical_frame"


        self.segmentation_models = {
            "fruit": {
                "model_path": rospy.get_param('fruit_weights_path'), 
                "drive_url": rospy.get_param('fruit_drive_url'),
                "confidence": rospy.get_param('fruit_confidence')
            },
            "peduncle": {
                "model_path": rospy.get_param('peduncle_weights_path'), 
                "drive_url": rospy.get_param('peduncle_drive_url'),
                "confidence": rospy.get_param('peduncle_confidence')
            }
        }

        self.Segmentation = SequentialSegmentation(self.segmentation_models)

        

        # gripper publishers
        # self.gripper_coarse_pose_publisher = rospy.Publisher('gripper_coarse_pose', Pepper, queue_size=10)
        # self.gripper_fine_pose_publisher = rospy.Publisher('fruit_fine_pose', Pepper, queue_size=10)
        # self.debug_gripper_fine_pose_publisher = rospy.Publisher('debug_gripper_fine_pose', PoseStamped, queue_size=10)
        # self.debug_gripper_pcd_pub = rospy.Publisher('debug_gripper_pcd', PointCloud2, queue_size=10)


        # # cutter publishers
        # self.cutter_coarse_pose_publisher = rospy.Publisher('cutter_coarse_pose', Pepper, queue_size=10)
        # self.debug_cutter_pose_pub = rospy.Publisher('debug_cutter_pose', PoseStamped, queue_size=10)
        # self.debug_cutter_pcd_pub = rospy.Publisher('debug_cutter_pcd', PointCloud2, queue_size=10)


        # gripper publishers
        self.gripper_debug_coarse_pose_array_pub = rospy.Publisher('gripper_debug_coarse_pose_array', PoseArray, queue_size=10)
        self.gripper_debug_fine_pose_array_pub = rospy.Publisher('gripper_debug_fine_pose_array', PoseArray, queue_size=10)

        self.gripper_coarse_pepper_array_pub = rospy.Publisher('gripper_coarse_pepper_array', PepperArray, queue_size=10)
        self.gripper_fine_pepper_array_pub = rospy.Publisher('gripper_fine_pepper_array', PepperArray, queue_size=10)

        # cutter publishers
        self.cutter_debug_coarse_pose_array_pub = rospy.Publisher('cutter_debug_coarse_pose_array', PoseArray, queue_size=10)
        self.cutter_debug_fine_pose_array_pub = rospy.Publisher('cutter_debug_fine_pose_array', PoseArray, queue_size=10)
        # No fine pose for cutter camera

        self.cutter_coarse_pepper_array_pub = rospy.Publisher('cutter_coarse_pepper_array', PepperArray, queue_size=10)
        # No fine pose for cutter camera


        # Subscribers
        rospy.Subscriber(
            "/gripper_cam/depth/image_rect_raw", 
            Image, 
            lambda msg: self.depth_callback(msg, "gripper")
        )
        rospy.Subscriber(
            '/gripper_cam/color/image_raw', 
            Image, 
            lambda msg: self.image_callback(msg, "gripper")
        )

        rospy.Subscriber(
            "/cutter_cam/depth/image_rect_raw", 
            Image, 
            lambda msg: self.depth_callback(msg, "cutter")
        )
        rospy.Subscriber(
            '/cutter_cam/color/image_raw', 
            Image, 
            lambda msg: self.image_callback(msg, "cutter")
        )

        self.gripper_camera_info_topic = '/gripper_cam/depth/camera_info'
        self.gripper_cam_frame_id = "gripper_cam_depth_optical_frame"

        self.cutter_camera_info_topic = '/cutter_cam/depth/camera_info'
        self.cutter_cam_frame_id = "cutter_cam_depth_optical_frame"

        self.rate = rospy.Rate(10)
        
        rospy.loginfo("Fruit detection node initialized")
    

    def depth_callback(self, msg, cam):
        self.bridge = CvBridge()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth_np = np.array(cv_image)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))   

        if cam == "gripper":
            self.gripper_depth = depth_np
        elif cam == "cutter":
            self.cutter_depth = depth_np


    def image_callback(self, msg, cam):

        self.bridge = CvBridge()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            image_np = np.array(cv_image)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
        
        if cam == "gripper":
            self.gripper_image = image_np
        elif cam == "cutter":
            self.cutter_image = image_np
       
    
    def process_data_and_publish(self):

        gripper_intrinsics = wait_for_camera_info(self.gripper_camera_info_topic, timeout=10.0)
        if gripper_intrinsics is None:
            rospy.logerr("Failed to get gripper camera intrinsics. Check the camera.")
        self.gripper_pose_estimator = PoseEstimation(gripper_intrinsics)

        cutter_intrinsics = wait_for_camera_info(self.cutter_camera_info_topic, timeout=10.0)
        if cutter_intrinsics is None:
            rospy.logerr("Failed to get cutter camera intrinsics. Check the camera.")
        self.cutter_pose_estimator = PoseEstimation(cutter_intrinsics)

        # Main processing loop
        while not rospy.is_shutdown():
            # Check if we have received both messages
            if self.gripper_depth is not None and self.gripper_image is not None:

                # Structure: results = {"fruit_masks": [], "peduncle_masks": []}
                results = self.Segmentation.infer_large_fov(self.gripper_image, coarse_only=False, verbose=False)

                for result in results:
                    pose_dict = self.gripper_pose_estimator.pose_estimation(self.gripper_image, self.gripper_depth, result, offset=np.array([0,0,0.045]))
                    self.gripper_pose_dict_array.append(pose_dict)

                debug_fine_pose_array_msg = pack_debug_pose_array_message(self.gripper_pose_dict_array, fine=True, frame_id=self.gripper_cam_frame_id)
                self.gripper_debug_fine_pose_array_pub.publish(debug_fine_pose_array_msg)

                debug_coarse_pose_array_msg = pack_debug_pose_array_message(self.gripper_pose_dict_array, fine=False, frame_id=self.gripper_cam_frame_id)
                self.gripper_debug_coarse_pose_array_pub.publish(debug_coarse_pose_array_msg)

                # coarse_pepper_array_msg = pack_pepper_array_message(self.pose_dict_array, fine=False, frame_id=self.gripper_cam_frame_id)

                # fine_pepper_array_msg = pack_pepper_array_message(self.pose_dict_array, fine=True, frame_id=self.gripper_cam_frame_id)
                coarse_pepper_array_msg, fine_pepper_array_msg = pack_ordered_pepper_array_message(self.gripper_pose_dict_array, fine=True, frame_id=self.gripper_cam_frame_id)
                self.gripper_coarse_pepper_array_pub.publish(coarse_pepper_array_msg)
                self.gripper_fine_pepper_array_pub.publish(fine_pepper_array_msg)


                self.gripper_pose_dict_array = []
            
            if self.cutter_depth is not None and self.cutter_image is not None:


                # Structure: results = {"fruit_masks": [], "peduncle_masks": []}
                results = self.Segmentation.infer_large_fov(self.cutter_image, coarse_only=True, verbose=False)

                for result in results:
                    pose_dict = self.cutter_pose_estimator.pose_estimation(self.cutter_image, self.cutter_depth, result, offset=np.array([0,0,0.045]))
                    self.cutter_pose_dict_array.append(pose_dict)

                debug_coarse_pose_array_msg = pack_debug_pose_array_message(self.cutter_pose_dict_array, fine=False, frame_id=self.cutter_cam_frame_id)
                self.cutter_debug_coarse_pose_array_pub.publish(debug_coarse_pose_array_msg)

                debug_fine_pose_array_msg = pack_debug_pose_array_message(self.cutter_pose_dict_array, fine=True, frame_id=self.cutter_cam_frame_id)
                self.cutter_debug_fine_pose_array_pub.publish(debug_fine_pose_array_msg)

                coarse_pepper_array_msg = pack_pepper_array_message(self.cutter_pose_dict_array, fine=False, frame_id=self.cutter_cam_frame_id)

                # fine_pepper_array_msg = pack_pepper_array_message(self.pose_dict_array, fine=True, frame_id=self.cutter_cam_frame_id)
                self.cutter_coarse_pepper_array_pub.publish(coarse_pepper_array_msg)


                self.cutter_pose_dict_array = []

            
            self.rate.sleep()

def main():
    try:
        node = FruitDetectionNode()
        node.process_data_and_publish()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated")

if __name__ == '__main__':
    main()