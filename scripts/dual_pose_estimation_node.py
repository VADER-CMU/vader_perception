#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PoseStamped
from vader_msgs.msg import Pepper
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.point_cloud2 as pc2
from pose_estimation import PoseEstimation, Segmentation
from msg_utils import pack_pepper_message, pack_debug_fruit_message, pack_debug_pcd

class FruitDetectionNode:
    def __init__(self):

        rospy.init_node('pose_estimation', anonymous=True)

        self.gripper_depth = None
        self.gripper_image = None
        self.gripper_position_estimate = None
        self.gripper_quaternion_estimate = None
        self.gripper_peduncle_position = None

        self.cutter_depth = None
        self.cutter_image = None
        self.cutter_position_estimate = None
        self.cutter_quaternion_estimate = None

        self.gripper_fruit_pcd = None
        self.cutter_fruit_pcd = None

        self.gripper_cam_frame_id = "gripper_cam_depth_optical_frame"
        self.cutter_cam_frame_id = "cutter_cam_depth_optical_frame"


        fruit_model = {
            "model_path": rospy.get_param('fruit_weights_path'), 
            "drive_url": rospy.get_param('fruit_drive_url')
        }
        peduncle_model = {
            "model_path": rospy.get_param('peduncle_weights_path'), 
            "drive_url": rospy.get_param('peduncle_drive_url')
        }

        self.FruitSeg = Segmentation(fruit_model) 
        self.PeduncleSeg = Segmentation(peduncle_model) 

        self.pepper_confidence = rospy.get_param('pepper_confidence', 0.8)
        self.peduncle_confidence = rospy.get_param('peduncle_confidence', 0.8)

        # gripper publishers
        self.gripper_coarse_pose_publisher = rospy.Publisher('gripper_coarse_pose', Pepper, queue_size=10)
        self.gripper_fine_pose_publisher = rospy.Publisher('fruit_fine_pose', Pepper, queue_size=10)
        self.debug_gripper_fine_pose_publisher = rospy.Publisher('debug_gripper_fine_pose', PoseStamped, queue_size=10)
        self.debug_gripper_pcd_pub = rospy.Publisher('debug_gripper_pcd', PointCloud2, queue_size=10)


        # cutter publishers
        self.cutter_coarse_pose_publisher = rospy.Publisher('cutter_coarse_pose', Pepper, queue_size=10)
        self.debug_cutter_pose_pub = rospy.Publisher('debug_cutter_pose', PoseStamped, queue_size=10)
        self.debug_cutter_pcd_pub = rospy.Publisher('debug_cutter_pcd', PointCloud2, queue_size=10)

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

        PoseEst = PoseEstimation()

        # Main processing loop
        while not rospy.is_shutdown():
            # Check if we have received both messages
            if self.gripper_depth is not None and self.gripper_image is not None:

                fruit_results = self.FruitSeg.infer(self.gripper_image[:, 104:744, :], confidence=self.pepper_confidence, verbose=False)


                # Pepper priority policy here
                fruit_result = fruit_results[0]
                fruit_masks = fruit_result.masks

                peduncle_results = self.PeduncleSeg.infer(self.gripper_image[:, 104:744, :], confidence=self.peduncle_confidence, verbose=False)


                # Pepper priority policy here
                peduncle_result = peduncle_results[0]
                peduncle_masks = peduncle_result.masks

                

                if not fruit_masks is None and not peduncle_masks is None:
                    fruit_mask = fruit_masks.data[0].cpu().numpy().astype('uint16')
                    fruit_mask = np.pad(fruit_mask, ((0, 0), (104, 104)), mode='constant', constant_values=0)

                    peduncle_mask = peduncle_masks.data[0].cpu().numpy().astype('uint16')
                    peduncle_mask = np.pad(peduncle_mask, ((0, 0), (104, 104)), mode='constant', constant_values=0)

                    kernel = np.ones((5, 5), np.uint8)
                    # peduncle_mask = cv2.erode(peduncle_mask, kernel, iterations=3)
                    

                    self.gripper_position_estimate, self.gripper_quaternion_estimate, self.gripper_peduncle_position, self.gripper_fruit_pcd = PoseEst.fine_fruit_pose_estimation(self.gripper_image, self.gripper_depth, fruit_mask, peduncle_mask)

                elif not fruit_masks is None:
                    # print("Number of detected masks: ", len(fruit_result.masks.data))
                    # print(fruit_masks.data.shape)
                    fruit_mask = fruit_masks.data[0].cpu().numpy().astype('uint16')
                    fruit_mask = np.pad(fruit_mask, ((0, 0), (104, 104)), mode='constant', constant_values=0)
                    

                    self.gripper_position_estimate, self.gripper_quaternion_estimate, self.gripper_fruit_pcd = PoseEst.coarse_fruit_pose_estimation(self.gripper_image, self.gripper_depth, fruit_mask)

                if self.gripper_position_estimate is not None:

                    # Do we have orientation? If so, we have fine pose estimation
                    if self.gripper_peduncle_position is not None:
                        # Publish on fine pose topic
                        fine_pose_msg = pack_pepper_message(position=self.gripper_position_estimate, quaternion=self.gripper_quaternion_estimate, peduncle_position=self.gripper_peduncle_position, frame_id=self.gripper_cam_frame_id)
                        self.gripper_fine_pose_publisher.publish(fine_pose_msg)


                    coarse_pose_msg = pack_pepper_message(position=self.gripper_position_estimate, quaternion=self.gripper_quaternion_estimate, peduncle_position=self.gripper_peduncle_position, frame_id=self.gripper_cam_frame_id)
                    self.gripper_coarse_pose_publisher.publish(coarse_pose_msg)
                    # Publish on debug topic
                    debug_pose_msg = pack_debug_fruit_message(position=self.gripper_position_estimate, quaternion=self.gripper_quaternion_estimate, frame_id=self.gripper_cam_frame_id)
                    self.debug_gripper_fine_pose_publisher.publish(debug_pose_msg)
                    debug_pcd_msg = pack_debug_pcd(self.gripper_fruit_pcd, frame_id=self.gripper_cam_frame_id)
                    self.debug_gripper_pcd_pub.publish(debug_pcd_msg)

                    # Set orientation to None again to avoid publishing old data
                    self.gripper_peduncle_position = None

                    # Compute the "up" orientation based on the robot base frame
                    up_orientation = [0, 0, 0, 1]  # To be changed

                    coarse_pose_msg = pack_pepper_message(position=self.gripper_position_estimate, quaternion=up_orientation, frame_id=self.gripper_cam_frame_id)
                    self.gripper_coarse_pose_publisher.publish(coarse_pose_msg)
            
            if self.cutter_depth is not None and self.cutter_image is not None:

                fruit_results = self.FruitSeg.infer(self.cutter_image[:, 104:744, :], confidence=self.pepper_confidence, verbose=False)

                # Pepper priority policy here
                fruit_result = fruit_results[0]
                fruit_masks = fruit_result.masks

                if not fruit_masks is None:
                    fruit_mask = fruit_masks.data[0].cpu().numpy().astype('uint16')
                    fruit_mask = np.pad(fruit_mask, ((0, 0), (104, 104)), mode='constant', constant_values=0)
                    self.cutter_position_estimate, self.cutter_quaternion_estimate, self.cutter_fruit_pcd = PoseEst.coarse_fruit_pose_estimation(self.cutter_image, self.cutter_depth, fruit_mask)

                if self.cutter_position_estimate is not None:

                    # Publish on debug topic
                    debug_pose_msg = pack_debug_fruit_message(position=self.cutter_position_estimate, quaternion=None, frame_id=self.cutter_cam_frame_id)
                    self.debug_cutter_pose_pub.publish(debug_pose_msg)
                    debug_pcd_msg = pack_debug_pcd(self.cutter_fruit_pcd, frame_id=self.cutter_cam_frame_id)
                    self.debug_cutter_pcd_pub.publish(debug_pcd_msg)

                    # Compute the "up" orientation based on the robot base frame
                    up_orientation  = [0, 0, 0, 1]  # To be changed

                    coarse_pose_msg = pack_pepper_message(position=self.cutter_position_estimate, quaternion=up_orientation, frame_id=self.cutter_cam_frame_id)
                    self.cutter_coarse_pose_publisher.publish(coarse_pose_msg)

            
            self.rate.sleep()

def main():
    try:
        node = FruitDetectionNode()
        node.process_data_and_publish()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated")

if __name__ == '__main__':
    main()