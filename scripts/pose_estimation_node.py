#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from vader_msgs.msg import Pepper
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from pose_estimation import PoseEstimation, Segmentation, SequentialSegmentation
from msg_utils import wait_for_camera_info, pack_pepper_message, pack_debug_fruit_message, pack_debug_pcd

class FruitDetectionNode:
    def __init__(self):

        rospy.init_node('pose_estimation', anonymous=True)

        self.latest_depth = None
        self.latest_image = None
        
        self.pose_dict = {}

        self.fruit_pcd = None

        self.peduncle_position = None

        self.segmentation_models = {
            "fruit": {
                "model_path": rospy.get_param('fruit_weights_path'), 
                "drive_url": rospy.get_param('fruit_drive_url'),
                "confidence": rospy.get_param('fruit_confidence', 0.8)
            },
            "peduncle": {
                "model_path": rospy.get_param('peduncle_weights_path'), 
                "drive_url": rospy.get_param('peduncle_drive_url'),
                "confidence": rospy.get_param('peduncle_confidence', 0.8)
            }
        }

        # self.FruitSeg = Segmentation(self.segmentation_models["fruit"]) 
        # self.PeduncleSeg = Segmentation(self.segmentation_models["peduncle"]) 

        self.Segmentation = SequentialSegmentation(self.segmentation_models)

        self.coarse_pose_publisher = rospy.Publisher('gripper_coarse_pose', Pepper, queue_size=10)
        self.fine_pose_publisher = rospy.Publisher('fruit_fine_pose', Pepper, queue_size=10)

        self.debug_fine_pose_pub = rospy.Publisher('debug_fine_pose', PoseStamped, queue_size=10)
        self.debug_fruit_pcd_pub = rospy.Publisher('debug_fruit_pcd', PointCloud2, queue_size=10)
        self.debug_peduncle_pcd_pub = rospy.Publisher('debug_peduncle_pcd', PointCloud2, queue_size=10)

        rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        self.rate = rospy.Rate(10)

        self.camera_info_topic = '/camera/depth/camera_info'
        self.cam_frame_id = "camera_depth_optical_frame"
        
        rospy.loginfo("Fruit detection node initialized")
    
    def depth_callback(self, msg):
        self.bridge = CvBridge()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth_np = np.array(cv_image)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))   
        
        self.latest_depth = depth_np

    
    def image_callback(self, msg):

        
        self.bridge = CvBridge()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            image_np = np.array(cv_image)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
        
        self.latest_image = image_np
       
    
    def process_data_and_publish(self):

        intrinsics = wait_for_camera_info(self.camera_info_topic, timeout=5.0)
        if intrinsics is None:
            rospy.logerr("Failed to get camera intrinsics. Check the camera.")
        self.PoseEst = PoseEstimation(intrinsics)

        peduncle_images_path = "/home/kshitij/Documents/Bell Pepper/vader_docker/docker_ws/catkin_ws/src/vader_perception/peduncle_imgs"
        # Main processing loop
        while not rospy.is_shutdown():

            # Check if we have received both messages
            if self.latest_depth is not None and self.latest_image is not None:
                # print(self.latest_image.shape)

                results = self.Segmentation.infer(self.latest_image, coarse_only=False, verbose=False)

                for result in results:
                    self.pose_dict = self.PoseEst.pose_estimation(self.latest_image, self.latest_depth, result)


                if "fruit_position" in self.pose_dict:

                    # Do we have orientation? If so, we have fine pose estimation
                    if "peduncle_position" in self.pose_dict:
                        # Publish on fine pose topic
                        fine_pose_msg = pack_pepper_message(position=self.pose_dict["fruit_position"], quaternion=self.pose_dict["fruit_quaternion"], peduncle_position=self.pose_dict["peduncle_position"], frame_id=self.cam_frame_id)
                        self.fine_pose_publisher.publish(fine_pose_msg)

                        debug_pcd_msg = pack_debug_pcd(self.pose_dict["peduncle_pcd"], frame_id=self.cam_frame_id)
                        self.debug_peduncle_pcd_pub.publish(debug_pcd_msg)

                    coarse_pose_msg = pack_pepper_message(position=self.pose_dict["fruit_position"], quaternion=self.pose_dict["fruit_quaternion"], peduncle_position=None, frame_id=self.cam_frame_id)
                    self.coarse_pose_publisher.publish(coarse_pose_msg)
                    # Publish on debug topic
                    debug_pose_msg = pack_debug_fruit_message(position=self.pose_dict["fruit_position"], quaternion=self.pose_dict["fruit_quaternion"], frame_id=self.cam_frame_id)
                    self.debug_fine_pose_pub.publish(debug_pose_msg)
                    debug_pcd_msg = pack_debug_pcd(self.pose_dict["fruit_pcd"], frame_id=self.cam_frame_id)
                    self.debug_fruit_pcd_pub.publish(debug_pcd_msg)

                    # Set orientation to None again to avoid publishing old data
                    self.peduncle_position = None

                    # Compute the "up" orientation based on the robot base frame
                    up_orientation  = [0, 0, 0, 1]  # To be changed

                    coarse_pose_msg = pack_pepper_message(position=self.pose_dict["fruit_position"], quaternion=up_orientation, frame_id=self.cam_frame_id)
                    self.coarse_pose_publisher.publish(coarse_pose_msg)

            
            self.rate.sleep()

def main():
    try:
        node = FruitDetectionNode()
        node.process_data_and_publish()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated")

if __name__ == '__main__':
    main()