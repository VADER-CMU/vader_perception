#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PoseStamped
from vader_msgs.msg import Pepper
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.point_cloud2 as pc2
from pose_estimation import PoseEstimation
from segmentation import Segmentation
from msg_utils import pack_pepper_message, pack_debug_fruit_message

class FruitDetectionNode:
    def __init__(self):

        rospy.init_node('pose_estimation', anonymous=True)

        self.latest_depth = None
        self.latest_image = None
        self.position = None
        self.quaternion = None 

        self.peduncle_position = None

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

        self.coarse_pose_publisher = rospy.Publisher('gripper_coarse_pose', Pepper, queue_size=10)
        self.fine_pose_publisher = rospy.Publisher('fruit_fine_pose', Pepper, queue_size=10)

        self.debug_fine_pose_pub = rospy.Publisher('debug_fine_pose', PoseStamped, queue_size=10)

        rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        self.rate = rospy.Rate(10)
        
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

        PoseEst = PoseEstimation()

        # Main processing loop
        while not rospy.is_shutdown():
            # Check if we have received both messages
            if self.latest_depth is not None and self.latest_image is not None:
                # print(self.latest_image.shape)

                fruit_results = self.FruitSeg.infer(self.latest_image[:, 104:744, :], confidence=0.7, verbose=False)


                # Pepper priority policy here
                fruit_result = fruit_results[0]
                fruit_masks = fruit_result.masks

                peduncle_results = self.PeduncleSeg.infer(self.latest_image[:, 104:744, :], confidence=0.7, verbose=False)
                
                
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
                    

                    self.position, self.quaternion, self.peduncle_position = PoseEst.fine_fruit_pose_estimation(self.latest_image, self.latest_depth, fruit_mask, peduncle_mask)

                elif not fruit_masks is None:
                    # print("Number of detected masks: ", len(fruit_result.masks.data))
                    # print(fruit_masks.data.shape)
                    fruit_mask = fruit_masks.data[0].cpu().numpy().astype('uint16')
                    fruit_mask = np.pad(fruit_mask, ((0, 0), (104, 104)), mode='constant', constant_values=0)
                    

                    self.position, _ = PoseEst.coarse_fruit_pose_estimation(self.latest_image, self.latest_depth, fruit_mask)

                
                if self.position is not None:

                    # Do we have orientation? If so, we have fine pose estimation
                    if self.peduncle_position is not None:
                        # Publish on fine pose topic
                        fine_pose_msg = pack_pepper_message(position=self.position, quaternion=self.quaternion, peduncle_position=self.peduncle_position, frame_id="camera_depth_optical_frame")
                        self.fine_pose_publisher.publish(fine_pose_msg)

                        # Publish on debug topic
                        debug_pose_msg = pack_debug_fruit_message(position=self.position, quaternion=self.quaternion, frame_id="camera_depth_optical_frame")
                        self.debug_fine_pose_pub.publish(debug_pose_msg)

                        # Set orientation to None again to avoid publishing old data
                        self.peduncle_position = None

                    # Compute the "up" orientation based on the robot base frame
                    up_orientation  = [0, 0, 0, 1]  # To be changed
                    
                    coarse_pose_msg = pack_pepper_message(position=self.position, quaternion=up_orientation, frame_id="camera_depth_optical_frame")
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