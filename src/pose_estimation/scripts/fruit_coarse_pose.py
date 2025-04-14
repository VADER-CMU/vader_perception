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

class FruitDetectionNode:
    def __init__(self):

        rospy.init_node('fruit_coarse_pose', anonymous=True)

        self.latest_depth = None
        self.latest_image = None
        self.position = None
        self.quaternion = None 
        
        self.pose_publisher = rospy.Publisher('fruit_coarse_pose', PoseStamped, queue_size=10)

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
        fruit_model_path = rospy.get_param('fruit_weights_path')
        peduncle_model_path = rospy.get_param('peduncle_weights_path')
        FruitSeg = Segmentation(fruit_model_path) 
        PeduncleSeg = Segmentation(peduncle_model_path) 
        # Main processing loop
        while not rospy.is_shutdown():
            # Check if we have received both messages
            if self.latest_depth is not None and self.latest_image is not None:
                print(self.latest_image.shape)
                
                fruit_results = FruitSeg.infer(self.latest_image[:, 104:744, :], confidence=0.7, verbose=True)
                
                
                # Pepper priority policy here
                fruit_result = fruit_results[0]
                fruit_masks = fruit_result.masks

                peduncle_results = PeduncleSeg.infer(self.latest_image[:, 104:744, :], confidence=0.7, verbose=True)
                
                
                # Pepper priority policy here
                peduncle_result = peduncle_results[0]
                peduncle_masks = peduncle_result.masks
                # print(result.probs)

                

                if not fruit_masks is None and not peduncle_masks is None:
                    # print("Number of detected masks: ", len(fruit_result.masks.data))
                    print(" Peduncle Detected ")
                    # print(fruit_masks.data.shape)
                    fruit_mask = fruit_masks.data[0].cpu().numpy().astype('uint16')
                    fruit_mask = np.pad(fruit_mask, ((0, 0), (104, 104)), mode='constant', constant_values=0)

                    peduncle_mask = peduncle_masks.data[0].cpu().numpy().astype('uint16')
                    peduncle_mask = np.pad(peduncle_mask, ((0, 0), (104, 104)), mode='constant', constant_values=0)

                    kernel = np.ones((5, 5), np.uint8)
                    peduncle_mask = cv2.erode(peduncle_mask, kernel, iterations=2)
                    

                    self.position, self.quaternion = PoseEst.fine_fruit_pose_estimation(self.latest_image, self.latest_depth, fruit_mask, peduncle_mask)

                elif not fruit_masks is None:
                    # print("Number of detected masks: ", len(fruit_result.masks.data))
                    # print(fruit_masks.data.shape)
                    fruit_mask = fruit_masks.data[0].cpu().numpy().astype('uint16')
                    fruit_mask = np.pad(fruit_mask, ((0, 0), (104, 104)), mode='constant', constant_values=0)
                    

                    self.position, _ = PoseEst.coarse_fruit_pose_estimation(self.latest_image, self.latest_depth, fruit_mask)



                    
                
                if self.position is not None:

                    center = self.position
                    # Create a PoseStamped message
                    pose_msg = PoseStamped()
                    
                    # Set header information
                    pose_msg.header.stamp = rospy.Time.now()
                    pose_msg.header.frame_id = "camera_depth_optical_frame"  # Use appropriate frame ID
        
                    pose_msg.pose.position.x = center[0]
                    pose_msg.pose.position.y = center[1]
                    pose_msg.pose.position.z = center[2]

                    if self.quaternion is not None:
                        # Set orientation (identity quaternion in this example)
                        pose_msg.pose.orientation.x = self.quaternion[1]
                        pose_msg.pose.orientation.y = self.quaternion[2]
                        pose_msg.pose.orientation.z = self.quaternion[3]
                        pose_msg.pose.orientation.w = self.quaternion[0]
                    else:
                        pose_msg.pose.orientation.x = 1.0
                        pose_msg.pose.orientation.y = 0.0
                        pose_msg.pose.orientation.z = 0.0
                        pose_msg.pose.orientation.w = 0.0
                    
                    # Set orientation (identity quaternion in this example)
                    # pose_msg.pose.orientation.x = quaternion[1]
                    # pose_msg.pose.orientation.y = quaternion[2]
                    # pose_msg.pose.orientation.z = quaternion[3]
                    # pose_msg.pose.orientation.w = quaternion[0]

                    # Set size of the pepper
                    # pose_msg.shape.type = 3 #cylinder
                    # pose_msg.shape.dimensions = [0.1, 0.075]
                
                    # Publish the pose
                    self.pose_publisher.publish(pose_msg)

            
            self.rate.sleep()

def main():
    try:
        node = FruitDetectionNode()
        node.process_data_and_publish()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated")

if __name__ == '__main__':
    main()