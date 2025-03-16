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
import matplotlib.pyplot as plt
class FruitDetectionNode:
    def __init__(self):

        rospy.init_node('fruit_coarse_pose', anonymous=True)

        self.latest_depth = None
        self.latest_image = None
        self.position = np.array([0.0, 0.0, 0.0])
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.pose_publisher = rospy.Publisher('fruit_pose', Pepper, queue_size=10)

        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        self.rate = rospy.Rate(10)
        
        rospy.loginfo("Fruit detection node initialized")
    
    def depth_callback(self, msg):
        self.bridge = CvBridge()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth_np = np.array(cv_image)
            rospy.logdebug("Converted image to numpy array")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))   
        
        self.latest_depth = depth_np

    
    def image_callback(self, msg):

        
        self.bridge = CvBridge()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            image_np = np.array(cv_image)
            rospy.logdebug("Converted image to numpy array")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
        
        self.latest_image = image_np
       
    
    def process_data_and_publish(self):

        PoseEst = PoseEstimation()
        model_path = rospy.get_param('weights_path')
        Seg = Segmentation(model_path) 
        # Main processing loop
        while not rospy.is_shutdown():
            # Check if we have received both messages
            if self.latest_depth is not None and self.latest_image is not None:
                
                
                print("Latest Image shape: ", self.latest_image.shape)
                results = Seg.infer(self.latest_image[:, 104:744, :], confidence=0.8)
                result = results[0]
                masks = result.masks
                if not masks is None:
                    mask = masks.data[0].cpu().numpy().astype('uint8') * 255
                    print("masks: ", mask.shape)

                    mask_pcd = PoseEst.coarse_fruit_pose_estimation(self.latest_depth, mask)
                    mean_x, mean_y, mean_z = mask_pcd.mean(axis=0)
                    # self.fruit_pcd = mask_pcd
                    self.position = np.array([mean_x, mean_y, mean_z])

                    
                # Create a PoseStamped message
                pose_msg = Pepper()
                
                # Set header information
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.header.frame_id = "camera_depth_optical_frame"  # Use appropriate frame ID
    
                pose_msg.fruit_data.pose.position.x = self.position[0]
                pose_msg.fruit_data.pose.position.y = self.position[1]
                pose_msg.fruit_data.pose.position.z = self.position[2]
                
                # Set orientation (identity quaternion in this example)
                pose_msg.fruit_data.pose.orientation.x = 1.0
                pose_msg.fruit_data.pose.orientation.y = 0.0
                pose_msg.fruit_data.pose.orientation.z = 0.0
                pose_msg.fruit_data.pose.orientation.w = 0.0


                
                # Publish the pose
                self.pose_publisher.publish(pose_msg)
                rospy.loginfo("Published fruit pose")
            
            self.rate.sleep()

def main():
    try:
        node = FruitDetectionNode()
        node.process_data_and_publish()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated")

if __name__ == '__main__':
    main()