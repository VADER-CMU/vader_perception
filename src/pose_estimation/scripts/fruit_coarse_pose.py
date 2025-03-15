#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PoseStamped
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.point_cloud2 as pc2
from pose_estimation import PoseEstimation
from segmentation import Segmentation

class FruitDetectionNode:
    def __init__(self):

        rospy.init_node('fruit_coarse_pose', anonymous=True)

        self.latest_pointcloud = None
        self.latest_image = None
        
        self.pose_publisher = rospy.Publisher('fruit_pose', PoseStamped, queue_size=10)

        rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.pointcloud_callback)
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        self.rate = rospy.Rate(10)
        
        rospy.loginfo("Fruit detection node initialized")
    
    def pointcloud_callback(self, msg):
        self.latest_pointcloud = msg
        pointcloud = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array(list(pointcloud))
        self.latest_pointcloud = points

    
    def image_callback(self, msg):

        self.latest_image = msg
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
            if self.latest_pointcloud is not None and self.latest_image is not None:
                # Create a PoseStamped message
                pose_msg = PoseStamped()
                
                # Set header information
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.header.frame_id = "camera_frame"  # Use appropriate frame ID
                
                
                results = Seg.infer(self.latest_image)
                result = results[0]
                masks = result.masks
                masks.data[0].cpu().numpy()
                print("masks: ", masks.shape)


                # position, quaternion = PoseEst.coarse_fruit_pose_estimation(
                #                             self.latest_pointcloud, 
                #                             self.latest_image
                #                         )
                position = np.array([0.0, 0.0, 0.0])
    
                pose_msg.pose.position.x = position[0]
                pose_msg.pose.position.y = position[1]
                pose_msg.pose.position.z = position[2]
                
                # Set orientation (identity quaternion in this example)
                pose_msg.pose.orientation.x = 1.0
                pose_msg.pose.orientation.y = 0.0
                pose_msg.pose.orientation.z = 0.0
                pose_msg.pose.orientation.w = 0.0
                
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