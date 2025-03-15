#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PoseStamped

class FruitDetectionNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('fruit_detection_node', anonymous=True)
        
        # Store the latest messages from each topic
        self.latest_pointcloud = None
        self.latest_image = None
        
        # Create publisher for fruit pose
        self.pose_publisher = rospy.Publisher('fruit_pose', PoseStamped, queue_size=10)
        
        # Create subscribers for the two input topics
        rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.pointcloud_callback)
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        
        # Processing rate (10 Hz)
        self.rate = rospy.Rate(10)
        
        rospy.loginfo("Fruit detection node initialized")
    
    def pointcloud_callback(self, msg):
        # Store the latest pointcloud message
        self.latest_pointcloud = msg
        rospy.logdebug("Received pointcloud data")
    
    def image_callback(self, msg):
        # Store the latest image message
        self.latest_image = msg
        rospy.logdebug("Received image data")
    
    def process_data_and_publish(self):
        # Main processing loop
        while not rospy.is_shutdown():
            # Check if we have received both messages
            if self.latest_pointcloud is not None and self.latest_image is not None:
                # Create a PoseStamped message
                pose_msg = PoseStamped()
                
                # Set header information
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.header.frame_id = "camera_frame"  # Use appropriate frame ID
                
                # Here you would implement fruit detection algorithm
                # to identify fruits in the image and get their 3D positions from pointcloud
                # This is a placeholder for the actual fruit detection logic
                
                # Example pose (would be replaced with actual detected positions)
                pose_msg.pose.position.x = 0.1
                pose_msg.pose.position.y = 0.2
                pose_msg.pose.position.z = 0.3
                
                # Set orientation (identity quaternion in this example)
                pose_msg.pose.orientation.x = 0.0
                pose_msg.pose.orientation.y = 0.0
                pose_msg.pose.orientation.z = 0.0
                pose_msg.pose.orientation.w = 1.0
                
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