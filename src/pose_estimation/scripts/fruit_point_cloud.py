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
import matplotlib.pyplot as plt
class FruitDetectionNode:
    def __init__(self):

        rospy.init_node('fruit_pcd', anonymous=True)

        self.latest_pointcloud = None
        self.latest_image = None
        self.fruit_pcd = None
        
        self.pcd_publisher = rospy.Publisher('fruit_pcd', PointCloud2, queue_size=10)

        rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.pointcloud_callback)
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        self.rate = rospy.Rate(10)
        
        rospy.loginfo("Fruit detection node initialized")
    
    def pointcloud_callback(self, msg):

        pointcloud = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array(list(pointcloud))
        self.latest_pointcloud = points

    
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
            if self.latest_pointcloud is not None and self.latest_image is not None:
                # # Create a PoseStamped message
                # pose_msg = PoseStamped()
                
                # # Set header information
                # pose_msg.header.stamp = rospy.Time.now()
                # pose_msg.header.frame_id = "camera_link"  # Use appropriate frame ID
                self.fruit_pcd = self.latest_pointcloud
                
                print("Latest Image shape: ", self.latest_image.shape)
                results = Seg.infer(self.latest_image[:, 104:744, :], confidence=0.8)
                result = results[0]
                masks = result.masks
                if not masks is None:
                    mask = masks.data[0].cpu().numpy().astype('uint8') * 255
                    print("masks: ", mask.shape)

                    # cv2.imshow("mask", mask)
                    # cv2.waitKey(1)
                    print("Latest Pointcloud shape: ", self.latest_pointcloud.shape)
                    if(len(self.latest_pointcloud) == 407040):
                        self.fruit_pcd = PoseEst.coarse_fruit_pose_estimation(
                                                    self.latest_pointcloud, 
                                                    mask
                                                )
                        print("fruit_pcd shape: ", self.fruit_pcd.shape)
                
    
                # Convert fruit_pcd to PointCloud2 message
                header = rospy.Header()
                header.stamp = rospy.Time.now()
                header.frame_id = "camera_link"
                
                fruit_pcd_msg = pc2.create_cloud_xyz32(header, self.fruit_pcd)
                
                # Publish the PointCloud2 message
                self.pcd_publisher.publish(fruit_pcd_msg)
                rospy.loginfo("Published fruit pcd")
            
            self.rate.sleep()

def main():
    try:
        node = FruitDetectionNode()
        node.process_data_and_publish()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated")

if __name__ == '__main__':
    main()