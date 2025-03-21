#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import pyrealsense2 as rs
import sensor_msgs.point_cloud2 as pc2
from pose_estimation import PoseEstimation
from segmentation import Segmentation
from sensor_msgs.msg import PointCloud2

class FruitDetectionNode:
    def __init__(self):

        rospy.init_node('fruit_pcd', anonymous=True)

        self.latest_depth = None
        self.latest_image = None
        self.fruit_pcd = []
        
        self.pcd_publisher = rospy.Publisher('fruit_pcd', PointCloud2, queue_size=10)

        rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)
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
                # Convert fruit_pcd to PointCloud2 message
                header = rospy.Header()
                header.stamp = rospy.Time.now()
                # cv2.imwrite(f'/home/vader/VADER/perception/pose_estimation/raw_data/rgb_{header.stamp}.png', self.latest_image)
                # cv2.imwrite(f'/home/vader/VADER/perception/pose_estimation/raw_data2/depth_{header.stamp}.png', self.latest_depth)
                depth_array = np.array(self.latest_depth, dtype=np.float32)
                # np.save(f'/home/vader/VADER/perception/pose_estimation/raw_data/depth_{header.stamp}.npy', depth_array)
                print("Latest Depth shape: ", depth_array.shape)
                # print("Latest Image shape: ", self.latest_image.shape)
                results = Seg.infer(self.latest_image[:, 104:744, :], confidence=0.8)
                result = results[0]
                masks = result.masks
                if not masks is None:
                    mask = masks.data[0].cpu().numpy().astype('uint16')
                    segmentation_mask = np.pad(mask, ((0, 0), (104, 104)), mode='constant', constant_values=0)
                    # print("segmentation mask shape: ", segmentation_mask.shape)
                    # cv2.imwrite(f'/home/vader/VADER/perception/pose_estimation/raw_data/mask_{header.stamp}.png', segmentation_mask)
                  
                    mask_pcd = PoseEst.rgbd_to_pcd(self.latest_image, self.latest_depth, segmentation_mask)
                    mask_pcd = np.asarray(mask_pcd.points)
                    self.fruit_pcd = mask_pcd

                
    
                header.frame_id = "camera_depth_optical_frame" 
                
                fruit_pcd_msg = pc2.create_cloud_xyz32(header, self.fruit_pcd)
                
                # Publish the PointCloud2 message
                self.pcd_publisher.publish(fruit_pcd_msg)
                rospy.loginfo("Published fruit pcd2")
            
            self.rate.sleep()

def main():
    try:
        node = FruitDetectionNode()
        node.process_data_and_publish()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated")

if __name__ == '__main__':
    main()