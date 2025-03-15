import numpy as np
import pyrealsense2 as rs
import cv2
import open3d as o3d
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

class PoseEstimation:
    def __init__(self):
        pass

    def coarse_fruit_pose_estimation(self, rgb_image, depth_image, mask, intrinsics):
        """
        Coarse fruit pose estimation gives the position and orientation of the fruit in the camera frame
        Args: rgb_image (np.ndarray): RGB image of size (640, 480, 3)
              depth_image (np.ndarray): Depth image of size (640, 480)
              mask (np.ndarray): A singular mask of a detected target fruit
              intrinsics (dictionary): Camera intrinsics matrix []
        Returns: position (np.ndarray): Position of the fruit in the camera frame
                 quaternion (np.ndarray): Orientation of the fruit in the camera frame
        """

        # Placeholder implementation for quaternion

        # Placeholder implementation for position
        position = np.array([0.0, 0.0, 0.0])
        # A quaternion is represented as [w, x, y, z]
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        return position, quaternion
    

    def fine_fruit_pose_estimation(self, rgb_image, depth_image, mask):
        """
        Fine fruit pose estimation gives the position and orientation of the fruit in the camera frame
        Args: rgb_image (np.ndarray): RGB image of size (640, 480, 3)
              depth_image (np.ndarray): Depth image of size (640, 480)
              mask (np.ndarray): A singular mask of a detected target fruit
        Returns: position (np.ndarray): Position of the fruit in the camera frame
                 quaternion (np.ndarray): Orientation of the fruit in the camera frame
        """

        # Placeholder implementation for quaternion

        # Placeholder implementation for position
        position = np.array([0.0, 0.0, 0.0])
        # A quaternion is represented as [w, x, y, z]
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        return position, quaternion
    

class PriorityPolicy:
    def __init__(self):
        pass

    def get_priority(self, fruit_list):
        """
        Get the priority of the fruit based on the policy
        Args: fruit_list (list): List of fruits detected in the image
        Returns: priority_list (list): List of priorities for each fruit
        """

        """
        Ideas largest visible fruit, nearest fruit, fruit with highest confidence
        """
        priority_list = [0.0] * len(fruit_list)
        return priority_list

