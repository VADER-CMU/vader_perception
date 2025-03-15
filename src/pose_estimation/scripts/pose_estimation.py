import numpy as np
import pyrealsense2 as rs
import cv2
import open3d as o3d
from PIL import Image, ImageDraw
import numpy as np
import yaml
import matplotlib.pyplot as plt

class PoseEstimation:
    def __init__(self):
        # with open('src/pose_estimation/config/gripper_camera.yaml', 'r') as file:
        #     config = yaml.safe_load(file)

        # self.fx = config['camera_intrinsics']['fx']
        # self.fy = config['camera_intrinsics']['fy']
        # self.cx = config['camera_intrinsics']['cx']
        # self.cy = config['camera_intrinsics']['cy']

        self.fx = 429.0426025390625
        self.fy = 428.5399475097656
        self.cx = 417.195068359375
        self.cy = 245.24171447753906

        self.intrinsics = np.array([
            [self.fx, 0,  self.cx],
            [0,  self.fy, self.cy],
            [0,  0,  1]
        ])

    def coarse_fruit_pose_estimation(self, full_point_cloud, mask):
        """
        Coarse fruit pose estimation gives the position and orientation of the fruit in the camera frame
        Args: rgb_image (np.ndarray): RGB image of size (640, 480, 3)
              depth_image (np.ndarray): Depth image of size (640, 480)
              mask (np.ndarray): A singular mask of a detected target fruit
        Returns: position (np.ndarray): Position of the fruit in the camera frame
                 quaternion (np.ndarray): Orientation of the fruit in the camera frame
        """

        mask_pcd = self.unproject(full_point_cloud, mask)
        pts = np.asarray(mask_pcd.points)
        mean_x, mean_y, mean_z = pts.mean(axis=0)



        position = np.array([mean_x, mean_y, mean_z])

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
    
    def unproject(self, points, mask):

        mask = np.fliplr(mask)
        colors = []
        new_points = []
        for i,point in enumerate(points):
            x,y,z = point
            u = mask.shape[1] - 1 - int((x * self.intrinsics[0, 0] / z) + self.intrinsics[0, 2])
            v = int((y * self.intrinsics[1, 1] / z) + self.intrinsics[1, 2])   
            if 0 <= u < mask.shape[1] and 0 <= v < mask.shape[0]:
                if mask[v,u]:
                    # colors[i] = [1,0,0]
                    new_points.append(point)
                    colors.append(points[i])

        mask_pcd = o3d.geometry.PointCloud()
        mask_pcd.points = o3d.utility.Vector3dVector(new_points)
        mask_pcd.colors = o3d.utility.Vector3dVector(colors)
        return mask_pcd

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

