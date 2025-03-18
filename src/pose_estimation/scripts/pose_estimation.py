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

        # Small cable camera
        # fx:  421.3145751953125
        # fy:  421.3145751953125
        # ppx:  419.26629638671875
        # ppy:  244.68508911132812

        # Large cable camera
        # fx:  421.1239318847656
        # fy:  421.1239318847656
        # ppx:  419.2647705078125
        # ppy:  243.88404846191406
        

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


        segmentation_mask = np.pad(mask, ((0, 0), (104, 104)), mode='constant', constant_values=0)
        print("segmentation mask shape: ", segmentation_mask.shape)

        mask_pcd = self.unproject(full_point_cloud, segmentation_mask)
        mask_pcd.transform([
            [0, 0, 1, 0], 
            [0, 1, 0, 0], 
            [-1, 0, 0, 0], 
            [0, 0, 0, 1]
            ])
        mask_pcd = np.array(mask_pcd.points)
        
        print("mask pcd shape: ", mask_pcd.shape)
        
        mean_x, mean_y, mean_z = mask_pcd.mean(axis=0)



        position = np.array([mean_x, mean_y, mean_z])

        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        return mask_pcd
    

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
            y,x ,z = point
            if not z == 0:
                u = mask.shape[1] - 1 - int((x * self.intrinsics[0, 0] / z) + self.intrinsics[0, 2])
                v = int((y * self.intrinsics[1, 1] / z) + self.intrinsics[1, 2])   
                if 0 <= u < mask.shape[1] and 0 <= v < mask.shape[0]:
                    if mask[v,u]>128:
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

