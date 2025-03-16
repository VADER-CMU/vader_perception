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


        # camer depth intrinsics
        # 421.53094482421875, 0.0, 419.7398986816406, 0.0, 421.53094482421875, 243.58074951171875, 0.0, 0.0, 1.0
        self.fx = 421.53094482421875
        self.fy = 421.53094482421875
        self.cx = 419.7398986816406
        self.cy = 243.58074951171875

        # 429.0426025390625, 0.0, 417.195068359375, 0.0, 428.5399475097656, 245.24171447753906, 0.0, 0.0, 1.0
        # self.fx = 429.0426025390625
        # self.fy = 428.5399475097656
        # self.cx = 417.195068359375
        # self.cy = 245.24171447753906
        self.intrinsics = np.array([
            [self.fx, 0,  self.cx],
            [0,  self.fy, self.cy],
            [0,  0,  1]
        ])

        self.depth_intrinsics = rs.pyrealsense2.intrinsics()
        self.depth_intrinsics.height = 480
        self.depth_intrinsics.width = 848
        self.depth_intrinsics.ppx = self.fx
        self.depth_intrinsics.ppy = self.fy
        self.depth_intrinsics.fx = self.cx
        self.depth_intrinsics.fy = self.cy
        self.depth_intrinsics.model = rs.pyrealsense2.distortion.inverse_brown_conrady
        self.depth_intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.depth_scale = 0.001

    def coarse_fruit_pose_estimation(self, depth_image, mask):
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

        points = []
        for v in range(segmentation_mask.shape[0]):
            for u in range(segmentation_mask.shape[1]):
                if segmentation_mask[v, u] > 128:
                    depth_in_meters, point = self.get_depth_at_point(depth_image, u, v)
                    x,y,z = point
                    y+=0.2
                    if depth_in_meters > 0:
                        points.append([x, y, z])
        
        # mask_pcd = self.unproject(points, segmentation_mask)
        mask_pcd = o3d.geometry.PointCloud()
        mask_pcd.points = o3d.utility.Vector3dVector(points)
        # mask_pcd.transform([[1, 0, 0, 0], 
        #                     [0, 1, 0, 0.2], 
        #                     [0, 0, 1, 0], 
        #                     [0, 0, 0, 1]])
        
        pts = np.asarray(mask_pcd.points)
        
        

        # Get the indices of the pixels in the mask
        return pts
    

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
                    if mask[u,v]>128:
                        # colors[i] = [1,0,0]
                        new_points.append(point)
                        colors.append(points[i])

        mask_pcd = o3d.geometry.PointCloud()
        mask_pcd.points = o3d.utility.Vector3dVector(new_points)
        mask_pcd.colors = o3d.utility.Vector3dVector(colors)
        return mask_pcd
    
    def get_depth_at_point(self, depth_image, x, y):
        
        depth_value = depth_image[y, x]

        # Calculate real-world coordinates
        depth_in_meters = depth_value * self.depth_scale
        pixel = [float(x), float(y)]  # Convert pixel coordinates to floats
        point = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, pixel, depth_in_meters)

        return depth_in_meters, point

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

