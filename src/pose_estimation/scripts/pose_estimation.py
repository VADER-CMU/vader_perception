import numpy as np
import pyrealsense2 as rs
import cv2
import open3d as o3d
from PIL import Image, ImageDraw
import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

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
        

        self.fx = 421.3145751953125
        self.fy = 421.3145751953125
        self.cx = 419.26629638671875
        self.cy = 244.68508911132812
        self.depth_scale = 0.001

    def coarse_fruit_pose_estimation(self, rgb, depth, mask):
        """
        Coarse fruit pose estimation gives the position and orientation of the fruit in the camera frame
        Args: rgb_image (np.ndarray): RGB image of size (640, 480, 3)
              depth_image (np.ndarray): Depth image of size (640, 480)
              mask (np.ndarray): A singular mask of a detected target fruit
        Returns: position (np.ndarray): Position of the fruit in the camera frame
                 quaternion (np.ndarray): Orientation of the fruit in the camera frame
        """
        pose = np.eye(4)
        pcd = self.rgbd_to_pcd(rgb, depth, mask, pose)
        center = pcd.get_center()
        quaternion = np.array([0, 0, 0, 1])
        return center, quaternion
    

    def fine_fruit_pose_estimation(self, rgb_image, depth_image, fruit_mask, peduncle_mask):
        """
        Fine fruit pose estimation gives the position and orientation of the fruit in the camera frame
        Args: rgb_image (np.ndarray): RGB image of size (640, 480, 3)
              depth_image (np.ndarray): Depth image of size (640, 480)
              mask (np.ndarray): A singular mask of a detected target fruit
        Returns: position (np.ndarray): Position of the fruit in the camera frame
                 quaternion (np.ndarray): Orientation of the fruit in the camera frame
        """
        pose = np.eye(4)
        fruit_pcd = self.rgbd_to_pcd(rgb_image, depth_image, fruit_mask, pose)
        fruit_center = fruit_pcd.get_center()

        peduncle_pcd = self.rgbd_to_pcd(rgb_image, depth_image, peduncle_mask, pose)
        peduncle_center = peduncle_pcd.get_center()

        position = fruit_center

        axis_vector = peduncle_center - fruit_center
        a_x = np.cross(axis_vector, fruit_center)
        a_x_hat = a_x/ np.linalg.norm(a_x)
        a_z = axis_vector #- (np.dot(axis_vector, a_x_hat)*a_x_hat)
        a_z_hat = a_z/ np.linalg.norm(a_z)
        a_y_hat = np.cross(a_z_hat, a_x_hat)
        
        R_co = np.array([a_x_hat, a_y_hat, a_z_hat]).T
        r = R.from_matrix(R_co)
        quaternion = r.as_quat()  # [x, y, z, w]
        
        # print(quaternion)
        return position, quaternion

    def rgbd_to_pcd(self, rgb, depth, mask, pose=np.eye(4)):
        """
        Converts RGBD and Mask data to a masked point cloud.
        Args:
            rgb (numpy.ndarray): The RGB image.
            depth (numpy.ndarray): The depth image.
            mask (numpy.ndarray): The mask to apply to the depth image.
            pose (numpy.ndarray, optional): The pose transformation matrix world frame to camera frame. Defaults to an identity matrix.
        Returns:
            open3d.geometry.PointCloud: The resulting masked point cloud.
        """

        masked_depth = mask*depth

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb),
                                                                  o3d.geometry.Image(masked_depth),
                                                                  depth_scale = 1/self.depth_scale,
                                                                  depth_trunc=1.0/self.depth_scale,
                                                                  convert_rgb_to_intensity=False)

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(height=rgb.shape[0],
                                 width=rgb.shape[1],
                                 fx=self.fx,
                                 fy=self.fy,
                                 cx=self.cx,
                                 cy=self.cy,
                                 )

        extrinsic = np.linalg.inv(pose)
        frame_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic)

        return frame_pcd

    def get_priority_mask(self, results):
        pass
        
        

