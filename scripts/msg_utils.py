import rospy
import numpy as np
import open3d
from geometry_msgs.msg import PoseStamped
from vader_msgs.msg import Pepper
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

def pack_pepper_message(position, quaternion=None, peduncle_position=None, frame_id="camera_depth_optical_frame"):
    """
    Packs the fruit and peduncle data into a Pepper message.
    Args:
        position (np.ndarray): Position of the fruit in the camera frame.
        quaternion (np.ndarray): Orientation of the fruit in the camera frame.
        peduncle_position (np.ndarray, optional): Position of the peduncle in the camera frame.
    Returns:
        Pepper: A Pepper message containing the fruit and peduncle data.
    """
    fine_pose_msg = Pepper()

    # Set header information
    fine_pose_msg.header.stamp = rospy.Time.now()
    fine_pose_msg.header.frame_id = frame_id

    fine_pose_msg.fruit_data.pose.position.x = position[0]
    fine_pose_msg.fruit_data.pose.position.y = position[1]
    fine_pose_msg.fruit_data.pose.position.z = position[2]

    if quaternion is None:
        # Default orientation (identity quaternion)
        quaternion = np.array([0, 0, 0, 1])

    # Set orientation (identity quaternion in this example)
    fine_pose_msg.fruit_data.pose.orientation.x = quaternion[0]
    fine_pose_msg.fruit_data.pose.orientation.y = quaternion[1]
    fine_pose_msg.fruit_data.pose.orientation.z = quaternion[2]
    fine_pose_msg.fruit_data.pose.orientation.w = quaternion[3]

    if peduncle_position is not None:
        # Set peduncle center with peduncle_position
        fine_pose_msg.peduncle_data.pose.position.x = peduncle_position[0]
        fine_pose_msg.peduncle_data.pose.position.y = peduncle_position[1]
        fine_pose_msg.peduncle_data.pose.position.z = peduncle_position[2]
        # Set peduncle orientation
        fine_pose_msg.peduncle_data.pose.orientation.x = quaternion[0]
        fine_pose_msg.peduncle_data.pose.orientation.y = quaternion[1]
        fine_pose_msg.peduncle_data.pose.orientation.z = quaternion[2]
        fine_pose_msg.peduncle_data.pose.orientation.w = quaternion[3]

    fine_pose_msg.fruit_data.shape.type = 3 #cylinder
    fine_pose_msg.fruit_data.shape.dimensions = [0.1, 0.075]
    fine_pose_msg.peduncle_data.shape.type = 3 #cylinder
    fine_pose_msg.peduncle_data.shape.dimensions = [0.02, 0.002]

    return fine_pose_msg

def pack_debug_fruit_message(position, quaternion=None, frame_id="camera_depth_optical_frame"):
    """
    Packs the fruit into a PoseStamped message.
    Args:
        position (np.ndarray): Position of the fruit in the camera frame.
        quaternion (np.ndarray): Orientation of the fruit in the camera frame.
    Returns:
        PoseStamped: A PoseStamped message containing the fruit pose.
    """
    debug_fine_pose_msg = PoseStamped()

    # Set header information
    debug_fine_pose_msg.header.stamp = rospy.Time.now()
    debug_fine_pose_msg.header.frame_id = frame_id

    debug_fine_pose_msg.pose.position.x = position[0]
    debug_fine_pose_msg.pose.position.y = position[1]
    debug_fine_pose_msg.pose.position.z = position[2]

    if quaternion is None:
        # Default orientation (identity quaternion)
        quaternion = np.array([0, 0, 0, 1])

    # Set orientation (identity quaternion in this example)
    debug_fine_pose_msg.pose.orientation.x = quaternion[0]
    debug_fine_pose_msg.pose.orientation.y = quaternion[1]
    debug_fine_pose_msg.pose.orientation.z = quaternion[2]
    debug_fine_pose_msg.pose.orientation.w = quaternion[3]

    return debug_fine_pose_msg

def pack_debug_pcd(fruit_pcd: open3d.geometry.PointCloud, frame_id="camera_depth_optical_frame"):
    """
    Packs the fruit point cloud into a PointCloud2 message.
    Args:
        fruit_pcd (open3d.geometry.PointCloud): The fruit point cloud.
    Returns:
        PointCloud2: A PointCloud2 message containing the fruit point cloud.
    """
    fruit_pcd_msg = PointCloud2()
    fruit_pcd_msg.header.stamp = rospy.Time.now()
    fruit_pcd_msg.header.frame_id = frame_id
    fruit_points = np.asarray(fruit_pcd.points)
    fruit_pcd_msg = pc2.create_cloud_xyz32(fruit_pcd_msg.header, fruit_points)

    return fruit_pcd_msg
