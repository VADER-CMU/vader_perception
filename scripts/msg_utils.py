import rospy
import numpy as np
import open3d
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from vader_msgs.msg import Pepper, PepperArray
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, CameraInfo

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

def pack_debug_pose_array_message(pose_dict_array, frame_id="camera_depth_optical_frame"):
    """
    Packs the fruit into a PoseArray message.
    Args:
        position (np.ndarray): Position of the fruit in the camera frame.
        quaternion (np.ndarray): Orientation of the fruit in the camera frame.
    Returns:
        PoseArray: A PoseArray message containing the fruit pose.
    """
    debug_fine_pose_array_msg = PoseArray()

    # Set header information
    debug_fine_pose_array_msg.header.stamp = rospy.Time.now()
    debug_fine_pose_array_msg.header.frame_id = frame_id

    for pose_dict in pose_dict_array:

        if "fruit_position" in pose_dict:
            pose = Pose()
            pose.position.x = pose_dict['fruit_position'][0]
            pose.position.y = pose_dict['fruit_position'][1]
            pose.position.z = pose_dict['fruit_position'][2]

            if "fruit_quaternion" in pose_dict:
                # Default orientation (identity quaternion)
                quaternion = pose_dict['fruit_quaternion']
            else:
                quaternion = np.array([0, 0, 0, 1])

            # Set orientation (identity quaternion in this example)
            pose.orientation.x = quaternion[0]
            pose.orientation.y = quaternion[1]
            pose.orientation.z = quaternion[2]
            pose.orientation.w = quaternion[3]

            debug_fine_pose_array_msg.poses.append(pose)

    return debug_fine_pose_array_msg

def pack_pepper_array_message(pose_dict_array, fine=True, frame_id="camera_depth_optical_frame"):
    """
    Packs the pepper into a PepperArray message.
    Args:
        pose_dict_array (list): List of dictionaries containing pose information.
        fine (bool): Whether to include peduncle data.
        frame_id (str): The frame ID for the message.
    Returns:
        PepperArray: A PepperArray message containing the pepper poses.
    """
    pepper_array = PepperArray()
    pepper_array.header.stamp = rospy.Time.now()
    pepper_array.header.frame_id = frame_id

    for pose_dict in pose_dict_array:
        if "fruit_position" not in pose_dict:
            continue

        pepper = Pepper()
        position = pose_dict['fruit_position']
        pepper.fruit_data.pose.position.x = position[0]
        pepper.fruit_data.pose.position.y = position[1]
        pepper.fruit_data.pose.position.z = position[2]

        if fine and "fruit_quaternion" in pose_dict:
            quaternion = pose_dict['fruit_quaternion']
            pepper.fruit_data.pose.orientation.x = quaternion[0]
            pepper.fruit_data.pose.orientation.y = quaternion[1]
            pepper.fruit_data.pose.orientation.z = quaternion[2]
            pepper.fruit_data.pose.orientation.w = quaternion[3]

            if "peduncle_position" in pose_dict:
                ped_pos = pose_dict['peduncle_position']
                pepper.peduncle_data.pose.position.x = ped_pos[0]
                pepper.peduncle_data.pose.position.y = ped_pos[1]
                pepper.peduncle_data.pose.position.z = ped_pos[2]

            if "peduncle_quaternion" in pose_dict:
                ped_quat = pose_dict['peduncle_quaternion']
                pepper.peduncle_data.pose.orientation.x = ped_quat[0]
                pepper.peduncle_data.pose.orientation.y = ped_quat[1]
                pepper.peduncle_data.pose.orientation.z = ped_quat[2]
                pepper.peduncle_data.pose.orientation.w = ped_quat[3]

        elif fine and "fruit_quaternion" not in pose_dict:
            continue
        else:
            quaternion = np.array([0, 0, 0, 1])
            pepper.fruit_data.pose.orientation.x = quaternion[0]
            pepper.fruit_data.pose.orientation.y = quaternion[1]
            pepper.fruit_data.pose.orientation.z = quaternion[2]
            pepper.fruit_data.pose.orientation.w = quaternion[3]

        pepper_array.poses.append(pepper)

    return pepper_array


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

def wait_for_camera_info(topic_name: str, timeout: float = 5.0):
    """
    Waits for camera info message on the specified topic.
    Args:
        topic_name (str): The topic name to wait for the camera info message.
        timeout (float): The maximum time to wait for the message in seconds.
    Returns:
        CameraInfo: The received camera info message.
    """
    try:
        cam_info_msg = rospy.wait_for_message(topic_name, CameraInfo, timeout=timeout)
        fx = cam_info_msg.K[0]
        fy = cam_info_msg.K[4]
        cx = cam_info_msg.K[2]
        cy = cam_info_msg.K[5]
        camera_info = {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy
        }
        return camera_info
    except rospy.ROSException as e:
        rospy.logerr(f"Timeout while waiting for camera info on topic {topic_name}: {e}")
        return None
