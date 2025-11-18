#!/usr/bin/env python3

import rospy
import sys
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray
from vader_msgs.msg import HarvestResult
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QFrame, QTextEdit)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QObject, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import tf2_ros
import tf2_geometry_msgs
import threading


class CameraWidget(QWidget):
    """Widget to display camera feed with pose array overlay"""
    
    # Signal to safely update UI from ROS callbacks
    image_ready = pyqtSignal()
    
    def __init__(self, title, image_topic, pose_topic, frame_id, camera_info_topic):
        super().__init__()
        self.title = title
        self.image_topic = image_topic
        self.pose_topic = pose_topic
        self.frame_id = frame_id
        self.camera_info_topic = camera_info_topic
        self.bridge = CvBridge()
        
        # Thread-safe data storage with lock
        self.data_lock = threading.Lock()
        self.current_image = None
        self.current_poses = None
        self.display_image = None  # Processed image ready for display
        
        # Camera calibration data
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False
        
        # TF buffer for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Setup UI
        self.setup_ui()
        
        # Connect signal to update method
        self.image_ready.connect(self.update_display_safe)
        
        # Setup ROS subscribers with smaller queue to prevent backlog
        self.camera_info_sub = rospy.Subscriber(
            self.camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1
        )
        self.image_sub = rospy.Subscriber(
            self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24
        )
        self.pose_sub = rospy.Subscriber(
            self.pose_topic, PoseArray, self.pose_callback, queue_size=1
        )
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Title label
        title_label = QLabel(self.title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)
        
        # Image display label
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #333; background-color: black;")
        self.image_label.setScaledContents(False)
        layout.addWidget(self.image_label)
        
        # Status label
        self.status_label = QLabel("Waiting for data...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def camera_info_callback(self, msg):
        """Callback for camera info - sets calibration once"""
        if not self.camera_info_received:
            with self.data_lock:
                # Extract camera matrix (K) from CameraInfo
                # K is a 3x3 row-major matrix: [fx 0 cx, 0 fy cy, 0 0 1]
                self.camera_matrix = np.array([
                    [msg.K[0], msg.K[1], msg.K[2]],
                    [msg.K[3], msg.K[4], msg.K[5]],
                    [msg.K[6], msg.K[7], msg.K[8]]
                ], dtype=np.float32)
                
                # Extract distortion coefficients
                self.dist_coeffs = np.array(msg.D, dtype=np.float32)
                
                self.camera_info_received = True
                
            rospy.loginfo(f"{self.title}: Camera calibration received")
            rospy.loginfo(f"  fx={msg.K[0]:.2f}, fy={msg.K[4]:.2f}, cx={msg.K[2]:.2f}, cy={msg.K[5]:.2f}")
            
            # Unsubscribe after receiving camera info once
            self.camera_info_sub.unregister()
        
    def image_callback(self, msg):
        """Callback for image messages - runs in ROS thread"""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Process image with pose overlay in this thread
            processed_image = self.process_image(cv_image)
            
            # Thread-safe update of display image
            with self.data_lock:
                self.display_image = processed_image
            
            # Emit signal to update UI in main thread
            self.image_ready.emit()
            
        except Exception as e:
            rospy.logerr(f"Error processing image in {self.title}: {e}")
            
    def pose_callback(self, msg):
        """Callback for PoseArray messages - runs in ROS thread"""
        with self.data_lock:
            self.current_poses = msg
        
    def process_image(self, cv_image):
        """Process image with pose overlays - called in ROS thread"""
        display_image = cv_image.copy()
        
        # Get current poses and camera calibration safely
        with self.data_lock:
            poses = self.current_poses
            camera_matrix = self.camera_matrix
            dist_coeffs = self.dist_coeffs
        
        # Only draw poses if we have camera calibration
        if camera_matrix is not None and poses is not None and len(poses.poses) > 0:
            # Draw each pose
            for pose in poses.poses:
                try:
                    self.draw_pose_axes(display_image, pose, camera_matrix, dist_coeffs)
                except Exception as e:
                    rospy.logwarn_throttle(5.0, f"Error drawing pose in {self.title}: {e}")
        
        return display_image
        
    def draw_pose_axes(self, image, pose, camera_matrix, dist_coeffs):
        """Draw 3D axes for a pose on the image"""
        try:
            # Define axis points in 3D (origin and three axis endpoints)
            axis_length = 0.05  # 5cm axes
            axis_points_3d = np.float32([
                [0, 0, 0],  # origin
                [axis_length, 0, 0],  # X axis (red)
                [0, axis_length, 0],  # Y axis (green)
                [0, 0, axis_length]   # Z axis (blue)
            ]).reshape(-1, 3)
            
            # Extract rotation and translation from pose
            position = pose.position
            orientation = pose.orientation
            
            # Convert quaternion to rotation matrix
            from scipy.spatial.transform import Rotation
            r = Rotation.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
            
            # Create translation and rotation vectors for cv2.projectPoints
            rvec = r.as_rotvec()
            tvec = np.array([position.x, position.y, position.z])
            
            # Project 3D points to 2D image plane
            image_points, _ = cv2.projectPoints(
                axis_points_3d, rvec, tvec, camera_matrix, dist_coeffs
            )
            image_points = image_points.reshape(-1, 2).astype(int)
            
            # Check if points are within image bounds
            h, w = image.shape[:2]
            if all(0 <= pt[0] < w and 0 <= pt[1] < h for pt in image_points):
                # Draw the axes
                origin = tuple(image_points[0])
                x_end = tuple(image_points[1])
                y_end = tuple(image_points[2])
                z_end = tuple(image_points[3])
                
                # X axis - Red
                cv2.line(image, origin, x_end, (0, 0, 255), 3)
                # Y axis - Green
                cv2.line(image, origin, y_end, (0, 255, 0), 3)
                # Z axis - Blue
                cv2.line(image, origin, z_end, (255, 0, 0), 3)
                
                # Draw origin point
                cv2.circle(image, origin, 5, (255, 255, 255), -1)
                
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Could not project points: {e}")
    
    @pyqtSlot()
    def update_display_safe(self):
        """Update the display - called in main Qt thread"""
        with self.data_lock:
            if self.display_image is None:
                return
            
            # Make a copy to avoid any threading issues
            display_image = self.display_image.copy()
            poses = self.current_poses
        
        # Update status label
        if poses is not None and len(poses.poses) > 0:
            self.status_label.setText(f"Displaying {len(poses.poses)} poses")
        else:
            self.status_label.setText("No poses detected")
        
        # Convert to Qt format - CRITICAL: copy data to ensure it persists
        height, width, channel = display_image.shape
        bytes_per_line = 3 * width
        
        # Create QImage with copied data to prevent segfault
        # The .copy() ensures QImage owns its data
        q_image = QImage(display_image.data, width, height, bytes_per_line, 
                        QImage.Format_RGB888).rgbSwapped().copy()
        
        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)


class StatusWidget(QWidget):
    """Widget to display system status"""
    
    def __init__(self):
        super().__init__()
        
        # Harvest status data
        self.harvest_result_code = None
        self.harvest_reason = "No harvest data received"
        self.status_lock = threading.Lock()
        
        self.setup_ui()
        
        # Subscribe to harvest status
        self.harvest_status_sub = rospy.Subscriber(
            "/harvest_status", HarvestResult, self.harvest_status_callback, queue_size=1
        )
        
        # Setup timer to update status
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_status)
        self.update_timer.start(1000)  # Update every second
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("System Status")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(title)
        
        # Status text area
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(200)
        self.status_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 2px solid #333;
                font-family: monospace;
                font-size: 11px;
            }
        """)
        layout.addWidget(self.status_text)
        
        self.setLayout(layout)
    
    def harvest_status_callback(self, msg):
        """Callback for harvest status messages"""
        with self.status_lock:
            self.harvest_result_code = msg.result
            self.harvest_reason = msg.reason
        
    def update_status(self):
        """Update status display with dummy data"""
        import datetime
        
        # Get harvest status safely
        with self.status_lock:
            result_code = self.harvest_result_code
            reason = self.harvest_reason
        
        # Format harvest status
        if result_code is not None:
            if result_code == 100:  # RESULT_SUCCESS
                harvest_status = f"HARVEST RESULT: {result_code} (SUCCESS)"
            else:
                harvest_status = f"HARVEST RESULT: {result_code} (FAILED)"
            harvest_status += f"\nREASON: {reason}"
        else:
            harvest_status = "HARVEST RESULT: Not available\nREASON: No harvest data received"
        
        status_lines = [
            f"=== Robot Status [{datetime.datetime.now().strftime('%H:%M:%S')}] ===",
            "",
            "Harvest Status:",
            f"  {harvest_status}",
            "",
            "Robot State: OPERATIONAL"
        ]
        
        self.status_text.setPlainText("\n".join(status_lines))


class RobotUIMainWindow(QMainWindow):
    """Main window for robot UI"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VADER: Vision-based Autonomous Dexterous Reaper")
        self.setMinimumSize(1400, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create camera displays
        camera_layout = QHBoxLayout()
        
        # Gripper camera
        self.gripper_cam = CameraWidget(
            "Gripper Camera",
            "/gripper_cam/color/image_raw",
            "gripper_debug_fine_pose_array",
            "gripper_cam_depth_optical_frame",
            "/gripper_cam/color/camera_info"
        )
        camera_layout.addWidget(self.gripper_cam)
        
        # Cutter camera
        self.cutter_cam = CameraWidget(
            "Cutter Camera",
            "/cutter_cam/color/image_raw",
            "cutter_debug_fine_pose_array",
            "cutter_cam_depth_optical_frame",
            "/cutter_cam/color/camera_info"
        )
        camera_layout.addWidget(self.cutter_cam)
        
        main_layout.addLayout(camera_layout)
        
        # Add status widget
        self.status_widget = StatusWidget()
        main_layout.addWidget(self.status_widget)
    
    def closeEvent(self, event):
        """Handle window close event"""
        rospy.signal_shutdown("GUI closed")
        event.accept()


def main():
    # Initialize ROS node
    rospy.init_node('robot_ui_node', anonymous=True)
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = RobotUIMainWindow()
    window.show()
    
    rospy.loginfo("Robot UI started successfully")
    
    # Run Qt event loop
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down Robot UI")
        rospy.signal_shutdown("User requested shutdown")
        sys.exit(0)


if __name__ == '__main__':
    main()