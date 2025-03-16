import pyrealsense2 as rs
import cv2
import numpy as np
import open3d as o3d


import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
model = YOLO("/home/vader/VADER/perception/pose_estimation/src/pose_estimation/weights/yolov8l-seg-300.pt")

def viz_depth_map():
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    
    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)
    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    try:

        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            grey_color = 153
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

            # Render images:
            #   depth align to color on left
            #   depth on right
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((bg_removed, depth_colormap))

            # cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            cv2.imshow('Align Example', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    finally:
        pipeline.stop()


def viz_pointcloud(model=model, isolate_pepper=True):
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    
    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)
    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale
    pointcloud = o3d.geometry.PointCloud()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.1, origin=[0, 0, 0]
                )

    try:

        while True:

            def update_points(vis):
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    return False

                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Get intrinsics for the depth camera
                depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()

                # Convert color to RGB (Open3D uses RGB format)
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

                depth_image_mask = None

                if isolate_pepper:
                    results = model.predict(color_image[:, 104:744, :], conf=0.8)
                    result = results[0]
                    masks = result.masks
                    
                    if not masks is None:
                        mask_img = masks.data[0].cpu().numpy().astype('uint8') * 255
                        
                        depth_image_mask = np.where((mask_img >128), depth_image[:,104:744], mask_img)
                        depth_image_mask = np.pad(depth_image_mask, ((0, 0), (104, 104)), mode='constant', constant_values=0)

                intrinsics = o3d.camera.PinholeCameraIntrinsic(
                    width=depth_intrinsics.width,
                    height=depth_intrinsics.height,
                    fx=depth_intrinsics.fx,
                    fy=depth_intrinsics.fy,
                    cx=depth_intrinsics.ppx,
                    cy=depth_intrinsics.ppy
                )

                # Create RGBD image
                print("color image shape: ", color_image.shape)
                print("depth image shape: ", depth_image.shape)
                color_o3d = o3d.geometry.Image(color_image)
                depth_o3d = o3d.geometry.Image(depth_image)
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_o3d, depth_o3d,
                    depth_scale=1.0/depth_scale,
                    depth_trunc=clipping_distance_in_meters,
                    convert_rgb_to_intensity=False
                )

                # Generate point cloud
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd_image,
                    intrinsics
                )

                # Transform to better viewing orientation
                pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

                if not depth_image_mask is None:
                    print("color image shape: ", color_image.shape)
                    print("depth image mask shape: ", depth_image_mask.shape)
                    color_o3d = o3d.geometry.Image(color_image)
                    depth_o3d_mask = o3d.geometry.Image(depth_image_mask)
                    rgbd_image_mask = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        color_o3d, depth_o3d_mask,
                        depth_scale=1.0/depth_scale,
                        depth_trunc=clipping_distance_in_meters,
                        convert_rgb_to_intensity=False
                    )

                    # Generate point cloud
                    pcd_mask = o3d.geometry.PointCloud.create_from_rgbd_image(
                        rgbd_image_mask,
                        intrinsics
                    )

                    # Transform to better viewing orientation
                    pcd_mask.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

                    # Calculate mean x, y, z coordinates
                    pts = np.asarray(pcd_mask.points)
                    mean_x, mean_y, mean_z = pts.mean(axis=0)
                    mesh_frame.translate([mean_x, mean_y, mean_z], relative=False)
                # print(f"Mean X: {mean_x}, Mean Y: {mean_y}, Mean Z: {mean_z}")
                # Add mean point to the point cloud
                # mean_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                # mean_point.paint_uniform_color([1, 0, 0])  # Red color
                # mean_point.translate([mean_x, mean_y, mean_z])
                
                # vis.update_geometry(mesh_frame)
                pointcloud.points = pcd.points
                pointcloud.colors = pcd.colors

                
                
                if not masks is None and isolate_pepper:
                    # vis.update_geometry(pointcloud)
                    vis.update_geometry(mesh_frame)
                    # o3d.visualization.draw_geometries([pointcloud, mesh_frame])
                    # vis.add_geometry(mesh_frame)
                vis.update_geometry(pointcloud)
                return False
            
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.register_animation_callback(update_points)
            vis.add_geometry(pointcloud)
            vis.add_geometry(mesh_frame)
            vis.run()
            vis.destroy_window()

    finally:
        pipeline.stop()
        vis.destroy_window()

if __name__ == "__main__":
    # viz_depth_map()

    viz_pointcloud(isolate_pepper=True)