import numpy as np
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import gdown
import pathlib
from ultralytics import YOLO
import cv2

class PoseEstimation:
    def __init__(self, intrinsics=None): 

        # Default intrinsics for Intel Realsense D405
        self.fx = 421.3145751953125
        self.fy = 421.3145751953125
        self.cx = 419.26629638671875
        self.cy = 244.68508911132812
        self.depth_scale = 0.001

        if intrinsics is not None:
            self.fx = intrinsics.get("fx", self.fx)
            self.fy = intrinsics.get("fy", self.fy)
            self.cx = intrinsics.get("cx", self.cx)
            self.cy = intrinsics.get("cy", self.cy)
    
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


    def _coarse_fruit_pose_estimation(self, rgb, depth, mask):
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
        quaternion = None
        return center, quaternion, pcd
    

    def _fine_fruit_pose_estimation(self, rgb_image, depth_image, fruit_mask, peduncle_mask):
        """
        Fine fruit pose estimation gives the position and orientation of the fruit in the camera frame
        Args: rgb_image (np.ndarray): RGB image of size (640, 480, 3)
              depth_image (np.ndarray): Depth image of size (640, 480)
              mask (np.ndarray): A singular mask of a detected target fruit
        Returns: position (np.ndarray): Position of the fruit in the camera frame
                 quaternion (np.ndarray): Orientation of the fruit in the camera frame
                 peduncle_center (np.ndarray): Center of the peduncle in the camera frame
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
        return position, quaternion, peduncle_center, fruit_pcd

    def pose_estimation(self, rgb_image, depth_image, masks, superellipsoid_method=True, offset=[0,0,0.02]):
        """
        Fine fruit pose estimation gives the position and orientation of the fruit in the camera frame
        Args: rgb_image (np.ndarray): RGB image of size (640, 480, 3)
              depth_image (np.ndarray): Depth image of size (640, 480)
              mask (np.ndarray): A singular mask of a detected target fruit
        Returns: position (np.ndarray): Position of the fruit in the camera frame
                 quaternion (np.ndarray): Orientation of the fruit in the camera frame
                 peduncle_center (np.ndarray): Center of the peduncle in the camera frame
        """
        pose = np.eye(4)
        fruit_pcd = self.rgbd_to_pcd(rgb_image, depth_image, masks["fruit_mask"], pose)

        offset = [0, 0, 0.02]



        result = {
            "fruit_position": fruit_pcd.get_center() + offset,
            "fruit_quaternion": [0,0,0,1],
            "fruit_pcd": fruit_pcd
        }

        if "peduncle_mask" in masks:
            
            # print("Peduncle detected")
            # Bad estimate of the center of the fruit
            refined_fruit_center = fruit_pcd.get_center() + offset

            if superellipsoid_method:
                processed_pcd, fruit_pcd, peduncle_pcd, initial_position, initial_quaternion = \
                    self._preprocess_pcd_for_superellipsoid(rgb_image, depth_image, masks)
                refined_fruit_center, _, _ = self._refine_pose_superellipsoid(processed_pcd, refined_fruit_center, initial_quaternion)
                refined_fruit_center = self._verify_superellipsoid_center(initial_position, refined_fruit_center)

            peduncle_center = peduncle_pcd.get_center()
            axis_vector = peduncle_center - refined_fruit_center
            a_x = np.cross(axis_vector, refined_fruit_center)
            a_x_hat = a_x/ np.linalg.norm(a_x)
            a_z = axis_vector #- (np.dot(axis_vector, a_x_hat)*a_x_hat)
            a_z_hat = a_z/ np.linalg.norm(a_z)
            a_y_hat = np.cross(a_z_hat, a_x_hat)
            
            R_co = np.array([a_x_hat, a_y_hat, a_z_hat]).T
            r = R.from_matrix(R_co)
            quaternion = r.as_quat()  # [x, y, z, w]

            result["peduncle_position"] = peduncle_center
            result["fruit_quaternion"] = quaternion
            result["peduncle_quaternion"] = quaternion
            # result["peduncle_pcd"] = peduncle_pcd
            # result["fruit_pcd"] = fitted_pcd if superellipsoid_method else fruit_pcd


        return result
    
    def _verify_superellipsoid_center(self, initial_position, refined_position):

        if np.linalg.norm(initial_position - refined_position) > 0.05:
            print("Warning: Superellipsoid center deviated significantly from initial estimate.")
            return initial_position

        return refined_position

    def _preprocess_pcd_for_superellipsoid(self, rgb_image, depth_image, masks):
        """
        Preprocesses the point cloud for superellipsoid fitting.
        This includes voxel downsampling, outlier removal
        
        Args:
            rgb_image (np.ndarray): RGB image of size (640, 480, 3)
            depth_image (np.ndarray): Depth image of size (640, 480)
            masks (dict): Masks for the fruit and peduncle

        Returns:
            processed_pcd (o3d.geometry.PointCloud): The processed point cloud
            fruit_pcd (o3d.geometry.PointCloud): The partial point cloud of the pepper.
            peduncle_pcd (o3d.geometry.PointCloud): The partial point cloud of the peduncle.
            initial_position (np.ndarray): Initial [x, y, z] estimate.
            initial_quaternion (np.ndarray): Initial [x, y, z, w] rotation estimate.
        """
        pose = np.eye(4)
        fruit_pcd = self.rgbd_to_pcd(rgb_image, depth_image, masks["fruit_mask"], pose)
        peduncle_pcd = self.rgbd_to_pcd(rgb_image, depth_image, masks["peduncle_mask"], pose)

        initial_position, initial_quaternion, _, _ = self._fine_fruit_pose_estimation(
            rgb_image, depth_image, masks["fruit_mask"], masks["peduncle_mask"]
        )
        

        # Erode the fruit mask to avoid edge points
        kernel = np.ones((5,5),np.uint8)
        processed_mask = cv2.erode(masks["fruit_mask"], kernel, iterations = 3)
        processed_mask = processed_mask.astype(bool).astype(np.uint8)

        processed_pcd = self.rgbd_to_pcd(rgb_image, depth_image, processed_mask, pose)
        # Downsample the point cloud and remove outliers
        processed_pcd = processed_pcd.voxel_down_sample(voxel_size=0.01)
        processed_pcd, _ = processed_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        return processed_pcd, fruit_pcd, peduncle_pcd, initial_position, initial_quaternion

    def _refine_pose_superellipsoid(self, processed_pcd, coarse_position, coarse_quaternion, max_iterations=200):
        """
        Fits a superellipsoid to the partial point cloud using the coarse pose as initialization.
        Uses the Solina cost function for optimization.
        
        Args:
            processed_pcd (o3d.geometry.PointCloud): The partial point cloud of the pepper.
            coarse_position (np.ndarray): Initial [x, y, z] estimate.
            coarse_quaternion (np.ndarray): Initial [x, y, z, w] rotation estimate.
            
        Returns:
            optimized_pose (np.ndarray): 4x4 transformation matrix of the fitted superellipsoid center/orientation.
            shape_params (dict): Dictionary containing {a, b, c, e1, e2}.
        """
        # 1. Prepare Point Cloud Data
        points_world = np.asarray(processed_pcd.points)
        
        # 2. Initial Guesses (x0)
        # Shape: bell peppers are roughly cubic/globular. Start with ~4cm radii.
        # Exponents: Blocky shapes have e < 1. Start with 0.5.
        a_init, b_init, c_init = 0.04, 0.04, 0.05 
        e1_init, e2_init = 0.5, 0.5
        
        # Pose: Use coarse position and convert quaternion to Rotation Vector for optimization (3 params vs 4)
        tx, ty, tz = coarse_position

        try:
            r_obj = R.from_quat(coarse_quaternion)
        except ValueError:
            print("Invalid quaternion provided for coarse orientation. Using default.")
            r_obj = R.from_quat([0, 0, 0, 1])  # Default to no rotation if invalid quaternion
        rot_vec = r_obj.as_rotvec() # [rx, ry, rz]
        
        # x0 vector: [a, b, c, e1, e2, tx, ty, tz, rx, ry, rz]
        x0 = np.array([a_init, b_init, c_init, e1_init, e2_init, tx, ty, tz, rot_vec[0], rot_vec[1], rot_vec[2]])

        # 3. Bounds (Keep shape positive, exponents within reasonable superquadric range)
        # a,b,c > 0.01m, 0.1 < e < 1.9 (convex-ish), translation unconstrained-ish
        lower_bounds = [0.01, 0.01, 0.01, 0.1, 0.1, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
        upper_bounds = [0.15, 0.15, 0.15, 1.9, 1.9, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]


        # Priors
        priors = {
            'center': coarse_position,
            'rotation': r_obj,
            'prior_center_weight': 0.01,
            'prior_rotation_weight': 0.01,
            'prior_scaling_weight': 0.5
        }
        
        # 4. Run Optimization
        res = least_squares(
            self._super_residuals, 
            x0, 
            bounds=(lower_bounds, upper_bounds),
            args=(points_world, priors),
            method='trf', # Trust Region Reflective handles bounds well
            loss='linear', # Standard least squares
            max_nfev=max_iterations,
            verbose=0,
            ftol=1e-15,
            gtol=1e-15,
            xtol=1e-15
        )

        # 5. Extract Results
        opt_params = res.x
        a, b, c, e1, e2 = opt_params[0:5]
        t_opt = opt_params[5:8]
        r_vec_opt = opt_params[8:11]
        
        # 6. Construct Final Pose Matrix
        final_rot = R.from_rotvec(r_vec_opt)
        optimized_pose = np.eye(4)
        optimized_pose[:3, :3] = final_rot.as_matrix()
        optimized_pose[:3, 3] = t_opt

        optimized_position = t_opt
        optimized_quaternion = final_rot.as_quat()

        roll, pitch, yaw = final_rot.as_euler('xyz')
        
        # shape_params = {"a": a, "b": b, "c": c, "e1": e1, "e2": e2}
        parameters = {
            'a': a, 'b': b, 'c': c,
            'e1': e1, 'e2': e2,
            'tx': t_opt[0],
            'ty': t_opt[1],
            'tz': t_opt[2],
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'cost': res.cost,
            'success': res.success
        }
        
        # Debug Visualization
        # superellipsoid_pcd = self.sample_superellipsoid_surface(parameters, num_samples=200)


        return optimized_position, optimized_quaternion, parameters

    def _super_residuals(self, params, points_world, priors):
        """
        Computes the Superellipsoid cost function residuals for the superellipsoid fit.
        Based on 'superellipsoid.h' implementation.
        """
        # Unpack parameters
        a, b, c, e1, e2 = params[0], params[1], params[2], params[3], params[4]
        tx, ty, tz = params[5], params[6], params[7]
        rx, ry, rz = params[8], params[9], params[10]

        prior_center = priors['center']
        prior_rotation = priors['rotation']
        prior_center_weight = priors['prior_center_weight']
        prior_rot_weight = priors['prior_rotation_weight']
        prior_scaling_weight = priors['prior_scaling_weight']
        
        # 1. Transform World Points to Canonical Superellipsoid Frame
        # Translation
        p_centered = points_world - np.array([tx, ty, tz])
        
        # Rotation (Inverse rotation to align points with canonical axes)
        # Note: Scipy rotation apply is P_rotated = R * P. We need P_canonical = R_inv * P_centered.
        # R.inv() corresponds to applying the negative rotation vector.
        r_obj = R.from_rotvec(np.array([rx, ry, rz]))
        p_canonical = r_obj.inv().apply(p_centered)
        
        x = p_canonical[:, 0]
        y = p_canonical[:, 1]
        z = p_canonical[:, 2]

        # 2. Compute Implicit Function F
        # Equation: ((|x|/a)^(2/e2) + (|y|/b)^(2/e2))^(e2/e1) + (|z|/c)^(2/e1)
        
        # Add epsilon to avoid division by zero in gradients if needed, though abs() usually handles it
        term_x = (np.abs(x) / a) ** (2.0 / e2)
        term_y = (np.abs(y) / b) ** (2.0 / e2)
        
        # f1 corresponds to the implicit surface equation value
        f1 = (term_x + term_y) ** (e2 / e1) + (np.abs(z) / c) ** (2.0 / e1)
        
        # 3. Solina Cost Function
        # Source: superellipsoid.h -> CostFunctionType::SOLINA
        # residual = sqrt(a*b*c) * abs(pow(f1, e1/2.) - 1.)
        
        # Note: The Solina metric minimizes the radial difference.
        scale_factor = np.sqrt(a * b * c)
        
        # We return the raw residual vector. least_squares will square and sum them.
        solina_residuals = scale_factor * np.abs(f1**(e1 / 2.0) - 1.0)

        # Regularization terms
        center_residual = prior_center_weight * np.sqrt(
            0.001 + (tx - prior_center[0])**2 + 
            (ty - prior_center[1])**2 + 
            (tz - prior_center[2])**2
        )

        scaling_residual = prior_scaling_weight * np.sqrt(
            0.001 + (a - b)**2 + (b - c)**2 + (c - a)**2
        )
        
        # Rotation regularization
            
        prior_z_axis = prior_rotation.as_matrix()[:, 1]  # Z-axis of prior rotation
        current_rotation = r_obj.inv().as_matrix()[:, 1]  # Z-axis of current rotation
        rotation_residual = prior_rot_weight * np.sqrt(
            0.001 + np.sum((current_rotation - prior_z_axis)**2)
        )

        residuals = np.concatenate([
            solina_residuals,
            [center_residual],
            [scaling_residual],
            [rotation_residual]
        ])
        
        return residuals
    
    def _c_func(self, w, m):
        """Signed cosine power function"""
        return np.sign(np.cos(w)) * np.power(np.abs(np.cos(w)), m)
    
    
    def _s_func(self, w, m):
        """Signed sine power function"""
        return np.sign(np.sin(w)) * np.power(np.abs(np.sin(w)), m)
    
    def sample_superellipsoid_surface(self, parameters, num_samples=1000):
        """
        Sample points on the superellipsoid surface for visualization.
        
        Args:
            parameters (dict): Superellipsoid parameters
            num_samples (int): Number of points to sample
            
        Returns:
            open3d.geometry.PointCloud: Sampled surface points
        """
        a = parameters['a']
        b = parameters['b']
        c = parameters['c']
        e1 = parameters['e1']
        e2 = parameters['e2']
        tx = parameters['tx']
        ty = parameters['ty']
        tz = parameters['tz']
        roll = parameters['roll']
        pitch = parameters['pitch']
        yaw = parameters['yaw']
        
        # Sample using parametric definition
        points = []
        u_samples = np.linspace(-np.pi, np.pi, int(np.sqrt(num_samples)))
        v_samples = np.linspace(-np.pi/2, np.pi/2, int(np.sqrt(num_samples)))
        
        r = 2.0 / e2
        t = 2.0 / e1
        
        rotation_matrix = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        
        for u in u_samples:
            for v in v_samples:
                # Parametric equations with signed power function
                x = a * self._c_func(v, 2.0/t) * self._c_func(u, 2.0/r)
                y = b * self._c_func(v, 2.0/t) * self._s_func(u, 2.0/r)
                z = c * self._s_func(v, 2.0/t)
                
                # Apply rotation and translation
                point = np.array([x, y, z])
                rotated_point = rotation_matrix @ point
                transformed_point = rotated_point + np.array([tx, ty, tz])
                
                points.append(transformed_point)
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        
        return pcd



class Segmentation:
    def __init__(self, weights_path_url, device='cuda'):
        """
        A class to perform instance segmentation using YOLO series of models
        Args: model_cfg (str): Path to model configuration file
              device (str): Device to run inference on (default: 'cuda')
        """
        self.device = device
        weights_path = weights_path_url["model_path"]
        drive_url = weights_path_url["drive_url"]

        if not pathlib.Path(weights_path).exists():
            gdown.download(drive_url, weights_path, quiet=False)

        self.model = YOLO(weights_path)
        self.model.to(self.device)

    def infer(self, rgb_image, confidence=0.8, verbose=True):
        """
        Runs the inference on a single image
        Args: rgb_image (np.ndarray): RGB image of size (640, 480, 3)
              conf (float): confidence threshold
        Returns: results (list): List of results containing bounding box coordinates, class labels, and confidence
        """
        # Load image
        results = self.model.predict(rgb_image, conf=confidence, verbose=verbose)
        
        return results

class SequentialSegmentation:
    def __init__(self, segmentation_models, device='cuda'):
        """
        A class to perform instance segmentation using YOLO series of models
        Args: model_cfg (str): Path to model configuration file
              device (str): Device to run inference on (default: 'cuda')
        """
        self.device = device
        self.model = {}
        self.confidence = {}
        self.peduncle_img_size = 640

        self.crop_fraction = 0.6

        for task in {"fruit", "peduncle"}:
            weights_path = segmentation_models[task]["model_path"]
            drive_url = segmentation_models[task]["drive_url"]
            self.confidence[task] = segmentation_models[task]["confidence"]

            if not pathlib.Path(weights_path).exists():
                gdown.download(drive_url, weights_path, quiet=False)

            self.model[task] = YOLO(weights_path)
            self.model[task].to(self.device)

    @staticmethod
    def predict_combined_masks(model, rgb_image, confidence=0.8, verbose=True):
        """
        Splits a wide image, runs batch inference, combines the masks,
        and then separates each combined object into its own mask and box
        Args:
            model: The segmentation model to use for inference.
            rgb_image (np.ndarray): The input RGB image.
            confidence (float, optional): Confidence threshold for detections. Defaults to 0.8.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
        Returns:
            list: A list of dictionaries, each containing 'mask' and 'box' for detected objects.
        """
        
        # Get original image dimensions
        H, W, _ = rgb_image.shape  # (480, 848, 3)

        # 1. Split left and right images
        left_image = rgb_image[:, :640, :]    # (480, 640, 3)
        right_image = rgb_image[:, -640:, :]  # (480, 640, 3)

        # 2. Run batch inference
        results = model([left_image, right_image], conf=confidence, verbose=verbose)
        fruit_results_left, fruit_results_right = results[0], results[1]

        # 3. Create a single, full-size mask canvas
        combined_mask_canvas = np.zeros((H, W), dtype=np.uint8)

        # 4. Process and "OR" all masks from the LEFT image
        if fruit_results_left.masks is not None:
            masks_left = fruit_results_left.masks.data.cpu().numpy().astype(np.uint8)
            if masks_left.shape[0] > 0:
                # Combine all left masks into one by taking the max
                combined_left_mask = np.max(masks_left, axis=0)
                # Place this on the left side of the canvas
                combined_mask_canvas[:, :640] = combined_left_mask

        # 5. Process and "OR" all masks from the RIGHT image
        if fruit_results_right.masks is not None:
            masks_right = fruit_results_right.masks.data.cpu().numpy().astype(np.uint8)
            if masks_right.shape[0] > 0:
                # Combine all right masks into one
                combined_right_mask = np.max(masks_right, axis=0)
                
                # Use bitwise_or to merge the right mask with the canvas's overlapping region
                combined_mask_canvas[:, -640:] = np.bitwise_or(
                    combined_mask_canvas[:, -640:],
                    combined_right_mask
                )

        # --- NEW LOGIC ---
        
        # 6. Find contours of all distinct objects in the combined canvas
        # cv2.RETR_EXTERNAL finds only the outermost contours
        contours, _ = cv2.findContours(combined_mask_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        object_list = []

        # 7. Iterate over each found contour (each object)
        for contour in contours:
            # Create an empty canvas for this single object
            individual_mask = np.zeros((H, W), dtype=np.uint8)
            
            # Draw the filled contour on the new mask (color=1 for binary)
            cv2.drawContours(individual_mask, [contour], -1, color=1, thickness=cv2.FILLED)
            
            # 8. Get the axis-aligned bounding box from the contour
            x, y, w, h = cv2.boundingRect(contour)
            box = np.array([x, y, x + w, y + h])  # Format: [x1, y1, x2, y2]
            
            # Add the individual mask and its box to the list
            object_list.append({
                "mask": individual_mask,
                "box": box
            })
                
        return object_list


    def infer(self, rgb_image, coarse_only=True, verbose=True):
        """
        Runs the inference on a single image
        Args: rgb_image (np.ndarray): RGB image of size (640, 480, 3)
              coarse_only (bool): If True, only fruit segmentation is performed
        Returns: results (list): List of results containing fruit and peduncle masks (if coarse_only is False)
        """
        masks_dicts = []

        rgb_image = rgb_image[:, 104:744, :] # width crop to 640, will be fixed later

        fruit_results = self.model["fruit"].predict(rgb_image, conf=self.confidence["fruit"], verbose=verbose)
        detections = fruit_results[0]

        height, width, _ = rgb_image.shape

        if detections.masks is not None:

            boxes = detections.boxes.xyxy.cpu().numpy()
            masks = detections.masks.data.cpu().numpy()

            for (box, mask) in zip(boxes, masks):
                mask_dict = {}
                mask_dict["fruit_mask"] = np.pad(mask.astype(np.uint16), ((0, 0), (104, 104)), mode='constant', constant_values=0)

                if not coarse_only:
                    # Get bounding box coordinates (xmin, ymin, xmax, ymax) as integers
                    x1, y1, x2, y2 = box[:4]
                    max_size = max(x2-x1, y2-y1) * self.crop_fraction

                    roi_y_min = max(0, (y1 - max_size//2).astype(np.int32))
                    roi_y_max = min(height, (y2 + max_size//2).astype(np.int32))
                    roi_x_min = max(0, (x1 - max_size//2).astype(np.int32))
                    roi_x_max = min(width, (x2 + max_size//2).astype(np.int32))

                    cropped_image = rgb_image[roi_y_min:roi_y_max, roi_x_min:roi_x_max, :]

                    cropped_image = cv2.resize(cropped_image, (self.peduncle_img_size, self.peduncle_img_size))
                    peduncle_results = self.model["peduncle"].predict(cropped_image, conf=self.confidence["peduncle"], verbose=verbose)

                    peduncle_on_original = np.zeros((height, width), dtype=np.uint8)
                    # get the highest confidence peduncle detection
                    peduncle_detections = peduncle_results[0]
                    if peduncle_detections.masks is not None:
                        # best_peduncle_idx = np.argmax(peduncle_detections.boxes.conf.cpu().numpy())
                        best_peduncle_mask = peduncle_detections.masks.data[0].cpu().numpy().astype(np.uint8)
                        # Resize the mask back to the original cropped image size
                        best_peduncle_mask_resized = cv2.resize(best_peduncle_mask, (roi_x_max - roi_x_min, roi_y_max - roi_y_min), interpolation=cv2.INTER_NEAREST)
                        peduncle_on_original[roi_y_min:roi_y_max, roi_x_min:roi_x_max] = best_peduncle_mask_resized

                        mask_dict["peduncle_mask"] = np.pad(peduncle_on_original.astype(np.uint16), ((0, 0), (104, 104)), mode='constant', constant_values=0)
                masks_dicts.append(mask_dict)

        return masks_dicts
    
    def infer_large_fov(self, rgb_image, coarse_only=True, verbose=True):
        """
        Runs the inference on a single image
        Args: rgb_image (np.ndarray): RGB image of size (640, 480, 3)
              coarse_only (bool): If True, only fruit segmentation is performed
        Returns: results (list): List of results containing fruit and peduncle masks (if coarse_only is False)
        """
        masks_dicts = []

        height, width, _ = rgb_image.shape
        fruit_result_dicts = self.predict_combined_masks(self.model["fruit"], rgb_image, confidence=self.confidence["fruit"], verbose=verbose)


        if len(fruit_result_dicts) > 0:

            for fruit_result in fruit_result_dicts:
                box = fruit_result["box"]
                mask = fruit_result["mask"]
                mask_dict = {}
                mask_dict["fruit_mask"] = mask.astype(np.uint16)

                if not coarse_only:
                    # Get bounding box coordinates (xmin, ymin, xmax, ymax) as integers
                    x1, y1, x2, y2 = box[:4]
                    max_size = max(x2-x1, y2-y1)

                    roi_y_min = max(0, (y1 - max_size//2).astype(np.int32))
                    roi_y_max = min(height, (y2 + max_size//2).astype(np.int32))
                    roi_x_min = max(0, (x1 - max_size//2).astype(np.int32))
                    roi_x_max = min(width, (x2 + max_size//2).astype(np.int32))

                    cropped_image = rgb_image[roi_y_min:roi_y_max, roi_x_min:roi_x_max, :]

                    cropped_image = cv2.resize(cropped_image, (self.peduncle_img_size, self.peduncle_img_size))
                    peduncle_results = self.model["peduncle"].predict(cropped_image, conf=self.confidence["peduncle"], verbose=verbose)

                    peduncle_on_original = np.zeros((height, width), dtype=np.uint8)
                    # get the highest confidence peduncle detection
                    peduncle_detections = peduncle_results[0]
                    if peduncle_detections.masks is not None:
                        # best_peduncle_idx = np.argmax(peduncle_detections.boxes.conf.cpu().numpy())
                        best_peduncle_mask = peduncle_detections.masks.data[0].cpu().numpy().astype(np.uint8)
                        # Resize the mask back to the original cropped image size
                        best_peduncle_mask_resized = cv2.resize(best_peduncle_mask, (roi_x_max - roi_x_min, roi_y_max - roi_y_min), interpolation=cv2.INTER_NEAREST)
                        peduncle_on_original[roi_y_min:roi_y_max, roi_x_min:roi_x_max] = best_peduncle_mask_resized

                        mask_dict["peduncle_mask"] = peduncle_on_original.astype(np.uint16)
                masks_dicts.append(mask_dict)

        return masks_dicts
